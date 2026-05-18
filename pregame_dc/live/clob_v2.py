"""V2-aware order builder for Polymarket CLOB after the April 2026 migration.

`py_clob_client v0.34.6` (current latest) still signs the v1 Order schema,
so every `client.post_order(...)` returns
    PolyApiException[status_code=400, error_message={'error': 'order_version_mismatch'}]

This module reproduces just the signing piece against the v2 schema, while
reusing `py_clob_client.client.ClobClient.post_order` for auth headers and
HTTP delivery — the latter only needs a "signed-order-shaped" object whose
`.dict()` returns the JSON payload (see `utilities.order_to_json`).

V2 changes captured here (from contract source at
0xE111180000d2663C0091e4f400237545B87B996B):
  - Removed:  taker, nonce, feeRateBps
  - Added:    timestamp (uint256, ms since epoch), metadata (bytes32),
              builder (bytes32)
  - Domain version bumped from "1" to "2"
  - Verifying contract is the new exchange (or its neg-risk twin)
  - Order field reordering — see `Order` class below

Once py_clob_client publishes a v2-aware release this module can be removed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from eth_utils import keccak
from poly_eip712_structs import Address, Bytes, EIP712Struct, Uint, make_domain
from py_order_utils.utils import generate_seed, normalize_address, prepend_zx

# ---- on-chain constants ----------------------------------------------------

V2_EXCHANGE = "0xE111180000d2663C0091e4f400237545B87B996B"
V2_NEG_RISK_EXCHANGE = "0xe2222d279d744050d28e00520010520000310F59"
V2_DOMAIN_NAME = "Polymarket CTF Exchange"
V2_DOMAIN_VERSION = "2"

ZERO_BYTES32 = b"\x00" * 32

# Side constants in the order struct.
BUY_INT, SELL_INT = 0, 1


class Order(EIP712Struct):
    """V2 Order EIP-712 struct.

    Verified against on-chain typehash
        0xbb86318a2138f5fa8ae32fbe8e659f8fcf13cc6ae4014a707893055433818589
    """
    salt = Uint(256)
    maker = Address()
    signer = Address()
    tokenId = Uint(256)
    makerAmount = Uint(256)
    takerAmount = Uint(256)
    side = Uint(8)
    signatureType = Uint(8)
    timestamp = Uint(256)
    metadata = Bytes(32)
    builder = Bytes(32)


@dataclass
class SignedOrderV2:
    """Lookalike for `py_order_utils.model.order.SignedOrder`.

    `py_clob_client.utilities.order_to_json` only calls `.dict()` on the
    object it's given, so any class providing this method is sufficient.
    """
    payload: dict

    def dict(self):
        return self.payload


def _make_v2_domain(chain_id: int, neg_risk: bool):
    exchange = V2_NEG_RISK_EXCHANGE if neg_risk else V2_EXCHANGE
    return make_domain(
        name=V2_DOMAIN_NAME,
        version=V2_DOMAIN_VERSION,
        chainId=str(chain_id),
        verifyingContract=exchange,
    )


def build_signed_order_v2(
    *,
    client,
    order_args,             # py_clob_client.clob_types.OrderArgs
    sig_type: int,          # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
    metadata: bytes = ZERO_BYTES32,
    builder: bytes = ZERO_BYTES32,
) -> SignedOrderV2:
    """Compute amounts, sign the v2 Order, return a SignedOrder-shaped wrapper.

    Reuses `client.builder` to compute makerAmount/takerAmount with the
    SDK's tested rounding logic, and reuses `client.signer` (and
    `client.builder.funder`) for signing.
    """
    from py_order_utils.signer import Signer as UtilsSigner
    from py_clob_client.order_builder.builder import ROUNDING_CONFIG

    # 1. Resolve tick size & neg-risk via the live SDK (these are token-level).
    tick_size = client._ClobClient__resolve_tick_size(order_args.token_id, None)
    neg_risk = client.get_neg_risk(order_args.token_id)
    round_config = ROUNDING_CONFIG[tick_size]

    # 2. Compute amounts using the SDK's helper (battle-tested rounding).
    side_int_v1, maker_amount, taker_amount = client.builder.get_order_amounts(
        order_args.side, order_args.size, order_args.price, round_config,
    )
    # py_order_utils encodes BUY=0, SELL=1; we use the same here.
    side_int = BUY_INT if side_int_v1 == 0 else SELL_INT

    # 3. Build the EIP-712 struct.
    salt = int(generate_seed())
    timestamp_ms = int(time.time() * 1000)
    funder = client.builder.funder
    signer_addr = client.signer.address()

    order_struct = Order(
        salt=salt,
        maker=normalize_address(funder),
        signer=normalize_address(signer_addr),
        tokenId=int(order_args.token_id),
        makerAmount=int(maker_amount),
        takerAmount=int(taker_amount),
        side=side_int,
        signatureType=int(sig_type),
        timestamp=timestamp_ms,
        metadata=metadata,
        builder=builder,
    )

    # 4. Sign against the v2 domain.
    domain = _make_v2_domain(client.signer.get_chain_id(), neg_risk)
    struct_hash = prepend_zx(keccak(order_struct.signable_bytes(domain=domain)).hex())
    signature = prepend_zx(client.signer.sign(struct_hash))

    # 5. Build the JSON payload that the v2 /order endpoint expects.
    # Field-type conventions mirror py_order_utils.model.order.SignedOrder.dict():
    #   - "amount-like" fields (tokenId/makerAmount/takerAmount) are STRINGS
    #     because they may overflow JS Number precision (uint256).
    #   - salt is left as an INT in v1 (despite also being uint256). Mirror that.
    #   - signatureType is INT in v1.
    #   - side is the string "BUY"/"SELL".
    # timestamp is uint256 like the amount fields, so stringify it to be safe.
    payload = {
        "salt": int(salt),
        "maker": normalize_address(funder),
        "signer": normalize_address(signer_addr),
        "tokenId": str(int(order_args.token_id)),
        "makerAmount": str(int(maker_amount)),
        "takerAmount": str(int(taker_amount)),
        "side": "BUY" if side_int == BUY_INT else "SELL",
        "signatureType": int(sig_type),
        "timestamp": str(timestamp_ms),
        "metadata": "0x" + metadata.hex(),
        "builder": "0x" + builder.hex(),
        "signature": signature,
    }
    return SignedOrderV2(payload=payload)
