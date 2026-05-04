"""Async download client for telonex.io.

Auth: `Authorization: Bearer $TELONEX_API_KEY`. URL pattern verified via live
probe on 2026-04-20 — `downloads` is plural:
    GET https://api.telonex.io/v1/downloads/polymarket/{channel}/{YYYY-MM-DD}?asset_id=X
Returns 302 to a signed Cloudflare R2 URL (15-min expiry). httpx follows the
redirect when `follow_redirects=True` on the client.
"""

import asyncio

import httpx

TELONEX_API_BASE = "https://api.telonex.io/v1"


class TelonexError(RuntimeError):
    pass


async def download_quotes(client, api_key, date, asset_id, out_path,
                          sem=None, max_retries=4):
    """Download one telonex `quotes` parquet. Returns file size in bytes.

    Retries on 429 with exponential backoff. Raises TelonexError on other errors.
    """
    url = f"{TELONEX_API_BASE}/downloads/polymarket/quotes/{date}?asset_id={asset_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async def _do_request():
        r = await client.get(url, headers=headers)
        return r

    acquired = False
    try:
        if sem is not None:
            await sem.acquire()
            acquired = True

        for attempt in range(max_retries):
            try:
                r = await _do_request()
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise TelonexError(f"transport error after {max_retries} tries: {e}")

            if r.status_code == 200:
                out_path.write_bytes(r.content)
                return len(r.content)
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                raise TelonexError(f"404 not found (asset_id={asset_id[:16]}... date={date})")
            raise TelonexError(f"HTTP {r.status_code}: {r.text[:200]}")

        raise TelonexError(f"exhausted {max_retries} retries (likely 429)")
    finally:
        if acquired:
            sem.release()
