# Polymarket Market Metadata Reference

Evidence-based reference for fetching market metadata and filtering for soccer markets. All claims verified via API calls on 2026-04-18.

## 1. Market Identifiers

A Polymarket market has several identifiers:

| Identifier | Format | Example | Notes |
|---|---|---|---|
| **condition_id** | hex string (0x...) | `0x1313a10ff...` | Gold-standard unique ID. Used as key in CLOB API and WS feeds. Immutable. |
| **market_id** | integer | `624454` | Gamma's internal numeric ID. Used in Gamma REST endpoints (`/markets/{id}`). |
| **token_id** | large decimal string | `42181257918...` | One per outcome (YES/NO). Used in WS subscriptions and on-chain. A market has exactly 2 token_ids (`clobTokenIds[0]` = YES, `[1]` = NO). |
| **question_id** | hex string (0x...) | `0x9539e6e2...` | Equals `neg_risk_market_id` for neg-risk markets. Groups related markets. |
| **slug** | kebab-case string | `afc-idn-ksa-2025-10-08-idn` | Human-readable. Not guaranteed unique across time. Called `slug` in Gamma, `market_slug` in CLOB. |

**Relationships:**
- `condition_id` is 1:1 with a market. Everything else can be derived from it.
- `market_id` is also 1:1 but only exists in Gamma (CLOB doesn't return it).
- Each market has exactly 2 `token_id`s (YES and NO).
- Multiple markets share one `question_id` when they're part of the same neg-risk group.

## 2. APIs for Fetching Market Metadata

### 2a. Gamma API (`gamma-api.polymarket.com`)

#### `GET /markets/{market_id}` — Single market by numeric ID
- **Works**: Yes, reliably.
- **Input**: Numeric market_id (e.g., `624454`).
- **Returns**: 82 fields including `conditionId`, `slug`, `question`, `sportsMarketType`, `gameStartTime`, `clobTokenIds`, `gameId`, volume, pricing, resolution status, etc.
- **Closed markets**: Works for closed/resolved markets.
- **Condition_id lookup**: Does NOT accept condition_ids. Passing a hex string returns 422.

#### `GET /markets?...` — Paginated market list
- **Working filters**: `closed` (required: `"true"` or `"false"`), `limit`, `offset`, `slug`, `id`
- **Broken filters** (silently ignored): `condition_id`, `conditionId`, `tag_slug`, `sportsMarketType`
- **Default (no `closed` param)**: Returns only open markets (~50K).
- **`closed=true`**: Returns ~250K closed markets. Offset caps out around 250K (422 at higher offsets).
- **`slug` filter**: Works but only returns results when `closed` matches the market's state. A closed market with `?slug=X` returns 0 unless `&closed=true` is also specified.
- **Fields**: Same 82 fields as `/markets/{id}`, plus `events` array with nested event objects.
- **Use case**: Bulk enumeration when you need all markets. NOT useful for lookup by condition_id.

#### `GET /events?...` — Paginated event list with nested markets
- **Working filters**: `tag_slug`, `series_id`, `slug`, `id`, `closed`, `active`, `limit`, `offset`
- **Returns**: Array of event objects, each with `markets` array. Nested market objects have the same 82 fields as `/markets/{id}` — no information loss.
- **Key advantage**: Supports `tag_slug` and `series_id` filtering (unlike `/markets`).
- **Use case**: Category-based discovery (e.g., all soccer markets).

#### `GET /markets/{id}` vs nested in `/events`
Both return identical 82-field market objects. There is no need to call `/markets/{id}` separately if you already have the market from `/events`.

### 2b. CLOB API (`clob.polymarket.com`)

#### `GET /markets/{condition_id}` — Single market by condition_id
- **Works**: Yes, reliably. The only API that accepts condition_id as a lookup key.
- **Input**: Hex condition_id string.
- **Returns**: 29 fields. Snake_case naming (vs Gamma's camelCase).
- **Closed markets**: Works for closed markets, including very old ones.
- **Key exclusive fields**: `tags` (string array, e.g. `["Sports", "Soccer", "Games", "AFC"]`), `tokens` (array with `token_id` and `outcome`).
- **Missing vs Gamma**: No `sportsMarketType`, no numeric `id` (market_id), no `events`, no volume data, no resolution details.

### 2c. Field Comparison: Gamma vs CLOB

| Field | Gamma | CLOB | Notes |
|---|---|---|---|
| condition_id | `conditionId` | `condition_id` | Both have it |
| slug | `slug` | `market_slug` | Same value, different key name |
| question | `question` | `question` | Same |
| game start time | `gameStartTime` | `game_start_time` | Same value, different format (`+00` vs `Z`) |
| sport market type | `sportsMarketType` | *missing* | **Gamma-only**. Values: `moneyline`, `spreads`, `totals`, `both_teams_to_score`, `soccer_exact_score`, `soccer_anytime_goalscorer`, `total_corners`, etc. |
| tags | *missing* | `tags` | **CLOB-only**. String array like `["Sports", "Soccer", "Games"]` |
| numeric market_id | `id` | *missing* | **Gamma-only** |
| token_ids | `clobTokenIds` | `tokens[].token_id` | Both have it, different structure |
| neg_risk | `negRisk` | `neg_risk` | Both |
| events/series | `events` (nested) | *missing* | **Gamma-only** |

### 2d. Recommended Lookup Workflows

**Given a numeric market_id:**
→ `GET gamma-api.polymarket.com/markets/{market_id}` — returns everything.

**Given a condition_id:**
→ `GET clob.polymarket.com/markets/{condition_id}` — always works, but missing `sportsMarketType` and numeric `id`.
→ There is NO Gamma endpoint that accepts a condition_id as a filter. The `?conditionId=` param is silently ignored.
→ If you also need `sportsMarketType`: use CLOB to get the `market_slug`, then `GET gamma-api.polymarket.com/markets?slug={slug}&closed=true` (or `closed=false`). The slug filter works but requires knowing the market's open/closed state.

**Given a slug:**
→ `GET gamma-api.polymarket.com/markets?slug={slug}&closed=true` (or `false`) — returns 0 or 1 results. Must specify correct `closed` value.
→ `GET gamma-api.polymarket.com/events?slug={event_slug}` — works for event-level slugs (without the market-specific suffix).

**Bulk enumeration of all markets:**
→ Paginate `GET gamma-api.polymarket.com/markets?closed=false&limit=500&offset=N` (~50K open)
→ Paginate `GET gamma-api.polymarket.com/markets?closed=true&limit=500&offset=N` (~250K closed, caps at ~250K offset)
→ Total: ~300K markets. Takes ~5 minutes.
→ For condition_ids not found in this set, fall back to CLOB (one request per cid).

## 3. Soccer Market Filtering

### 3a. Available Methods

#### Method 1: `tag_slug=soccer` on `/events`
```
GET gamma-api.polymarket.com/events?tag_slug=soccer&closed={true,false}&limit=500&offset=N
```
- **Coverage**: 18,764 events, 97,866 condition_ids (as of 2026-04-18).
- **What it catches**: Per-match games across all leagues, plus outrights (league winners, top scorers, relegation, Ballon d'Or, World Cup winner, etc.).
- **What it misses**: 596 events (1,802 condition_ids) that have a soccer series_id but are NOT tagged soccer. These are real per-match EPL games — Polymarket simply forgot to tag them.
- **False positives**: None observed — everything tagged soccer is soccer-related.

#### Method 2: `series_id=X` on `/events` (79 known soccer series)
```
GET gamma-api.polymarket.com/events?series_id={sid}&closed={true,false}&limit=500&offset=N
```
- **Coverage**: 18,194 events, 92,087 condition_ids.
- **What it catches**: Per-match games from known leagues. Very few outrights (most have no series).
- **What it misses**: 1,166 events (7,581 condition_ids) that are tagged soccer but have NO series. These include Nations League, friendlies, AFCON, Intercontinental Cup, and all outrights.
- **False positives**: None observed.

#### Method 3: `sportsMarketType` field
- **Values**: `moneyline`, `spreads`, `totals`, `both_teams_to_score`, `soccer_exact_score`, `soccer_anytime_goalscorer`, `soccer_halftime_result`, `total_corners`
- **Coverage**: Present on per-match markets from series-tagged events. Empty (`""`) on: all outrights/futures, all non-series events, all early-era events (before smt was introduced), and all NBA/NFL/other sports.
- **Soccer-exclusive?** In practice yes — no non-soccer market was observed with a non-empty `sportsMarketType`. But this is not explicitly documented and could change.
- **Use case**: Good for filtering WITHIN a known-soccer event to identify market type (moneyline vs spread vs totals vs btts). Not reliable for identifying whether an arbitrary market is soccer.

#### Method 4: CLOB `tags` field
```
GET clob.polymarket.com/markets/{condition_id} → tags: ["Sports", "Soccer", ...]
```
- **Coverage**: Every market has tags. Soccer markets have "Soccer" tag. NBA has "Basketball"/"NBA", NHL has "Hockey"/"NHL", etc.
- **Advantage**: Works on any condition_id, including those not in Gamma's paginated results.
- **Disadvantage**: One request per condition_id (no bulk endpoint). Slow for large sets.
- **Reliability**: Tags appear consistent across all tested markets. "Soccer" tag was present on all soccer markets tested.

### 3b. Coverage Comparison

| Method | Events | Condition IDs |
|---|---|---|
| `tag_slug=soccer` | 18,764 | 97,866 |
| `series_id` (79 IDs) | 18,194 | 92,087 |
| **Union of both** | **19,360** | **99,668** |
| tag only (no series) | 1,166 | 7,581 |
| series only (no tag) | 596 | 1,802 |

Neither method is a superset of the other.

### 3c. What's in each exclusive set?

**Tag-only (1,166 events)**: Events with `soccer` tag but no series assignment.
- Outrights: "Premier League Winner", "Ballon d'Or", "World Cup Winner", relegation, top scorers
- International: Nations League, FIFA Friendlies, AFCON, Intercontinental Cup
- Pattern: these events have `series: []` (empty) and `sportsMarketType: ""` (empty)

**Series-only (596 events)**: Events with a soccer series but missing the `soccer` tag.
- All are real per-match games (e.g., EPL matchday games)
- Pattern: these have a valid `series` and `sportsMarketType` but Polymarket forgot to add the tag

### 3d. Recommended Soccer Filtering Workflow

**For maximum coverage (all soccer markets):**
1. Query `/events?tag_slug=soccer` (both `closed=true` and `closed=false`)
2. Query `/events?series_id=X` for each of the 79 `SOCCER_SERIES_IDS`
3. Union the results, dedup by event ID
4. This yields ~99,668 condition_ids across ~19,360 events

**For per-match games only (filtering by market type):**
Use `sportsMarketType` when available (Gamma). When not available (CLOB, old markets with empty sportsMarketType), classify from the `question` field using these regex patterns (ordered — first match wins):

| Type | Pattern | Example |
|---|---|---|
| `moneyline` | `Will .+ win on \d{4}-\d{2}-\d{2}\?$` | "Will Arsenal FC win on 2026-03-01?" |
| `moneyline` | `end in a draw\?$` | "Will Arsenal FC vs. Chelsea FC end in a draw?" |
| `spreads` | `^Spread: .+ \(-[\d.]+\)$` | "Spread: Arsenal FC (-1.5)" |
| `total_corners` | `Total Corners$` | "Arsenal FC vs. Chelsea FC: O/U 9.5 Total Corners" |
| `totals` | `O/U [\d.]+` | "Arsenal FC vs. Chelsea FC: O/U 2.5" |
| `both_teams_to_score` | `Both Teams to Score$` | "Arsenal FC vs. Chelsea FC: Both Teams to Score" |
| `soccer_exact_score` | `^Exact Score:` | "Exact Score: Arsenal FC 1 - 1 Chelsea FC?" |
| `soccer_halftime_result` | `at halftime\?$` | "Arsenal FC leading at halftime?" |
| `soccer_anytime_goalscorer` | `Anytime Goalscorer$` | "Leandro Trossard: Anytime Goalscorer" |

**Important:** `total_corners` must be checked before `totals` (both contain "O/U").

Validated on 59,382 markets with known `sportsMarketType`: 100.0% accuracy, 0 misclassifications, 0 unclassified. Also validated: 0% false positives on 1,059 soccer outrights (all correctly rejected). The `question` field is identical between Gamma and CLOB (verified on 40 markets).

Note: these patterns also match non-soccer per-match markets (NBA, NFL use the same question format). The question classifier determines market *type*, not *sport*. Always combine with soccer identification (tag/series or CLOB "Soccer" tag) first.

**For unknown condition_ids (e.g., from pmxt parquets):**
1. Check against the union lookup built above
2. For unmatched: query CLOB API, check if `"Soccer"` is in `tags`
3. CLOB tags are reliable for sport identification but don't provide `sportsMarketType`
4. **Do NOT use `"football"` as a soccer tag** — it matches American football (NFL). Only use `"Soccer"` (capital S).
5. Classify market type from `question` using the regex patterns above

### 3e. Complete-set games (12 markets)
A "complete" soccer game for our purposes has exactly 12 markets:
- 3 moneyline (Team A win, Draw, Team B win)
- 4 spread (Team A -1.5, Team B -1.5, Team A -2.5, Team B -2.5)
- 4 totals (O/U 1.5, 2.5, 3.5, 4.5)
- 1 btts (Both Teams to Score)

These are split across two events:
- Main event (has `gameId`): contains 3 moneyline markets
- `-more-markets` event (`gameId` is null): contains 4 spread + 4 totals + 1 btts
- Both share the same slug prefix (e.g., `epl-che-mun-2026-04-18` and `epl-che-mun-2026-04-18-more-markets`)
- `gameId` must be propagated from the main event to the more-markets event via the shared game slug

### 3f. Series ID maintenance
The `SOCCER_SERIES_IDS` list in `src/polymarket/soccer.py` should be periodically refreshed. New leagues are added by Polymarket over time. To discover new series:
```
GET gamma-api.polymarket.com/events?tag_slug=soccer&closed=false&limit=500
```
Each event has a `series` array with `id` and `title`. Collect all unique series IDs and diff against the current list.
