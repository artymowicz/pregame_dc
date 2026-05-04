"""Soccer market constants and utilities for Polymarket.

Canonical location for soccer series IDs, market type mappings, and slug
parsing. Imported by pipelines, strategies, and scripts.
"""

import re

# Series IDs for all soccer leagues on Polymarket.
# Maintained by querying: GET gamma-api.polymarket.com/events?tag_slug=soccer
# and extracting unique series[].id values. See docs/polymarket_market_metadata.md.
SOCCER_SERIES_IDS = [
    # Domestic top flights
    36,     # EPL (parent tag-series; distinct from 10188 Premier League 2025)
    10188,  # Premier League 2025
    10193,  # La Liga 2025
    10194,  # Bundesliga 2025
    10195,  # Ligue 1 2025
    10203,  # Serie A 2025
    10189,  # MLS 2025
    10286,  # Eredivisie 2025
    10290,  # Liga MX 2025
    10292,  # Süper Lig 2025
    10312,  # Primera División Argentina
    10313,  # Russian Premier League
    10330,  # Primeira Liga (Portugal)
    10355,  # EFL Championship
    10359,  # Brazil Serie A
    10360,  # Japan J League
    10361,  # Saudi Professional League
    10362,  # Norway Eliteserien
    10363,  # Denmark Superliga
    10364,  # Indian Super League
    10438,  # A League Soccer
    10439,  # Chinese Super League
    10443,  # Japan J2 League
    10444,  # K-league
    10674,  # Scottish Premiership
    10964,  # Primera A (Colombia)
    10965,  # Primera Division (Chile)
    10966,  # Bolivia 1
    10967,  # Liga 1 (Peru)
    10968,  # Morocco 1
    10969,  # Egypt 1
    10970,  # Czechia 1
    10971,  # Romania 1
    11241,  # Ukraine Premier Liha
    11438,  # Liga Nacional Guatemala
    11451,  # Prva Liga
    11452,  # Nike Liga
    11462,  # NWSL
    # Second divisions
    10230,  # EFL
    10670,  # Bundesliga 2
    10672,  # La Liga 2
    10675,  # Ligue 2
    10676,  # Serie B
    10973,  # Brazil Serie B
    11435,  # League One
    11436,  # League Two
    11445,  # National League
    11434,  # Liga Promerica
    # Domestic cups
    10287,  # Coppa Italia
    10314,  # FA Cup
    10315,  # Coupe de France
    10316,  # Copa del Rey
    10317,  # DFB-Pokal
    10329,  # EFL Cup
    10863,  # Spanish Super Cup
    11453,  # TFF Super Kupa
    11454,  # Taca de Portugal
    11455,  # OFB Cup
    11457,  # Greek Cup
    11458,  # Scottish Cup
    11459,  # KNVB Beker
    11460,  # Copa do Brasil
    # UEFA club competitions
    10204,  # UEFA Champions League
    10209,  # UEFA Europa League
    10437,  # Europa Conference League
    11240,  # Women's Champions League
    10037,  # UEL (legacy)
    # International / qualifiers / continental
    10238,  # FIFA Friendly
    10240,  # CAF
    10241,  # AFC
    10243,  # UEFA Qualifiers
    10244,  # Concacaf
    10246,  # Conmebol
    10289,  # Copa Libertadores
    10291,  # Copa Sudamericana
    11433,  # FIFA World Cup
    11446,  # UEFA Nations League
    11463,  # WCQ Inter-Confederation Playoffs
    11464,  # CONCACAF Champions Cup
]

# Maps Gamma's sportsMarketType values to our canonical type names.
SPORTS_TYPE_MAP = {
    "moneyline": "moneyline",
    "spreads": "spread",
    "totals": "totals",
    "both_teams_to_score": "btts",
}


def parse_game_slug(event_slug):
    """Derive game slug by stripping date suffix and -more-markets.

    'epl-che-mun-2026-04-18'              -> 'epl-che-mun'
    'epl-che-mun-2026-04-18-more-markets' -> 'epl-che-mun'
    'epl-che-mun-2026-04-18-spread-home-1pt5' -> 'epl-che-mun'
    """
    slug = re.sub(r"-more-markets$", "", event_slug)
    slug = re.sub(r"-\d{4}-\d{2}-\d{2}.*$", "", slug)
    return slug


def parse_game_date_slug(event_slug):
    """Derive dated game slug: teams + date, without market-type suffixes.

    'epl-che-mun-2026-04-18'                   -> 'epl-che-mun-2026-04-18'
    'epl-che-mun-2026-04-18-more-markets'      -> 'epl-che-mun-2026-04-18'
    'epl-che-mun-2026-04-18-spread-home-1pt5'  -> 'epl-che-mun-2026-04-18'
    'epl-che-mun-2026-04-18-idn'               -> 'epl-che-mun-2026-04-18'
    """
    slug = re.sub(r"-more-markets$", "", event_slug)
    # Keep through the YYYY-MM-DD, strip everything after
    m = re.match(r"(.+-\d{4}-\d{2}-\d{2})", slug)
    if m:
        return m.group(1)
    return slug


# Question-based market type classifier.
# See docs/polymarket_market_metadata.md for validation details.
# Order matters — total_corners must come before totals.
_SMT_PATTERNS = [
    ("moneyline", re.compile(r"Will .+ win on \d{4}-\d{2}-\d{2}\?$")),
    ("moneyline", re.compile(r"end in a draw\?$")),
    ("spreads", re.compile(r"^Spread: .+ \(-[\d.]+\)$")),
    ("total_corners", re.compile(r"Total Corners$")),
    ("totals", re.compile(r"O/U [\d.]+")),
    ("both_teams_to_score", re.compile(r"Both Teams to Score$")),
    ("soccer_exact_score", re.compile(r"^Exact Score:")),
    ("soccer_halftime_result", re.compile(r"(at halftime\?$|Draw at halftime\?$)")),
    ("soccer_anytime_goalscorer", re.compile(r"Anytime Goalscorer$")),
]


def classify_market_type(question):
    """Classify a soccer market's type from its question text.

    Returns the sportsMarketType string, or None if unclassifiable (e.g., outright).
    """
    for smt, pat in _SMT_PATTERNS:
        if pat.search(question):
            return smt
    return None
