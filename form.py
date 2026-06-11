"""
Gemeinsame Form-Berechnung für Training (prepare_data.py) und Serving (predict.py).
Single Source of Truth → strukturelle Vermeidung von Training-Serving-Skew.

Phase A: einfacher Mittelwert.
Phase B: Spielart-Gewichtung via tournament_weight().
Phase C: Gegner-Gewichtung via opponent_factor_*() — KORRIGIERT.
"""

import numpy as np

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "Confederations Cup": 1.0,
    "UEFA Euro": 1.0,
    "Copa América": 1.0,
    "African Cup of Nations": 1.0,
    "AFC Asian Cup": 1.0,
    "Gold Cup": 1.0,
    "OFC Nations Cup": 1.0,
    "FIFA World Cup qualification": 0.75,
    "UEFA Euro qualification": 0.75,
    "Copa América qualification": 0.75,
    "African Cup of Nations qualification": 0.75,
    "AFC Asian Cup qualification": 0.75,
    "UEFA Nations League": 0.75,
    "CONCACAF Nations League": 0.75,
    "Friendly": 0.4,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.5
WINDOW_SIZE = 5
FALLBACK_FORM = 1.3

# Cap-Grenzen für den Gegner-Faktor (verhindert Extremwerte bei Außenseitern)
OPPONENT_FACTOR_MIN = 0.7
OPPONENT_FACTOR_MAX = 1.5


def tournament_weight(tournament) -> float:
    """Spielart-Gewicht. NaN/Unbekannt → DEFAULT_TOURNAMENT_WEIGHT."""
    if not isinstance(tournament, str):
        return DEFAULT_TOURNAMENT_WEIGHT
    return TOURNAMENT_WEIGHTS.get(tournament, DEFAULT_TOURNAMENT_WEIGHT)


def opponent_factor_for_scored(opponent_mv, median_mv: float) -> float:
    """Faktor für GESCHOSSENE Tore.
    Tore gegen schwache Gegner → runterskalieren (Faktor < 1).
    Tore gegen starke Gegner → hochskalieren (Faktor > 1).
    Faktor = opponent_mv / median, gecappt auf [0.5, 2.0].

    Beispiele bei median=229M:
      - 6:0 gegen Liechtenstein (5M): 5/229=0.02 → cap 0.5 → 3 'echte' Tore
      - 1:0 gegen Frankreich (1350M): 1350/229=5.9 → cap 2.0 → 2 'echte' Tore
    """
    if opponent_mv is None or not (opponent_mv > 0) or not (median_mv > 0):
        return 1.0
    return float(np.clip(opponent_mv / median_mv, OPPONENT_FACTOR_MIN, OPPONENT_FACTOR_MAX))


def opponent_factor_for_conceded(opponent_mv, median_mv: float) -> float:
    """Faktor für KASSIERTE Tore.
    Gegentore gegen Starke → milder bewerten (Faktor < 1).
    Gegentore gegen Schwache → härter bewerten (Faktor > 1).
    Faktor = median / opponent_mv, gecappt auf [0.5, 2.0].

    Beispiele bei median=229M:
      - 0:2 gegen Frankreich (1350M): 229/1350=0.17 → cap 0.5 → 1 'echtes' Gegentor
      - 0:2 gegen Liechtenstein (5M): 229/5=45.8 → cap 2.0 → 4 'echte' Gegentore
    """
    if opponent_mv is None or not (opponent_mv > 0) or not (median_mv > 0):
        return 1.0
    return float(np.clip(median_mv / opponent_mv, OPPONENT_FACTOR_MIN, OPPONENT_FACTOR_MAX))


def compute_form(
    scored: list,
    conceded: list,
    tournaments: list | None = None,
    opponent_market_values: list | None = None,
    median_market_value: float | None = None,
) -> tuple[float, float]:
    """Liefert (form_attack, form_defense) aus den letzten WINDOW_SIZE Spielen.

    Kombiniert:
      - Spielart-Gewichtung (Friendlies zählen weniger als Pflichtspiele)
      - Gegner-Gewichtung (Tore gegen Schwache werden runterskaliert)

    Beide sind optional & defensiv: fehlende Inputs → Phase reaktiviert sich automatisch
    auf einfachen Mittelwert (Backwards-compatible).
    """
    if not scored:
        return FALLBACK_FORM, FALLBACK_FORM

    last_scored = np.array(scored[-WINDOW_SIZE:], dtype=float)
    last_conceded = np.array(conceded[-WINDOW_SIZE:], dtype=float)
    n = len(last_scored)

    # --- Gegner-Faktor (Phase C) ---
    if (
        opponent_market_values is not None
        and len(opponent_market_values) == len(scored)
        and median_market_value is not None
        and median_market_value > 0
    ):
        last_opp_mv = opponent_market_values[-WINDOW_SIZE:]
        scored_factors = np.array(
            [opponent_factor_for_scored(mv, median_market_value) for mv in last_opp_mv],
            dtype=float,
        )
        conceded_factors = np.array(
            [opponent_factor_for_conceded(mv, median_market_value) for mv in last_opp_mv],
            dtype=float,
        )
        last_scored = last_scored * scored_factors
        last_conceded = last_conceded * conceded_factors

    # --- Spielart-Gewichte (Phase B) ---
    if tournaments is not None and len(tournaments) == len(scored):
        last_tournaments = tournaments[-WINDOW_SIZE:]
        weights = np.array([tournament_weight(t) for t in last_tournaments], dtype=float)
    else:
        weights = np.ones(n, dtype=float)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        return float(last_scored.mean()), float(last_conceded.mean())

    form_attack = float((last_scored * weights).sum() / weight_sum)
    form_defense = float((last_conceded * weights).sum() / weight_sum)

    return form_attack, form_defense