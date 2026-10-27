"""Ehrliche Auswertung: Gruppenphase nach Tippspiel-Schema, K.o.-Phase nur als Weiterkommens-Tipp."""
import numpy as np
import pandas as pd

m = pd.read_csv("data/processed/tippspiel_auswertung.csv")


def pts(th, ta, ah, aa):
    if th == ah and ta == aa:
        return 4
    if th == ta:
        return 2 if ah == aa else 0
    if ah == aa or (th > ta) != (ah > aa):
        return 0
    return 3 if th - ta == ah - aa else 2


g = m[m.stage == "GROUP_STAGE"].copy()
k = m[m.stage != "GROUP_STAGE"].copy()
rng = np.random.default_rng(0)


def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return np.percentile(b, [2.5, 97.5])


print(f"GRUPPENPHASE: {len(g)} Spiele, {g.points.sum()} P, Ø {g.points.mean():.2f}, 95%-KI {ci(g.points.values).round(2)}")
print(g.points.value_counts().sort_index().to_dict(), f"Tendenz {g.tendency_ok.mean():.1%}, exakt {(g.points==4).mean():.1%}")
print("getippte Remis:", g.pred_draw.sum(), "| echte Remis:", g.act_draw.sum())

# Baselines auf denselben Spielen
for name, (a, b) in {"immer 1:1": (1, 1), "immer 2:1 Heim": (2, 1), "immer 1:0 Heim": (1, 0)}.items():
    p = [pts(a, b, x, y) for x, y in zip(g.home_score, g.away_score)]
    print(f"Baseline {name}: {sum(p)} P (Ø {np.mean(p):.2f})")
fav = np.where(g.home_market_value >= g.away_market_value, 1, 0)
p = [pts(2, 1, x, y) if f else pts(1, 2, x, y) for f, x, y in zip(fav, g.home_score, g.away_score)]
print(f"Baseline Marktwert-Favorit 2:1: {sum(p)} P (Ø {np.mean(p):.2f})")
# gleiche Quote Remis wie Engine ist nicht nötig; Anteil Heimsiege/Remis/Auswärts im Turnier
print("Gruppenphase real: Heim", (g.home_score > g.away_score).mean().round(2), "Remis", g.act_draw.mean().round(2), "Ausw.", (g.home_score < g.away_score).mean().round(2))

# K.o.: Weiterkommen
print("\nK.O.-PHASE:", len(k), "Spiele; predicted_winner vs. Endstand (inkl. Verl./Elfm.)")
k["act_w"] = np.where(k.home_score > k.away_score, "home", "away")
print(k.predicted_winner.value_counts(dropna=False).to_dict())
ok = (k.predicted_winner == k.act_w)
print(f"Weiterkommen richtig: {ok.sum()}/{len(k)} = {ok.mean():.1%}; Heim-Baseline: {(k.act_w=='home').mean():.1%}")
fav = np.where(k.home_market_value >= k.away_market_value, "home", "away")
print(f"Marktwert-Favorit: {(fav==k.act_w).mean():.1%}")
print("Remis-Tipps in K.o.:", (k.tipp_home == k.tipp_away).sum(), "von", len(k), "| 90-Min-Tipp kann Verl./Elfm. nicht treffen")
