"""Bewertet die Engine-Tipps mit dem Tippspiel-Schema und erzeugt Plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "plots"

h = pd.read_csv("data/processed/prediction_history.csv")
r = pd.read_csv("data/raw/wm2026_live.csv")
r = r[r.status == "FINISHED"].copy()
r["kickoff"] = pd.to_datetime(r.date, utc=True)
r["match_key"] = r.kickoff.dt.strftime("%Y-%m-%d") + "_" + r.home_team + "_" + r.away_team

# Letzte Vorhersage vor Anpfiff pro Spiel (Datum der Historie = lokales Datum, daher Fallback auf match_key via Team-Paar)
h["predicted_at"] = pd.to_datetime(h.predicted_at, utc=True)
h["pair"] = h.home_team + "_" + h.away_team
r["pair"] = r.home_team + "_" + r.away_team
m = r.merge(h.drop(columns=["stage", "home_team", "away_team", "date", "match_key"]), on="pair", how="left")
n_missing = m.tipp_home.isna().groupby(m.pair).all().sum()
m = m[m.predicted_at <= m.kickoff]  # nur Tipps vor Anpfiff
m = m.sort_values("predicted_at").groupby("pair").tail(1).sort_values("kickoff").reset_index(drop=True)


def points(th, ta, ah, aa):
    if th == ah and ta == aa:
        return 4
    if th == ta:  # Unentschieden getippt
        return 2 if ah == aa else 0
    if ah == aa or (th > ta) != (ah > aa):
        return 0
    return 3 if th - ta == ah - aa else 2


m["points"] = [points(*x) for x in zip(m.tipp_home, m.tipp_away, m.home_score, m.away_score)]
m["category"] = m.points.map({0: "Falsch", 2: "Tendenz", 3: "Tendenz + Differenz", 4: "Exakt"})
m["pred_draw"] = m.tipp_home == m.tipp_away
m["act_draw"] = m.home_score == m.away_score
m["tendency_ok"] = m.points > 0
m["abs_err"] = (m.pred_home_goals - m.home_score).abs() + (m.pred_away_goals - m.away_score).abs()

m_all = m
m = m[m.stage == "GROUP_STAGE"].reset_index(drop=True)  # K.o.: Endstand inkl. Verl./Elfm. nicht mit 90-Min-Tipp vergleichbar
total = m.points.sum()
print(f"Spiele bewertet: {len(m)} (von {len(r)} beendeten)")
print(f"Engine-Punkte: {total}  (Ø {total/len(m):.2f} pro Spiel)")
print(m.points.value_counts().sort_index().to_string())
print(f"Tendenz korrekt: {m.tendency_ok.mean():.1%} | exakt: {(m.points==4).mean():.1%}")
print("Pro Phase:\n", m.groupby("stage").points.agg(["count", "sum", "mean"]).round(2).to_string())
print("Getippte Unentschieden:", m.pred_draw.sum(), "| echte Unentschieden:", m.act_draw.sum())
m_all.to_csv("data/processed/tippspiel_auswertung.csv", index=False)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"Falsch": "#c0392b", "Tendenz": "#e0a526", "Tendenz + Differenz": "#4a90c2", "Exakt": "#2e8b57"}

# 1) Punkteverteilung
fig, ax = plt.subplots(figsize=(7, 4.5))
cats = list(C)
cnt = m.category.value_counts().reindex(cats).fillna(0).astype(int)
bars = ax.bar(cats, cnt, color=[C[c] for c in cats])
ax.bar_label(bars)
ax.set_ylabel("Anzahl Spiele")
ax.set_title(f"Gruppenphase: {total} Punkte aus {len(m)} Spielen")
plt.xticks(rotation=10)
fig.tight_layout(); fig.savefig(f"{OUT}/7_tippspiel_verteilung.png", dpi=150); plt.close()

# 2) Kumulierte Punkte Engine vs. Durchschnitt
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(m) + 1), m.points.cumsum(), color="#2e8b57", lw=2, label=f"Engine ({total} P)")
ax.set_xlabel("Spiel Nr. (chronologisch)"); ax.set_ylabel("Kumulierte Punkte")
ax.set_title("Punkteverlauf der Engine (Gruppenphase)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/8_tippspiel_kumuliert.png", dpi=150); plt.close()

# 3) Baselines (Gruppenphase)
def _p(a, b): return np.mean([points(a, b, x, y) for x, y in zip(m.home_score, m.away_score)])
fav = m.home_market_value >= m.away_market_value
mv = np.mean([points(2, 1, x, y) if f else points(1, 2, x, y) for f, x, y in zip(fav, m.home_score, m.away_score)])
names = ["Immer 1:1", "Immer 2:1 Heim", "Engine", "Marktwert-\nFavorit 2:1"]
vals = [_p(1, 1), _p(2, 1), m.points.mean(), mv]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(names, vals, color=["#aaa", "#aaa", "#2e8b57", "#aaa"])
ax.bar_label(bars, fmt="%.2f"); ax.set_ylabel("Ø Punkte pro Spiel")
ax.set_title("Engine vs. einfache Baselines (Gruppenphase, 72 Spiele)")
fig.tight_layout(); fig.savefig(f"{OUT}/9_tippspiel_baselines.png", dpi=150); plt.close()

# 4) Tipp vs. Ergebnis-Heatmap der getippten Resultate
fig, ax = plt.subplots(figsize=(6, 5))
ct = pd.crosstab(m.tipp_home.astype(str) + ":" + m.tipp_away.astype(str), m.category)
ct = ct.reindex(columns=cats, fill_value=0)
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
ct.plot(kind="barh", stacked=True, ax=ax, color=[C[c] for c in cats])
ax.invert_yaxis(); ax.set_xlabel("Anzahl"); ax.set_ylabel("Getippter Spielstand")
ax.set_title("Getippte Ergebnisse und ihre Trefferqualität"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/10_tippspiel_ergebnisse.png", dpi=150); plt.close()
