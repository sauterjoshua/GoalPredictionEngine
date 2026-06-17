import sys
sys.path.insert(0, "/home/joshi/Projects/CornerPredictionEngine")
from predict import predict_match

# Basis-Stats Template
def make_stats(market_value, titles):
    return {
        "form_attack": 1.5, "form_defense": 1.2,
        "squad_total_market_value_eur": market_value,
        "world_cup_titles_before": titles
    }

# Branch 1: Gruppenspiel → draw erlaubt
r = predict_match(make_stats(800, 4), make_stats(800, 4), is_knockout=False)
assert r["predicted_winner"] == "draw" or r["decided_by"] == "90min"
print("✅ Branch 1 (Gruppe):", r["predicted_winner"], r["decided_by"])

# Branch 2: KO, klarer Favorit → sollte in 90min oder ET entscheiden
r = predict_match(make_stats(1500, 5), make_stats(200, 0), is_knockout=True)
assert r["predicted_winner"] != "draw"
print("✅ Branch 2 (KO Favorit):", r["predicted_winner"], r["decided_by"])

# Branch 3: KO, ausgeglichene Teams → Elfmeter (Marktwert + Titel entscheidet)
r = predict_match(make_stats(800, 4), make_stats(800, 0), is_knockout=True)
assert r["predicted_winner"] in ["home", "away"]
print("✅ Branch 3 (Elfmeter):", r["predicted_winner"], r["decided_by"], 
      "| home_et:", r["home_goals_et"], "away_et:", r["away_goals_et"])