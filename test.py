"""
F1 Fantasy Predictive Model
============================

Trains a model that predicts fantasy points per driver, per race, from
pre-race features (form, quali pace, constructor strength, track history).

NOTE ON DATA: This script uses SYNTHETIC historical race data so it runs
end-to-end as a demo. To use real data, replace `load_race_history()` with
a pull from the FastF1 package:

    import fastf1
    fastf1.Cache.enable_cache('./f1_cache')
    session = fastf1.get_session(2024, 'Monza', 'R')
    session.load()
    results = session.results  # has finishing pos, grid pos, status, etc.

You'd loop this over every race in your training window (e.g. last 3-4
seasons), assembling one row per driver per race, then compute official
fantasy points from finishing position / quali position / overtakes / DNF
using that season's F1 Fantasy scoring rules.

Run: python model_training.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


# ---------------------------------------------------------------------------
# 1. LOAD / SIMULATE HISTORICAL RACE DATA
# ---------------------------------------------------------------------------
def load_race_history(n_races: int = 60, seed: int = 7) -> pd.DataFrame:
    """
    Returns one row per (race, driver) with:
      - race_id, race_number (chronological order — important for time-based splits)
      - driver, team
      - grid_position (quali result)
      - finish_position
      - dnf (bool)
      - fantasy_points (the TARGET — what we're trying to predict pre-race)

    Replace this function with real historical results (FastF1 / official
    F1 Fantasy scoring) when you have network access to those sources.
    """
    rng = np.random.default_rng(seed)
    drivers = {
        "Verstappen": ("Red Bull", 0.92), "Perez": ("Red Bull", 0.68),
        "Norris": ("McLaren", 0.88), "Piastri": ("McLaren", 0.82),
        "Leclerc": ("Ferrari", 0.80), "Sainz": ("Ferrari", 0.75),
        "Hamilton": ("Mercedes", 0.74), "Russell": ("Mercedes", 0.72),
        "Alonso": ("Aston Martin", 0.65), "Stroll": ("Aston Martin", 0.45),
        "Gasly": ("Alpine", 0.55), "Ocon": ("Alpine", 0.52),
        "Albon": ("Williams", 0.50), "Sargeant": ("Williams", 0.35),
        "Tsunoda": ("RB", 0.58), "Ricciardo": ("RB", 0.56),
        "Bottas": ("Sauber", 0.48), "Zhou": ("Sauber", 0.40),
        "Magnussen": ("Haas", 0.46), "Hulkenberg": ("Haas", 0.50),
    }

    rows = []
    for race_num in range(1, n_races + 1):
        # skill -> grid position via noisy ranking (better skill -> better avg grid)
        noisy_skill = {d: s[1] + rng.normal(0, 0.12) for d, s in drivers.items()}
        grid_order = sorted(noisy_skill, key=noisy_skill.get, reverse=True)

        for grid_pos, driver in enumerate(grid_order, start=1):
            team, skill = drivers[driver]
            # finishing position correlates with grid but with race-day noise + occasional big swings
            finish_noise = rng.normal(0, 2.5)
            finish_position = grid_pos + finish_noise
            dnf = rng.random() < 0.06  # ~6% DNF rate
            finish_position = 21 if dnf else int(np.clip(round(finish_position), 1, 20))

            # simplified fantasy scoring: quali points + race points + overtake bonus - dnf penalty
            quali_pts = max(0, 21 - grid_pos)
            race_pts = max(0, 21 - finish_position) * 2 if not dnf else 0
            overtakes = max(0, grid_pos - finish_position) if not dnf else 0
            overtake_pts = overtakes * 1.5
            dnf_penalty = -20 if dnf else 0
            fantasy_points = quali_pts + race_pts + overtake_pts + dnf_penalty

            rows.append({
                "race_id": race_num,
                "race_number": race_num,
                "driver": driver,
                "team": team,
                "grid_position": grid_pos,
                "finish_position": finish_position,
                "dnf": dnf,
                "fantasy_points": round(fantasy_points, 1),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds PRE-RACE features — i.e. only things you'd actually know before
    a race happens. This is critical: never let a feature leak information
    from the race you're trying to predict (e.g. don't use that race's own
    grid_position as a feature if you don't have quali results yet when you
    need the prediction — decide up front which point in the weekend you're
    predicting FROM).

    Here we predict after qualifying (grid_position known, so it's a valid
    pre-race feature), using rolling stats computed only from PRIOR races.
    """
    df = df.sort_values(["driver", "race_number"]).copy()

    def rolling_shifted(group, col, window):
        # shift(1) ensures we only use races BEFORE the current one — no leakage
        return group[col].shift(1).rolling(window, min_periods=1).mean()

    df["avg_finish_last3"] = df.groupby("driver", group_keys=False).apply(
        lambda g: rolling_shifted(g, "finish_position", 3)
    ).values
    df["avg_finish_last5"] = df.groupby("driver", group_keys=False).apply(
        lambda g: rolling_shifted(g, "finish_position", 5)
    ).values
    df["avg_points_last3"] = df.groupby("driver", group_keys=False).apply(
        lambda g: rolling_shifted(g, "fantasy_points", 3)
    ).values
    df["dnf_rate_last10"] = df.groupby("driver", group_keys=False).apply(
        lambda g: rolling_shifted(g, "dnf", 10)
    ).values

    # constructor/team form: rolling avg finish across BOTH team cars, prior races only
    team_race = df.groupby(["team", "race_number"])["finish_position"].mean().reset_index()
    team_race = team_race.sort_values(["team", "race_number"])
    team_race["team_form"] = team_race.groupby("team", group_keys=False).apply(
        lambda g: rolling_shifted(g, "finish_position", 5)
    ).values
    df = df.merge(team_race[["team", "race_number", "team_form"]], on=["team", "race_number"], how="left")

    # drop the first few races per driver where rolling stats are NaN (no history yet)
    df = df.dropna(subset=["avg_finish_last3", "avg_finish_last5", "team_form"])

    return df


FEATURE_COLS = [
    "grid_position",       # this race's quali result (known pre-race)
    "avg_finish_last3",
    "avg_finish_last5",
    "avg_points_last3",
    "dnf_rate_last10",
    "team_form",
]
TARGET_COL = "fantasy_points"


# ---------------------------------------------------------------------------
# 3. TRAIN + TIME-AWARE VALIDATION
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    """
    Uses GroupKFold on race_number so that entire races stay together in a
    fold (avoids leaking a driver's race-day result across train/test in
    the same race). For a real project, prefer a simple chronological
    train/test split (e.g. train on seasons 2021-2023, test on 2024) —
    that best simulates "predicting the future."
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df["race_number"]

    gkf = GroupKFold(n_splits=5)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.2f} fantasy points")

    print(f"\nAverage MAE across folds: {np.mean(maes):.2f}")

    # train final model on ALL data for actual use
    final_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    final_model.fit(X, y)

    importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance.to_string())

    return final_model


# ---------------------------------------------------------------------------
# 4. PREDICT FOR AN UPCOMING RACE
# ---------------------------------------------------------------------------
def predict_next_race(model, df: pd.DataFrame, upcoming_grid: dict) -> pd.DataFrame:
    """
    upcoming_grid: {"Verstappen": 1, "Norris": 2, ...} — this race's quali results
    Uses each driver's MOST RECENT rolling stats (from their last known race)
    combined with their new grid position for this race.
    """
    latest = df.sort_values("race_number").groupby("driver").tail(1).set_index("driver")

    rows = []
    for driver, grid_pos in upcoming_grid.items():
        if driver not in latest.index:
            continue
        row = latest.loc[driver]
        rows.append({
            "driver": driver,
            "grid_position": grid_pos,
            "avg_finish_last3": row["avg_finish_last3"],
            "avg_finish_last5": row["avg_finish_last5"],
            "avg_points_last3": row["avg_points_last3"],
            "dnf_rate_last10": row["dnf_rate_last10"],
            "team_form": row["team_form"],
        })

    pred_df = pd.DataFrame(rows)
    pred_df["predicted_points"] = model.predict(pred_df[FEATURE_COLS])
    return pred_df.sort_values("predicted_points", ascending=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading race history...")
    raw = load_race_history(n_races=60)

    print("Building features (with proper time-shifting to avoid leakage)...")
    features_df = build_features(raw)
    print(f"Training rows: {len(features_df)}\n")

    print("Training model with grouped cross-validation...\n")
    model = train_and_evaluate(features_df)

    # simulate an upcoming race's qualifying result
    upcoming_grid = {
        "Verstappen": 1, "Norris": 2, "Piastri": 3, "Leclerc": 4, "Sainz": 5,
        "Hamilton": 6, "Russell": 7, "Alonso": 8, "Perez": 9, "Gasly": 10,
    }
    print("\nPredicting next race from quali results...")
    predictions = predict_next_race(model, features_df, upcoming_grid)
    print(predictions[["driver", "grid_position", "predicted_points"]].to_string(index=False))