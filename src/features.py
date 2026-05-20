import pandas as pd
import numpy as np

def load_raw():
    df_races = pd.read_csv("data/raw/races.csv")
    df_weather = pd.read_csv("data/raw/weather.csv")
    df_quali = pd.read_csv("data/raw/qualifying.csv")
    print(f"Races: {df_races.shape} | Weather: {df_weather.shape} | Qualifying: {df_quali.shape}")
    return df_races, df_weather, df_quali

def clean_races(df_races):
    df_races = df_races.copy()
    df_races = df_races.dropna(subset=["position", "grid"])
    df_races["grid"] = df_races["grid"].astype(int)
    df_races["position"] = pd.to_numeric(df_races["position"], errors="coerce")
    df_races["podium"] = (df_races["position"] <= 3).astype(int)
    print(f"Races after cleaning: {len(df_races)} | Podiums: {df_races['podium'].sum()}")
    return df_races

def merge(df_races, df_weather, df_quali):
    df = df_races.merge(
        df_quali[["season","round","driver","best_quali_time","gap_to_pole"]],
        on=["season","round","driver"],
        how="left"
    )
    df = df.merge(df_weather, on=["season","round"], how="left")
    print(f"Merged shape: {df.shape}")
    return df

def clean_features(df):
    # drop missing qualifying times
    df = df.dropna(subset=["gap_to_pole", "best_quali_time"]).copy()

    # fix negative gap to pole and clip outliers
    df["gap_to_pole"] = df["gap_to_pole"].abs()
    df["gap_to_pole"] = df["gap_to_pole"].clip(upper=5.0)

    # impute missing weather with global median
    for col in ["avg_temp", "avg_humidity", "avg_wind", "rainfall"]:
        df[col] = df[col].fillna(df[col].median())

    print(f"After quali cleaning: {len(df)} rows")
    return df

def add_rolling_features(df):
    # sort by time — critical for rolling calculations
    df = df.sort_values(["season","round"]).reset_index(drop=True)

    # driver rolling form — last 5 races
    df["rolling_podium_rate"] = (
        df.groupby("driver")["podium"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["rolling_points"] = (
        df.groupby("driver")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # team rolling form — last 5 races
    df["team_rolling_points"] = (
        df.groupby("team")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # circuit-specific podium rate per driver
    df["circuit_podium_rate"] = (
        df.groupby(["driver","circuit"])["podium"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # fill nulls for first appearances
    for col in ["rolling_podium_rate","rolling_points",
                "team_rolling_points","circuit_podium_rate"]:
        df[col] = df[col].fillna(0)

    print("Rolling features added")
    return df

def save(df):
    feature_cols = [
        "grid", "gap_to_pole", "best_quali_time",
        "rolling_podium_rate", "rolling_points",
        "team_rolling_points", "circuit_podium_rate",
        "avg_temp", "avg_humidity", "avg_wind", "rainfall",
        "podium"
    ]
    meta_cols = ["season", "round", "circuit", "driver", "team"]

    df_final = df[meta_cols + feature_cols].copy()

    # final null check
    nulls = df_final.isnull().sum().sum()
    if nulls > 0:
        print(f"WARNING: {nulls} nulls found in final dataset")
    else:
        print("Null check passed")

    print(df_final.isnull().sum()[df_final.isnull().sum() > 0])

    df_final.to_csv("data/processed/features.csv", index=False)
    print(f"Saved: {df_final.shape} → data/processed/features.csv")
    print(f"Podiums: {df_final['podium'].sum()} / {len(df_final)}")

if __name__ == "__main__":
    df_races, df_weather, df_quali = load_raw()
    df_races = clean_races(df_races)
    df = merge(df_races, df_weather, df_quali)
    df = clean_features(df)
    df = add_rolling_features(df)
    save(df)