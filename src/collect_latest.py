import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("data/cache")

def get_latest_round():
    df = pd.read_csv("data/raw/races.csv")
    last_season = int(df["season"].max())
    last_round = int(df[df["season"] == last_season]["round"].max())
    return last_season, last_round

def get_next_race(last_season, last_round):
    # try next round in same season
    try:
        session = fastf1.get_session(last_season, last_round + 1, "R")
        session.load(telemetry=False, weather=False,
                     messages=False, laps=False)
        return last_season, last_round + 1
    except:
        pass

    # try round 1 of next season
    try:
        session = fastf1.get_session(last_season + 1, 1, "R")
        session.load(telemetry=False, weather=False,
                     messages=False, laps=False)
        return last_season + 1, 1
    except:
        return None, None

def collect_new_race(season, round_num):
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(telemetry=False, weather=True,
                     messages=False, laps=False)

        results = session.results
        new_races = []
        for _, row in results.iterrows():
            new_races.append({
                "season": season,
                "round": round_num,
                "circuit": session.event["EventName"],
                "driver": row["Abbreviation"],
                "team": row["TeamName"],
                "grid": row["GridPosition"],
                "position": row["Position"],
                "points": row["Points"],
                "status": row["Status"],
                "podium": int(pd.to_numeric(
                    row["Position"], errors="coerce") <= 3)
            })

        w = session.weather_data
        new_weather = [{
            "season": season,
            "round": round_num,
            "avg_temp": round(w["AirTemp"].mean(), 2),
            "avg_humidity": round(w["Humidity"].mean(), 2),
            "avg_wind": round(w["WindSpeed"].mean(), 2),
            "rainfall": int(w["Rainfall"].any()),
        }]

        quali_session = fastf1.get_session(season, round_num, "Q")
        quali_session.load(telemetry=False, weather=False,
                           messages=False, laps=False)
        quali_results = quali_session.results
        pole_time = pd.to_numeric(
            quali_results["Q3"].dt.total_seconds(),
            errors="coerce"
        ).min()

        new_quali = []
        for _, row in quali_results.iterrows():
            best_time = None
            for q in ["Q3", "Q2", "Q1"]:
                t = row[q]
                if pd.notna(t):
                    best_time = t.total_seconds()
                    break
            new_quali.append({
                "season": season,
                "round": round_num,
                "driver": row["Abbreviation"],
                "grid_position": row["Position"],
                "best_quali_time": best_time,
                "gap_to_pole": (best_time - pole_time)
                                if best_time and pole_time else None,
                "made_q3": int(pd.notna(row["Q3"])),
                "made_q2": int(pd.notna(row["Q2"])),
            })

        return (pd.DataFrame(new_races),
                pd.DataFrame(new_weather),
                pd.DataFrame(new_quali))

    except Exception as e:
        print(f"Failed to collect round {round_num}: {e}")
        return None, None, None

def append_to_csv(new_df, path):
    if new_df is None or len(new_df) == 0:
        return
    existing = pd.read_csv(path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(
        subset=["season","round","driver"]
        if "driver" in combined.columns
        else ["season","round"],
        keep="last"
    ).to_csv(path, index=False)
    print(f"  {path}: {len(existing)} → {len(combined)} rows")

if __name__ == "__main__":
    last_season, last_round = get_latest_round()
    print(f"Last collected: {last_season} R{last_round}")

    next_season, next_round = get_next_race(last_season, last_round)

    if next_season is None:
        print("No new race available yet — skipping")
        exit(1)

    print(f"Collecting: {next_season} R{next_round}")
    new_races, new_weather, new_quali = collect_new_race(next_season, next_round)

    if new_races is not None:
        append_to_csv(new_races, "data/raw/races.csv")
        append_to_csv(new_weather, "data/raw/weather.csv")
        append_to_csv(new_quali, "data/raw/qualifying.csv")
        print(f"Successfully added {next_season} R{next_round}")
    else:
        print("Collection failed")
        exit(1)