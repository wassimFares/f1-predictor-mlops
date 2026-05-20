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

        # check results loaded successfully
        if session.results is None or len(session.results) == 0:
            print(f"No results data available for {season} R{round_num}")
            return None, None, None

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

        # handle missing weather gracefully
        try:
            w = session.weather_data
            if w is not None and len(w) > 0:
                new_weather = [{
                    "season": season, "round": round_num,
                    "avg_temp": round(w["AirTemp"].mean(), 2),
                    "avg_humidity": round(w["Humidity"].mean(), 2),
                    "avg_wind": round(w["WindSpeed"].mean(), 2),
                    "rainfall": int(w["Rainfall"].any()),
                }]
            else:
                raise ValueError("No weather data")
        except:
            print("  Weather unavailable — using median imputation")
            new_weather = [{
                "season": season, "round": round_num,
                "avg_temp": None,
                "avg_humidity": None,
                "avg_wind": None,
                "rainfall": 0,
            }]

        # qualifying
        try:
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
                    "season": season, "round": round_num,
                    "driver": row["Abbreviation"],
                    "grid_position": row["Position"],
                    "best_quali_time": best_time,
                    "gap_to_pole": (best_time - pole_time)
                                    if best_time and pole_time else None,
                    "made_q3": int(pd.notna(row["Q3"])),
                    "made_q2": int(pd.notna(row["Q2"])),
                })
        except Exception as e:
            print(f"  Qualifying unavailable: {e}")
            new_quali = []

        return (pd.DataFrame(new_races),
                pd.DataFrame(new_weather),
                pd.DataFrame(new_quali) if new_quali else None)

    except Exception as e:
        print(f"Failed to collect round {round_num}: {e}")
        return None, None, None
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

max_failures = 3
failures = 0

while True:
    next_season, next_round = get_next_race(last_season, last_round)

    if next_season is None:
        print(f"No more races available — collected {collected} new races")
        break

    print(f"Collecting: {next_season} R{next_round}")
    new_races, new_weather, new_quali = collect_new_race(next_season, next_round)

    if new_races is None or len(new_races) == 0:
        failures += 1
        print(f"Could not collect {next_season} R{next_round} — skipping ({failures}/{max_failures})")
        last_season, last_round = next_season, next_round
        if failures >= max_failures:
            print("Too many failures — stopping")
            break
        continue

    failures = 0  # reset on success
    append_to_csv(new_races, "data/raw/races.csv")
    append_to_csv(new_weather, "data/raw/weather.csv")
    if new_quali is not None and len(new_quali) > 0:
        append_to_csv(new_quali, "data/raw/qualifying.csv")

    print(f"Successfully added {next_season} R{next_round}")
    collected += 1
    last_season, last_round = next_season, next_round

if collected == 0:
    print("No new races collected — skipping retraining")
    exit(1)