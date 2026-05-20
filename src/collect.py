import fastf1
import pandas as pd

fastf1.Cache.enable_cache("data/cache")

def collect_all(seasons):
    races = []
    weather = []
    qualifying = []

    for season in seasons:
        for rnd in range(1, 25):
            try:
                # --- race session ---
                race_session = fastf1.get_session(season, rnd, "R")
                race_session.load(telemetry=False, weather=True,
                                  messages=False, laps=False)

                results = race_session.results
                for _, row in results.iterrows():
                    races.append({
                        "season": season,
                        "round": rnd,
                        "circuit": race_session.event["EventName"],
                        "driver": row["Abbreviation"],
                        "team": row["TeamName"],
                        "grid": row["GridPosition"],
                        "position": row["Position"],
                        "points": row["Points"],
                        "status": row["Status"],
                        "podium": int(pd.to_numeric(
                            row["Position"], errors="coerce") <= 3)
                    })

                w = race_session.weather_data
                weather.append({
                    "season": season, "round": rnd,
                    "avg_temp": round(w["AirTemp"].mean(), 2),
                    "avg_humidity": round(w["Humidity"].mean(), 2),
                    "avg_wind": round(w["WindSpeed"].mean(), 2),
                    "rainfall": int(w["Rainfall"].any()),
                })

                # --- qualifying session ---
                quali_session = fastf1.get_session(season, rnd, "Q")
                quali_session.load(telemetry=False, weather=False,
                                   messages=False, laps=False)

                quali_results = quali_session.results
                pole_time = pd.to_numeric(
                    quali_results["Q3"].dt.total_seconds(),
                    errors="coerce"
                ).min()

                for _, row in quali_results.iterrows():
                    best_time = None
                    for q in ["Q3", "Q2", "Q1"]:
                        t = row[q]
                        if pd.notna(t):
                            best_time = t.total_seconds()
                            break

                    qualifying.append({
                        "season": season,
                        "round": rnd,
                        "driver": row["Abbreviation"],
                        "grid_position": row["Position"],
                        "best_quali_time": best_time,
                        "gap_to_pole": (best_time - pole_time)
                                        if best_time and pole_time else None,
                        "made_q3": int(pd.notna(row["Q3"])),
                        "made_q2": int(pd.notna(row["Q2"])),
                    })

                print(f"  {season} R{rnd} — {race_session.event['EventName']} ✓")

            except Exception as e:
                print(f"  {season} R{rnd} skipped: {e}")
                if rnd > 22:
                    break

    return (pd.DataFrame(races),
            pd.DataFrame(weather),
            pd.DataFrame(qualifying))
    races = []
    weather = []

    for season in seasons:
        for rnd in range(1, 25):
            try:
                session = fastf1.get_session(season, rnd, "R")
                session.load(telemetry=False, weather=True,
                             messages=False, laps=False)

                # race results
                results = session.results
                for _, row in results.iterrows():
                    races.append({
                        "season": season,
                        "round": rnd,
                        "circuit": session.event["EventName"],
                        "driver": row["Abbreviation"],
                        "team": row["TeamName"],
                        "grid": row["GridPosition"],
                        "position": row["Position"],
                        "points": row["Points"],
                        "status": row["Status"],
                        "podium": int(row["Position"] <= 3)
                            if str(row["Position"]).isdigit() else 0
                    })

                # weather
                w = session.weather_data
                weather.append({
                    "season": season,
                    "round": rnd,
                    "avg_temp": round(w["AirTemp"].mean(), 2),
                    "avg_humidity": round(w["Humidity"].mean(), 2),
                    "avg_wind": round(w["WindSpeed"].mean(), 2),
                    "rainfall": int(w["Rainfall"].any()),
                })

                print(f"  {season} R{rnd} — {session.event['EventName']} ✓")

            except Exception as e:
                print(f"  {season} R{rnd} skipped: {e}")
                if rnd > 22:
                    break

    return pd.DataFrame(races), pd.DataFrame(weather)


seasons = [2022, 2023, 2024, 2025]

print("Collecting from FastF1...")
df_races, df_weather, df_qualifying = collect_all(seasons)

df_races.to_csv("data/raw/races.csv", index=False)
df_weather.to_csv("data/raw/weather.csv", index=False)
df_qualifying.to_csv("data/raw/qualifying.csv", index=False)

print(f"\nRace rows: {len(df_races)}")
print(f"Weather rows: {len(df_weather)}")
print(f"Qualifying rows: {len(df_qualifying)}")