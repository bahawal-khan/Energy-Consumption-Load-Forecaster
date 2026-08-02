"""
generate_sample_data.py
========================
Generates a synthetic CSV that matches the exact schema of the Fedesoriano
"Electric Power Consumption" (Tetouan City) Kaggle dataset:

    DateTime, Temperature, Humidity, Wind Speed,
    general diffuse flows, diffuse flows,
    Zone 1 Power Consumption, Zone 2 Power Consumption, Zone 3 Power Consumption

WHY THIS EXISTS
----------------
This repo cannot reach the internet to download the real Kaggle file for you.
This script produces a *realistic stand-in* (10-minute cadence, daily +
weekly seasonality, temperature-driven load, weekend dip) purely so you can
run `train.py` and the Streamlit app end-to-end immediately.

>>> REPLACE data/powerconsumption.csv WITH THE REAL KAGGLE FILE FOR PRODUCTION USE. <<<
Download it manually from:
    https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption
and drop it in `data/powerconsumption.csv` (same column names) before
running train.py for a real model.
"""

import numpy as np
import pandas as pd


def generate(n_days: int = 180, freq_minutes: int = 10, seed: int = 42, out_path: str = "data/powerconsumption.csv") -> str:
    rng = np.random.default_rng(seed)
    n_points = int(n_days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range("2017-01-01", periods=n_points, freq=f"{freq_minutes}min")

    hour = timestamps.hour + timestamps.minute / 60
    day_of_week = timestamps.dayofweek
    day_of_year = timestamps.dayofyear

    # Temperature: yearly + daily sinusoid + noise
    temperature = (
        18
        + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
        + rng.normal(0, 1.2, n_points)
    )

    humidity = np.clip(
        60 - 0.8 * (temperature - 18) + rng.normal(0, 5, n_points), 5, 100
    )
    wind_speed = np.clip(3 + 2 * np.sin(2 * np.pi * hour / 24 + 1) + rng.normal(0, 1, n_points), 0, None)

    general_diffuse_flows = np.clip(
        200 * np.maximum(0, np.sin(2 * np.pi * (hour - 6) / 24)) + rng.normal(0, 10, n_points), 0, None
    )
    diffuse_flows = np.clip(general_diffuse_flows * 0.3 + rng.normal(0, 5, n_points), 0, None)

    is_weekend = (day_of_week >= 5).astype(float)

    def zone_load(base, temp_sens, evening_peak, weekend_dip):
        daily = (
            base
            + 6000 * np.maximum(0, np.sin(2 * np.pi * (hour - 7) / 24))  # morning ramp
            + evening_peak * np.maximum(0, np.sin(2 * np.pi * (hour - 17) / 24))  # evening peak
            + temp_sens * np.maximum(0, temperature - 24) ** 1.3  # AC load above 24C
            + temp_sens * 0.5 * np.maximum(0, 10 - temperature)  # heating load below 10C
        )
        daily *= (1 - weekend_dip * is_weekend)
        noise = rng.normal(0, base * 0.03, n_points)
        return np.clip(daily + noise, 0, None)

    zone1 = zone_load(base=22000, temp_sens=180, evening_peak=8000, weekend_dip=0.18)
    zone2 = zone_load(base=15000, temp_sens=120, evening_peak=6000, weekend_dip=0.15)
    zone3 = zone_load(base=10000, temp_sens=90, evening_peak=4000, weekend_dip=0.10)

    df = pd.DataFrame({
        "DateTime": timestamps,
        "Temperature": temperature.round(2),
        "Humidity": humidity.round(2),
        "Wind Speed": wind_speed.round(2),
        "general diffuse flows": general_diffuse_flows.round(2),
        "diffuse flows": diffuse_flows.round(2),
        "Zone 1 Power Consumption": zone1.round(2),
        "Zone 2 Power Consumption": zone2.round(2),
        "Zone 3 Power Consumption": zone3.round(2),
    })

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
