"""
validation.py
==============
Phase 3 of the Energy Consumption Forecasting system: strict input & type
validation so the app never crashes or feeds physically impossible values
into the trained ANN.

Every check returns a `ValidationResult` (ok / error message) rather than
raising, so the Streamlit layer can render friendly inline alerts instead of
stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Physical / meteorological guardrails (Phase 3 spec)
LIMITS = {
    "temperature": {"min": -20.0, "max": 50.0, "unit": "°C"},
    "humidity": {"min": 0.0, "max": 100.0, "unit": "%"},
    "wind_speed": {"min": 0.0, "max": 60.0, "unit": "m/s"},  # 60 m/s ~ hurricane-force ceiling
    "hour": {"min": 0, "max": 23, "unit": "h"},
    "day_of_week": {"min": 0, "max": 6, "unit": ""},  # 0=Mon .. 6=Sun
}


@dataclass
class ValidationResult:
    ok: bool
    value: Optional[float] = None
    errors: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def _to_number(raw: Any, field_name: str, integer: bool = False) -> ValidationResult:
    """Type-checking step: safely coerce user input to int/float."""
    result = ValidationResult(ok=True)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        result.add(f"{field_name} is required.")
        return result
    try:
        value = int(raw) if integer else float(raw)
    except (ValueError, TypeError):
        result.add(
            f"'{raw}' is not a valid number for {field_name}. "
            f"Please enter a numeric value."
        )
        return result
    result.value = value
    return result


def validate_temperature(raw: Any) -> ValidationResult:
    r = _to_number(raw, "Temperature")
    if not r.ok:
        return r
    lo, hi = LIMITS["temperature"]["min"], LIMITS["temperature"]["max"]
    if not (lo <= r.value <= hi):
        r.add(f"Temperature must be between {lo}°C and {hi}°C (got {r.value}°C). "
              f"Extreme/impossible values are blocked.")
    return r


def validate_humidity(raw: Any) -> ValidationResult:
    r = _to_number(raw, "Humidity")
    if not r.ok:
        return r
    lo, hi = LIMITS["humidity"]["min"], LIMITS["humidity"]["max"]
    if not (lo <= r.value <= hi):
        r.add(f"Humidity must be between {lo}% and {hi}% (got {r.value}%).")
    return r


def validate_wind_speed(raw: Any) -> ValidationResult:
    r = _to_number(raw, "Wind Speed")
    if not r.ok:
        return r
    lo, hi = LIMITS["wind_speed"]["min"], LIMITS["wind_speed"]["max"]
    if r.value < lo:
        r.add(f"Wind speed cannot be negative (got {r.value} m/s).")
    elif r.value > hi:
        r.add(f"Wind speed of {r.value} m/s exceeds the realistic sensor ceiling of {hi} m/s.")
    return r


def validate_hour(raw: Any) -> ValidationResult:
    r = _to_number(raw, "Hour", integer=True)
    if not r.ok:
        return r
    lo, hi = LIMITS["hour"]["min"], LIMITS["hour"]["max"]
    if not (lo <= r.value <= hi):
        r.add(f"Hour must be an integer between {lo} and {hi} (got {r.value}).")
    return r


def validate_day_of_week(raw: Any) -> ValidationResult:
    r = _to_number(raw, "Day of Week", integer=True)
    if not r.ok:
        return r
    lo, hi = LIMITS["day_of_week"]["min"], LIMITS["day_of_week"]["max"]
    if not (lo <= r.value <= hi):
        r.add(f"Day of week must be an integer between {lo} (Mon) and {hi} (Sun) (got {r.value}).")
    return r


def validate_is_holiday(raw: Any) -> ValidationResult:
    """Accepts bool, 0/1, or 'yes'/'no' style inputs from a checkbox/select."""
    result = ValidationResult(ok=True)
    if isinstance(raw, bool):
        result.value = int(raw)
        return result
    if isinstance(raw, (int, float)) and raw in (0, 1):
        result.value = int(raw)
        return result
    if isinstance(raw, str) and raw.strip().lower() in ("yes", "no", "true", "false", "0", "1"):
        result.value = 1 if raw.strip().lower() in ("yes", "true", "1") else 0
        return result
    result.add("Holiday flag must be Yes/No.")
    return result


def validate_all(payload: dict) -> tuple[bool, dict, list[str]]:
    """
    Validates a full prediction request payload.
    Returns: (all_ok, cleaned_values_dict, list_of_all_error_messages)
    """
    checks = {
        "temperature": validate_temperature(payload.get("temperature")),
        "humidity": validate_humidity(payload.get("humidity")),
        "wind_speed": validate_wind_speed(payload.get("wind_speed")),
        "hour": validate_hour(payload.get("hour")),
        "day_of_week": validate_day_of_week(payload.get("day_of_week")),
        "is_holiday": validate_is_holiday(payload.get("is_holiday")),
    }

    all_ok = all(r.ok for r in checks.values())
    cleaned = {k: r.value for k, r in checks.items()}
    errors = [msg for r in checks.values() for msg in r.errors]
    return all_ok, cleaned, errors
