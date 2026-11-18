"""Date tools the agent can call.

LLMs are unreliable at calendar arithmetic ("2 months before 31 March?").
These functions do it exactly. The LLM decides *when* to call them and with
*which* numbers (read from the sources); Python does the maths.

Limits (also written in the README): calendar dates only. They do not apply
legal extension rules (for example a deadline that ends on a Sunday or a
public holiday), but they do say which weekday the date falls on.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _add_months(d: date, months: int) -> date:
    """Same day number N months later; if that day does not exist (31 Feb),
    use the last day of the month. This matches the usual French rule for
    deadlines in months (art. 641 Code de procédure civile)."""
    month_index = d.month - 1 + months
    year, month = d.year + month_index // 12, month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def add_to_date(start_date: str, amount: int, unit: str, direction: str = "after") -> dict:
    """Move a date forwards or backwards.

    Args:
        start_date: ISO date, e.g. "2027-03-31".
        amount: a whole number >= 0, e.g. 2.
        unit: "days", "weeks", "months" or "years".
        direction: "after" (later date) or "before" (earlier date).
    """
    d = date.fromisoformat(start_date)
    if amount < 0:
        raise ValueError("amount must be >= 0; use direction='before' to go back in time")
    if direction not in ("after", "before"):
        raise ValueError("direction must be 'after' or 'before'")
    sign = 1 if direction == "after" else -1
    if unit == "days":
        result = d + timedelta(days=sign * amount)
    elif unit == "weeks":
        result = d + timedelta(weeks=sign * amount)
    elif unit == "months":
        result = _add_months(d, sign * amount)
    elif unit == "years":
        result = _add_months(d, sign * 12 * amount)
    else:
        raise ValueError("unit must be days, weeks, months or years")
    return {"date": result.isoformat(), "weekday": WEEKDAYS[result.weekday()]}


def days_between(start_date: str, end_date: str) -> dict:
    """Number of calendar days from start_date to end_date (negative if end is earlier)."""
    delta = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    return {"days": delta}
