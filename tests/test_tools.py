import pytest

from rag_assistant.tools import add_to_date, days_between


def test_add_days_and_weeks():
    assert add_to_date("2027-01-01", 15, "days")["date"] == "2027-01-16"
    assert add_to_date("2027-01-01", 2, "weeks", "before")["date"] == "2026-12-18"


def test_months_keep_day_number():
    assert add_to_date("2027-03-15", 2, "months", "before")["date"] == "2027-01-15"


def test_months_clamp_to_end_of_month():
    # 31 March minus 1 month: there is no 31 February -> last day of February
    assert add_to_date("2027-03-31", 1, "months", "before")["date"] == "2027-02-28"
    assert add_to_date("2028-01-31", 1, "months")["date"] == "2028-02-29"  # leap year


def test_years_and_year_boundary():
    assert add_to_date("2026-11-30", 3, "months")["date"] == "2027-02-28"
    assert add_to_date("2024-02-29", 1, "years")["date"] == "2025-02-28"


def test_weekday_is_reported():
    assert add_to_date("2026-11-18", 0, "days")["weekday"] == "Wednesday"


def test_days_between():
    assert days_between("2027-01-01", "2027-03-01")["days"] == 59
    assert days_between("2027-03-01", "2027-01-01")["days"] == -59


@pytest.mark.parametrize("kwargs", [
    {"amount": -1, "unit": "days"},
    {"amount": 1, "unit": "fortnights"},
    {"amount": 1, "unit": "days", "direction": "sideways"},
])
def test_bad_input_raises(kwargs):
    with pytest.raises(ValueError):
        add_to_date("2027-01-01", **kwargs)
