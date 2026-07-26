import pytest

from urbanflow.api.schemas import SensorResponse
from urbanflow.dashboard.context import (
    SELECTED_LOCATION_ID_KEY,
    clear_selected_location_if_missing,
    get_selected_location_id,
    set_selected_location_id,
)


def _sensor(location_id: int) -> SensorResponse:
    return SensorResponse(
        location_id=location_id,
        sensor_name=f"Sensor {location_id}",
        sensor_description="Test sensor",
        status="Active",
        latitude=-37.81,
        longitude=144.96,
    )


@pytest.mark.parametrize("invalid_value", [None, "101", True, 1.5])
def test_get_selected_location_id_clears_invalid_values(invalid_value: object) -> None:
    session_state: dict[str, object] = {SELECTED_LOCATION_ID_KEY: invalid_value}

    assert get_selected_location_id(session_state) is None
    assert SELECTED_LOCATION_ID_KEY not in session_state


def test_set_selected_location_id_records_only_the_integer_focus() -> None:
    session_state: dict[str, object] = {"unrelated": "preserved"}

    set_selected_location_id(session_state, 101)

    assert session_state == {
        "unrelated": "preserved",
        SELECTED_LOCATION_ID_KEY: 101,
    }


def test_clear_selected_location_if_missing_keeps_a_catalogued_focus() -> None:
    session_state: dict[str, object] = {SELECTED_LOCATION_ID_KEY: 101}

    selected = clear_selected_location_if_missing(
        session_state,
        [_sensor(101), _sensor(202)],
    )

    assert selected == 101
    assert session_state == {SELECTED_LOCATION_ID_KEY: 101}


def test_clear_selected_location_if_missing_removes_a_stale_focus() -> None:
    session_state: dict[str, object] = {SELECTED_LOCATION_ID_KEY: 101}

    selected = clear_selected_location_if_missing(session_state, [_sensor(202)])

    assert selected is None
    assert SELECTED_LOCATION_ID_KEY not in session_state
