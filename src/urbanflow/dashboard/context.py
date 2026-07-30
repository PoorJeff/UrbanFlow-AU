from collections.abc import MutableMapping, Sequence

from urbanflow.api.schemas import SensorResponse

SELECTED_LOCATION_ID_KEY = "selected_location_id"


def get_selected_location_id(
    session_state: MutableMapping[str, object],
) -> int | None:
    selected = session_state.get(SELECTED_LOCATION_ID_KEY)
    if type(selected) is not int:
        session_state.pop(SELECTED_LOCATION_ID_KEY, None)
        return None
    return selected


def set_selected_location_id(
    session_state: MutableMapping[str, object],
    location_id: int,
) -> None:
    session_state[SELECTED_LOCATION_ID_KEY] = location_id


def clear_selected_location_if_missing(
    session_state: MutableMapping[str, object],
    sensors: Sequence[SensorResponse],
) -> int | None:
    selected = get_selected_location_id(session_state)
    if selected is None:
        return None
    if any(sensor.location_id == selected for sensor in sensors):
        return selected

    session_state.pop(SELECTED_LOCATION_ID_KEY, None)
    return None
