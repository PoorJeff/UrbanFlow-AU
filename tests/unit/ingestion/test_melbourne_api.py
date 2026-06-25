import json

import httpx
import pytest

from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError


def json_response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_fetch_all_records_paginates_until_source_total_count() -> None:
    requested_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_offsets.append(int(request.url.params["offset"]))
        assert request.url.params["limit"] == "2"
        if request.url.params["offset"] == "0":
            return json_response(
                {"total_count": 3, "results": [{"location_id": 1}, {"location_id": 2}]}
            )
        return json_response({"total_count": 3, "results": [{"location_id": 3}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    page = api_client.fetch_all_records("pedestrian-counting-system-sensor-locations", limit=2)

    assert requested_offsets == [0, 2]
    assert page.total_count == 3
    assert page.records == [{"location_id": 1}, {"location_id": 2}, {"location_id": 3}]
    assert page.source_url.endswith("/pedestrian-counting-system-sensor-locations/records")


def test_fetch_all_records_rejects_malformed_payload_without_results() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: json_response({"total_count": 1}))
    )
    api_client = MelbourneApiClient(http_client=http_client)

    with pytest.raises(MelbourneApiError, match="results"):
        api_client.fetch_all_records("pedestrian-counting-system-sensor-locations", limit=100)
