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


def test_count_records_uses_limit_zero_and_where_clause() -> None:
    requested_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_params.append(dict(request.url.params))
        return json_response({"total_count": 2295, "results": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    result = api_client.count_records(
        "pedestrian-counting-system-monthly-counts-per-hour",
        where="sensing_date = date'2025-01-01'",
    )

    assert requested_params == [{"limit": "0", "where": "sensing_date = date'2025-01-01'"}]
    assert result.dataset == "pedestrian-counting-system-monthly-counts-per-hour"
    assert result.total_count == 2295
    assert result.source_url.endswith("/pedestrian-counting-system-monthly-counts-per-hour/records")


def test_count_records_rejects_non_integer_total_count() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: json_response({"total_count": "2295", "results": []})
        )
    )
    api_client = MelbourneApiClient(http_client=http_client)

    with pytest.raises(MelbourneApiError, match="total_count"):
        api_client.count_records("pedestrian-counting-system-monthly-counts-per-hour")


def test_export_url_points_to_dataset_export_endpoint() -> None:
    api_client = MelbourneApiClient(base_url="https://example.test/datasets")

    assert (
        api_client.export_url(
            "pedestrian-counting-system-monthly-counts-per-hour",
            export_format="csv",
        )
        == "https://example.test/datasets/"
        "pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
    )


def test_export_csv_streams_selected_columns_to_file(tmp_path) -> None:
    requested_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_params.append(dict(request.url.params))
        assert request.url.path.endswith(
            "/pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
        )
        return httpx.Response(
            200,
            content=b"id,location_id,sensing_date\n51120250101,51,2025-01-01\n",
            headers={"content-type": "text/csv; charset=utf-8"},
        )

    output_path = tmp_path / "records.csv"
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    api_client.export_csv(
        "pedestrian-counting-system-monthly-counts-per-hour",
        output_path=output_path,
        select=["id", "location_id", "sensing_date"],
        where="sensing_date = date'2025-01-01'",
    )

    assert requested_params == [
        {
            "delimiter": ",",
            "select": "id,location_id,sensing_date",
            "where": "sensing_date = date'2025-01-01'",
            "with_bom": "false",
        }
    ]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "id,location_id,sensing_date",
        "51120250101,51,2025-01-01",
    ]
