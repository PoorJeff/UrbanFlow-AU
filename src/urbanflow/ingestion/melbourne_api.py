from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MELBOURNE_API_BASE_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"


class MelbourneApiError(RuntimeError):
    """Raised when the Melbourne Open Data API cannot return a usable records page."""


@dataclass(frozen=True)
class DatasetRecords:
    dataset: str
    source_url: str
    total_count: int
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class DatasetRecordCount:
    dataset: str
    source_url: str
    total_count: int


class MelbourneApiClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        base_url: str = MELBOURNE_API_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self._base_url = base_url.rstrip("/")

    def records_url(self, dataset: str) -> str:
        if not dataset:
            raise ValueError("dataset must not be empty")
        return f"{self._base_url}/{dataset}/records"

    def export_url(self, dataset: str, *, export_format: str) -> str:
        if not dataset:
            raise ValueError("dataset must not be empty")
        if not export_format:
            raise ValueError("export_format must not be empty")
        return f"{self._base_url}/{dataset}/exports/{export_format}"

    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        source_url = self.records_url(dataset)
        offset = 0
        total_count: int | None = None
        records: list[dict[str, Any]] = []

        while total_count is None or len(records) < total_count:
            payload = self._fetch_page(source_url, limit=limit, offset=offset)
            page_total_count = payload["total_count"]
            page_results = payload["results"]

            if not isinstance(page_total_count, int):
                raise MelbourneApiError("API response field 'total_count' must be an integer")
            if not isinstance(page_results, list):
                raise MelbourneApiError("API response field 'results' must be a list")
            if total_count is None:
                total_count = page_total_count
            if not page_results and len(records) < total_count:
                raise MelbourneApiError("API returned an empty page before total_count was reached")

            records.extend(page_results)
            offset += len(page_results)

        return DatasetRecords(
            dataset=dataset,
            source_url=source_url,
            total_count=total_count or 0,
            records=records,
        )

    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        source_url = self.records_url(dataset)
        params: dict[str, str | int] = {"limit": 0}
        if where:
            params["where"] = where

        payload = self._fetch_records_query(source_url, params=params)
        total_count = payload["total_count"]
        if not isinstance(total_count, int):
            raise MelbourneApiError("API response field 'total_count' must be an integer")

        return DatasetRecordCount(
            dataset=dataset,
            source_url=source_url,
            total_count=total_count,
        )

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: Sequence[str],
        where: str | None = None,
    ) -> None:
        if not select:
            raise ValueError("select must contain at least one column")

        export_url = self.export_url(dataset, export_format="csv")
        params: dict[str, str] = {
            "delimiter": ",",
            "select": ",".join(select),
            "with_bom": "false",
        }
        if where:
            params["where"] = where

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._http_client.stream("GET", export_url, params=params) as response:
                response.raise_for_status()
                with output_path.open("wb") as output_file:
                    for chunk in response.iter_bytes():
                        output_file.write(chunk)
        except httpx.HTTPError as exc:
            raise MelbourneApiError(f"Melbourne API CSV export failed: {exc}") from exc

    def _fetch_page(self, source_url: str, *, limit: int, offset: int) -> dict[str, Any]:
        payload = self._fetch_records_query(
            source_url,
            params={"limit": limit, "offset": offset},
        )
        if "results" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'results'")
        return payload

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, MelbourneApiError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _fetch_records_query(
        self,
        source_url: str,
        *,
        params: dict[str, str | int],
    ) -> dict[str, Any]:
        try:
            response = self._http_client.get(source_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MelbourneApiError(f"Melbourne API request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise MelbourneApiError("Melbourne API response was not valid JSON") from exc

        if not isinstance(payload, dict):
            raise MelbourneApiError("Melbourne API response must be a JSON object")
        if "total_count" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'total_count'")
        return payload
