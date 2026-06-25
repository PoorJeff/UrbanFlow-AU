from __future__ import annotations

from dataclasses import dataclass
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

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, MelbourneApiError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _fetch_page(self, source_url: str, *, limit: int, offset: int) -> dict[str, Any]:
        try:
            response = self._http_client.get(source_url, params={"limit": limit, "offset": offset})
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
        if "results" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'results'")
        return payload
