"""Helius Enhanced API client for wallet transaction history."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from src.utils.retry import default_retry


class HeliusClient:
    """Thin Helius Enhanced API client with pagination and lookback cutoff support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_sec: int = 30,
        page_size: int = 100,
        max_pages: int = 100,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.page_size = page_size
        self.max_pages = max_pages
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client: httpx.Client | None = None

    def __enter__(self) -> "HeliusClient":
        """Open a persistent HTTP client for repeated requests."""
        self._client = httpx.Client(timeout=self.timeout_sec)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        """Close the persistent HTTP client."""
        self.close()

    def close(self) -> None:
        """Close any open HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @default_retry()
    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Send a GET request and return JSON."""
        if not self.api_key:
            raise ValueError("HELIUS_API_KEY is required")
        query = dict(params or {})
        query["api-key"] = self.api_key
        url = f"{self.base_url}{path}"
        client = self._client or httpx.Client(timeout=self.timeout_sec)
        close_after = self._client is None
        try:
            response = client.get(url, params=query)
            response.raise_for_status()
            payload = response.json()
        finally:
            if close_after:
                client.close()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected Helius response type: {type(payload)}")
        return payload

    @default_retry()
    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Send a POST request and return parsed JSON."""
        if not self.api_key:
            raise ValueError("HELIUS_API_KEY is required")
        query = dict(params or {})
        query["api-key"] = self.api_key
        url = f"{self.base_url}{path}"
        client = self._client or httpx.Client(timeout=self.timeout_sec)
        close_after = self._client is None
        try:
            response = client.post(url, params=query, json=payload)
            response.raise_for_status()
            return response.json()
        finally:
            if close_after:
                client.close()

    @staticmethod
    def _timestamp_to_datetime(timestamp: Any) -> datetime | None:
        """Convert a Helius timestamp field to a naive UTC datetime for comparisons."""
        if timestamp is None:
            return None
        try:
            return datetime.utcfromtimestamp(int(timestamp))
        except (TypeError, ValueError, OSError):
            return None

    def fetch_wallet_transactions(self, wallet: str, cutoff_dt: datetime) -> list[dict[str, Any]]:
        """Fetch parsed wallet transaction history until the lookback cutoff or page limit is reached."""
        all_rows: list[dict[str, Any]] = []
        before: Optional[str] = None
        cutoff_naive = cutoff_dt.replace(tzinfo=None)

        for page in range(self.max_pages):
            params: dict[str, Any] = {"limit": self.page_size}
            if before:
                params["before"] = before

            rows = self._get(f"/v0/addresses/{wallet}/transactions", params=params)
            if not rows:
                break

            eligible_rows: list[dict[str, Any]] = []
            page_datetimes = [self._timestamp_to_datetime(row.get("timestamp")) for row in rows]
            valid_page_datetimes = [tx_dt for tx_dt in page_datetimes if tx_dt is not None]
            newest_dt = valid_page_datetimes[0] if valid_page_datetimes else None
            oldest_dt = valid_page_datetimes[-1] if valid_page_datetimes else None

            for row in rows:
                tx_dt = self._timestamp_to_datetime(row.get("timestamp"))
                if tx_dt is None:
                    continue
                if tx_dt < cutoff_naive:
                    break
                eligible_rows.append(row)

            all_rows.extend(eligible_rows)

            self.logger.info(
                "Fetched %s txs for %s on page %s; kept %s within cutoff%s",
                len(rows),
                wallet,
                page + 1,
                len(eligible_rows),
                f" (range {oldest_dt} -> {newest_dt})" if newest_dt and oldest_dt else "",
            )
            before = rows[-1].get("signature")
            if not before:
                break
            if oldest_dt is not None and oldest_dt < cutoff_naive:
                self.logger.info("Reached lookback cutoff for %s on page %s", wallet, page + 1)
                break

        return all_rows

    def parse_transactions(
        self, signatures: list[str], commitment: str = "confirmed"
    ) -> list[dict[str, Any]]:
        """Parse one or more transaction signatures via the Enhanced Transactions API."""
        if not signatures:
            return []
        payload = {"transactions": signatures}
        response = self._post(
            "/v0/transactions", payload=payload, params={"commitment": commitment}
        )
        if not isinstance(response, list):
            raise ValueError(f"Unexpected Helius parse response type: {type(response)}")
        return [row for row in response if isinstance(row, dict)]
