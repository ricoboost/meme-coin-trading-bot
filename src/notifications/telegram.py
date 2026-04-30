"""Optional Telegram notifications."""

from __future__ import annotations

import logging

import httpx

from src.bot.config import BotConfig


class TelegramNotifier:
    """Small optional Telegram sender."""

    def __init__(self, config: BotConfig) -> None:
        self.enabled = config.enable_telegram and bool(
            config.telegram_bot_token and config.telegram_chat_id
        )
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.logger = logging.getLogger(self.__class__.__name__)

    def _chat_id_payload(self) -> int | str:
        raw = self.chat_id.strip()
        if raw.startswith("-") and raw[1:].isdigit():
            return int(raw)
        if raw.isdigit():
            return int(raw)
        return raw

    def send(self, message: str) -> None:
        """Best-effort send."""
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(
                    url, json={"chat_id": self._chat_id_payload(), "text": message}
                )
                if response.status_code >= 400:
                    self.logger.warning(
                        "Telegram send failed with status %s: %s",
                        response.status_code,
                        response.text,
                    )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Telegram send failed: %s", exc)
