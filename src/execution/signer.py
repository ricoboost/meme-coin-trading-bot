"""Local transaction signer using BOT_PRIVATE_KEY_B58.

This module provides Solana transaction signing from a base58-encoded
private key supplied via environment variable.  The private key is NEVER
logged or exposed outside this module.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SOL-native mint constant
SOL_MINT = "So11111111111111111111111111111111111111112"


class SignerError(Exception):
    """Raised when signing validation or operation fails."""


class LocalSigner:
    """Sign Solana transactions using a base58-encoded keypair.

    The signer is intentionally lazy – it only attempts to import
    ``solders`` and decode the key when :meth:`validate` is called so that
    the paper-trading build can import this module without the dependency.
    """

    def __init__(self, private_key_b58: str) -> None:
        self._raw_key_b58 = private_key_b58
        self._keypair: object | None = None  # solders.keypair.Keypair once validated
        self._public_key_str: str | None = None
        self._validated = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate the private key and return ``(ok, error_msg)``.

        Returns ``(True, None)`` on success, ``(False, reason)`` on failure.
        """
        if not self._raw_key_b58:
            return False, "BOT_PRIVATE_KEY_B58 is empty or missing"

        try:
            import base58
            from solders.keypair import Keypair  # type: ignore[import-untyped]

            decoded = base58.b58decode(self._raw_key_b58)
            if len(decoded) not in (32, 64):
                return (
                    False,
                    f"Decoded key length {len(decoded)} invalid; expected 32 (seed) or 64 (full keypair)",
                )

            if len(decoded) == 64:
                kp = Keypair.from_bytes(decoded)
            else:
                kp = Keypair.from_seed(decoded)

            self._keypair = kp
            self._public_key_str = str(kp.pubkey())
            self._validated = True
            logger.info("Signer validated – public key: %s", self._public_key_str)
            return True, None

        except ImportError as exc:
            return False, f"Missing signing dependency: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Key validation failed: {exc}"

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_public_key(self) -> str | None:
        """Return the derived public key string, or ``None`` if not validated."""
        return self._public_key_str

    @property
    def keypair(self):
        """Return the ``solders.keypair.Keypair`` instance.

        Raises :class:`SignerError` if the signer has not been validated.
        """
        if not self._validated or self._keypair is None:
            raise SignerError("Signer not validated – call validate() first")
        return self._keypair

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign_transaction(self, raw_transaction: bytes) -> bytes:
        """Deserialize, sign, and re-serialize a versioned transaction.

        Parameters
        ----------
        raw_transaction:
            The raw unsigned transaction bytes (as returned by Jupiter swap
            endpoint after base64-decoding).

        Returns
        -------
        bytes
            The signed transaction bytes ready for broadcasting.
        """
        if not self._validated or self._keypair is None:
            raise SignerError("Cannot sign – signer not validated")

        try:
            from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

            tx = VersionedTransaction.from_bytes(raw_transaction)
            signed_tx = VersionedTransaction(tx.message, [self._keypair])
            return bytes(signed_tx)
        except Exception as exc:
            raise SignerError(f"Transaction signing failed: {exc}") from exc
