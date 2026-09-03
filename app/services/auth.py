"""Small signed-token authentication for the local two-role demo."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

from .knowledge import KnowledgeRepository


@dataclass(frozen=True)
class DemoUser:
    username: str
    display_name: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict[str, str]:
        return {"username": self.username, "display_name": self.display_name, "role": self.role}


class LocalAuthService:
    """Signs short-lived local sessions without storing secrets in the browser."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository
        configured_secret = os.getenv("DEMO_SESSION_SECRET", "").strip()
        # A demo can run without configuration, but a public fallback secret
        # would make signed role claims forgeable.  An ephemeral high-entropy
        # key keeps sessions local to this running server; deployments may set
        # their own persistent secret through the ignored .env file.
        insecure_examples = {"change-this-local-demo-session-secret", "changzhou-local-demo-session-key-change-me"}
        self.secret = (
            configured_secret
            if len(configured_secret) >= 32 and configured_secret not in insecure_examples
            else secrets.token_urlsafe(48)
        ).encode("utf-8")
        self.ttl_seconds = max(900, int(os.getenv("DEMO_SESSION_TTL_SECONDS", "28800")))

    def login(self, username: str, password: str) -> tuple[str, DemoUser] | None:
        record = self.repository.authenticate(username.strip(), password)
        if not record:
            return None
        user = DemoUser(**record)
        return self._sign(user), user

    def decode(self, token: str) -> DemoUser | None:
        try:
            encoded, signature = token.split(".", 1)
            expected = hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return None
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
            if not isinstance(payload, dict) or float(payload.get("expires_at", 0)) < time.time():
                return None
            user = DemoUser(username=str(payload["username"]), display_name=str(payload["display_name"]), role=str(payload["role"]))
            if user.role not in {"admin", "officer"}:
                return None
            stored = self.repository.user_identity(user.username)
            if not stored or stored["role"] != user.role or stored["display_name"] != user.display_name:
                return None
            return DemoUser(**stored)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _sign(self, user: DemoUser) -> str:
        payload: dict[str, Any] = {**user.to_dict(), "expires_at": int(time.time()) + self.ttl_seconds}
        encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
        signature = hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"
