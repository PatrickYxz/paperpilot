"""Local authentication helpers for the Web workbench."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from paperpilot.web.task_store import DuplicateUsernameError, TaskStore, WebUser

SESSION_COOKIE_NAME = "paperpilot_session"

_HASH_NAME = "sha256"
_HASH_ITERATIONS = 210_000
_HASH_LENGTH = 32


class UsernameAlreadyExistsError(ValueError):
    """Raised when registering a duplicate username."""


class InvalidCredentialsError(ValueError):
    """Raised when login credentials do not match a user."""


@dataclass(frozen=True)
class AuthenticatedSession:
    user: WebUser
    token: str


class AuthService:
    """Owns local user registration, login, and session lookup."""

    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def register(self, *, username: str, password: str) -> AuthenticatedSession:
        normalized = normalize_username(username)
        salt = secrets.token_hex(16)
        password_hash = hash_password(password, salt)
        try:
            user = self.store.create_user(
                username=normalized,
                password_hash=password_hash,
                password_salt=salt,
            )
        except DuplicateUsernameError as exc:
            raise UsernameAlreadyExistsError("username already exists") from exc
        token = self.store.create_session(user.id)
        return AuthenticatedSession(user=user, token=token)

    def login(self, *, username: str, password: str) -> AuthenticatedSession:
        normalized = normalize_username(username)
        user = self.store.get_user_by_username(normalized)
        if user is None:
            raise InvalidCredentialsError("invalid username or password")
        if not verify_password(password, user.password_salt, user.password_hash):
            raise InvalidCredentialsError("invalid username or password")
        token = self.store.create_session(user.id)
        return AuthenticatedSession(user=user, token=token)

    def get_user_for_token(self, token: str | None) -> WebUser | None:
        if not token:
            return None
        return self.store.get_user_for_session(token)

    def logout(self, token: str | None) -> None:
        if token:
            self.store.delete_session(token)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        _HASH_NAME,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _HASH_ITERATIONS,
        dklen=_HASH_LENGTH,
    ).hex()


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    actual_hash = hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)
