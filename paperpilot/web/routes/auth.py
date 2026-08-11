"""Authentication HTTP routes."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response

from paperpilot.web.auth import (
    SESSION_COOKIE_NAME,
    AuthService,
    InvalidCredentialsError,
    UsernameAlreadyExistsError,
)
from paperpilot.web.schemas import AuthRequest, UserResponse
from paperpilot.web.task_store import WebUser


RequireUser = Callable[..., WebUser]


def build_auth_router(
    *,
    auth: AuthService,
    require_user: RequireUser,
) -> APIRouter:
    router = APIRouter(prefix="/api/auth")

    @router.post("/register", response_model=UserResponse, status_code=201)
    def register(payload: AuthRequest, response: Response) -> dict[str, str]:
        try:
            session = auth.register(
                username=payload.username,
                password=payload.password,
            )
        except UsernameAlreadyExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail="username already exists",
            ) from exc
        _set_session_cookie(response, session.token)
        return session.user.to_public_dict()

    @router.post("/login", response_model=UserResponse)
    def login(payload: AuthRequest, response: Response) -> dict[str, str]:
        try:
            session = auth.login(
                username=payload.username,
                password=payload.password,
            )
        except InvalidCredentialsError as exc:
            raise HTTPException(
                status_code=401,
                detail="invalid username or password",
            ) from exc
        _set_session_cookie(response, session.token)
        return session.user.to_public_dict()

    @router.post("/logout", status_code=204)
    def logout(
        response: Response,
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
    ) -> None:
        auth.logout(session_token)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return None

    @router.get("/me", response_model=UserResponse)
    def get_current_user(
        user: WebUser = Depends(require_user),
    ) -> dict[str, str]:
        return user.to_public_dict()

    return router


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
    )
