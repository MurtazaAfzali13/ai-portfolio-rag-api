# app/core/security.py
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

# ساخت خودکار آدرس JWKS از روی CLERK_FRONTEND_API
_raw_frontend_api = os.getenv("CLERK_FRONTEND_API", "").strip().rstrip("/")
if _raw_frontend_api and not _raw_frontend_api.startswith("http"):
    _raw_frontend_api = f"https://{_raw_frontend_api}"

CLERK_JWKS_URL = f"{_raw_frontend_api}/.well-known/jwks.json" if _raw_frontend_api else ""
MAX_QUESTIONS_PER_USER = int(os.getenv("MAX_QUESTIONS_PER_USER", "2"))

# اختیاری: اگر بخواهید علاوه بر metadata داشبورد Clerk، با ID هم ادمین تعریف کنید
ADMIN_USER_IDS: Set[str] = {
    uid.strip()
    for uid in os.getenv("ADMIN_USER_IDS", "").split(",")
    if uid.strip()
}

_jwks_client: Optional[PyJWKClient] = (
    PyJWKClient(CLERK_JWKS_URL, cache_keys=True) if CLERK_JWKS_URL else None
)

security_scheme = HTTPBearer(auto_error=False)
DB_PATH = Path("data/usage_limits.db")


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_usage (
                user_id TEXT PRIMARY KEY,
                question_count INTEGER NOT NULL DEFAULT 0,
                last_asked_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


_init_db()


def verify_clerk_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Dict[str, Any]:
    """توکن Bearer را اعتبارسنجی کرده و اطلاعات کاربر (user_id و is_admin) را برمی‌گرداند."""
    if not _jwks_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured on the backend (CLERK_FRONTEND_API missing).",
        )

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Please sign in first.",
        )

    token = credentials.credentials
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: user ID not found.",
            )

        # بررسی نقش ادمین از Public Metadata درون توکن Clerk
        metadata = payload.get("metadata") or payload.get("public_metadata") or {}
        role = metadata.get("role") if isinstance(metadata, dict) else None
        is_admin = (role == "admin") or (user_id in ADMIN_USER_IDS)

        return {"user_id": user_id, "is_admin": is_admin}

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please refresh the page.",
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


def enforce_user_quota(user: Dict[str, Any] = Depends(verify_clerk_token)) -> str:
    """ادمین‌ها نامحدود هستند؛ سایر کاربران حداکثر ۲ پیام مجاز دارند."""
    user_id: str = user["user_id"]

    # اگر در داشبورد Clerk نقش admin دارد، بدون محدودیت عبور کند
    if user.get("is_admin"):
        return user_id

    now_iso = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT question_count FROM user_usage WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        current_count = row["question_count"] if row else 0

        if current_count >= MAX_QUESTIONS_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"You have reached the demo limit of {MAX_QUESTIONS_PER_USER} questions. "
                    "Feel free to reach out to Murtaza directly via email or LinkedIn!"
                ),
            )

        if row:
            conn.execute(
                "UPDATE user_usage SET question_count = question_count + 1, last_asked_at = ? WHERE user_id = ?",
                (now_iso, user_id),
            )
        else:
            conn.execute(
                "INSERT INTO user_usage (user_id, question_count, last_asked_at) VALUES (?, 1, ?)",
                (user_id, now_iso),
            )
        conn.commit()

    return user_id