"""
JWT Authentication, password hashing, and Role-Based Access Control (RBAC)
for CleanPath AI.
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from database import User, get_db

load_dotenv()

logger = logging.getLogger("cleanpath.auth")

SECRET_KEY = (
    os.getenv("JWT_SECRET_KEY")
    or os.getenv("SECRET_KEY", "cleanpath-ai-super-secret-dev-key-2026")
)
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

security_scheme = HTTPBearer(auto_error=False)

# Optional bcrypt / passlib support with safe PBKDF2-HMAC-SHA256 fallback
try:
    import bcrypt as _bcrypt_lib

    HAS_BCRYPT = True
except Exception:
    _bcrypt_lib = None
    HAS_BCRYPT = False

try:
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    HAS_PASSLIB = True
except Exception:
    pwd_context = None
    HAS_PASSLIB = False


def hash_password(password: str) -> str:
    """
    Hash a plaintext password securely using bcrypt (if available) or passlib PBKDF2-SHA256.
    Never stores plaintext passwords.
    """
    if not password:
        raise ValueError("Password cannot be empty")

    if HAS_BCRYPT and _bcrypt_lib is not None:
        try:
            salt = _bcrypt_lib.gensalt()
            hashed = _bcrypt_lib.hashpw(password.encode("utf-8"), salt)
            return hashed.decode("utf-8")
        except Exception as exc:
            logger.warning("bcrypt hash failed, falling back to passlib/pbkdf2: %s", exc)

    if HAS_PASSLIB and pwd_context is not None:
        return pwd_context.hash(password)

    # Standard library cryptographic PBKDF2 fallback
    salt_bytes = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
    return f"pbkdf2_sha256${salt_bytes.hex()}${dk.hex()}"


get_password_hash = hash_password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored password hash."""
    if not plain_password or not hashed_password:
        return False

    if hashed_password.startswith("$2") and HAS_BCRYPT and _bcrypt_lib is not None:
        try:
            return _bcrypt_lib.checkpw(
                plain_password.encode("utf-8"),
                hashed_password.encode("utf-8"),
            )
        except Exception as exc:
            logger.error("bcrypt verification error: %s", exc)
            return False

    if hashed_password.startswith("pbkdf2_sha256$"):
        try:
            _, salt_hex, dk_hex = hashed_password.split("$", 2)
            salt_bytes = bytes.fromhex(salt_hex)
            expected_dk = bytes.fromhex(dk_hex)
            actual_dk = hashlib.pbkdf2_hmac(
                "sha256", plain_password.encode("utf-8"), salt_bytes, 100_000
            )
            return hmac.compare_digest(actual_dk, expected_dk)
        except Exception as exc:
            logger.error("PBKDF2 verification error: %s", exc)
            return False

    if HAS_PASSLIB and pwd_context is not None:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as exc:
            logger.error("Passlib verification error: %s", exc)
            return False

    return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token. Raises HTTPException on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token has expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        return payload
    except JWTError as exc:
        logger.debug("JWT verification failed: %s", exc)
        raise credentials_exception from exc


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency to retrieve the currently authenticated user."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    FastAPI dependency that returns the current user if a valid Bearer token
    is supplied, or None for guest citizen usage.
    """
    if credentials is None or not credentials.credentials:
        return None
    try:
        payload = verify_token(credentials.credentials)
        username = payload.get("sub")
        if not username:
            return None
        return db.query(User).filter(User.username == username).first()
    except HTTPException:
        return None


def require_role(*allowed_roles: str) -> Callable[..., User]:
    """
    Role-Based Access Control (RBAC) dependency factory.
    Ensures the authenticated user holds one of the specified roles.
    """
    normalized_roles = {r.lower() for r in allowed_roles}

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = (current_user.role or "").lower()
        if user_role not in normalized_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}.",
            )
        return current_user

    return role_checker


def seed_demo_users(db: Session) -> None:
    """
    Ensure default citizen and officer demo accounts exist in the database.
    Citizen: citizen / citizen123
    Officer: officer / officer123
    """
    demo_accounts = [
        {
            "username": "citizen",
            "email": "citizen@cleanpath.local",
            "password": "citizen123",
            "role": "citizen",
        },
        {
            "username": "officer",
            "email": "officer@cleanpath.local",
            "password": "officer123",
            "role": "officer",
        },
    ]

    for account in demo_accounts:
        existing = (
            db.query(User)
            .filter(
                (User.username == account["username"])
                | (User.email == account["email"])
            )
            .first()
        )
        if existing is None:
            user = User(
                username=account["username"],
                email=account["email"],
                password_hash=hash_password(account["password"]),
                role=account["role"],
            )
            db.add(user)
            logger.info("Created demo user '%s' with role '%s'", account["username"], account["role"])
    db.commit()
