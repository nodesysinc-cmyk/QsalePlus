import secrets
import jwt

from pwdlib import PasswordHash


password_hash = PasswordHash.recommended()

JWT_SECRET = "CHANGE_THIS_SECRET"
JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str
) -> bool:
    return password_hash.verify(
        password,
        hashed_password
    )


def create_verification_token() -> str:
    return secrets.token_urlsafe(32)


def create_auth_token(
    restaurant_id: int
) -> str:
    return jwt.encode(
        {"restaurant_id": restaurant_id},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
