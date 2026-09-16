from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.db.models import Restaurant, Menu
from src.services.security import (
    hash_password,
    verify_password,
    create_verification_token,
    create_auth_token,
)

router = APIRouter()


async def get_restaurant_from_auth(
    authorization: str,
    db: AsyncSession,
) -> Restaurant:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )

    auth_code = authorization.replace(
        "Bearer ",
        "",
        1
    ).strip()

    if not auth_code:
        raise HTTPException(
            status_code=401,
            detail="Authorization token is missing"
        )

    result = await db.execute(
        select(Restaurant).where(
            Restaurant.verification_token == auth_code
        )
    )

    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=401,
            detail="Invalid auth code"
        )

    return restaurant


@router.post("/restaurant/add")
async def add_restaurant(
    name: str,
    email: str,
    password: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Restaurant).where(
            Restaurant.email == email
        )
    )

    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    token = create_verification_token()

    restaurant = Restaurant(
        name=name,
        email=email,
        password_hash=hash_password(password),
        is_verified=False,
        verification_token=token,
    )

    db.add(restaurant)

    await db.commit()
    await db.refresh(restaurant)

    return {
        "success": True,
        "restaurant_id": restaurant.id,
        "verification_token": token,
        "verification_link": (
            f"/router/restaurant/verify/{token}"
        ),
    }


# VERIFY RESTAURANT
@router.get("/restaurant/verify/{token}")
async def verify_restaurant(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Restaurant).where(
            Restaurant.verification_token == token
        )
    )

    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=404,
            detail="Invalid verification token"
        )

    restaurant.is_verified = True
    restaurant.verification_token = None

    await db.commit()

    return {
        "success": True,
        "message": "Restaurant verified successfully"
    }


# SIGN IN
@router.post("/restaurant/signin")
async def signin(
    email: str,
    password: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Restaurant).where(
            Restaurant.email == email
        )
    )

    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        password,
        restaurant.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # if not restaurant.is_verified:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Please verify your email first"
    #     )

    auth_code = create_auth_token(
        restaurant.id
    )

    restaurant.verification_token = auth_code

    await db.commit()
    return {
        "success": True,
        "auth_code": auth_code,
        "restaurant_id": restaurant.id,
    }


@router.get("/restaurant/menus")
async def list_menus(
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    # Get restaurant from auth token
    restaurant = await get_restaurant_from_auth(
        authorization,
        db
    )

    if not restaurant:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    # Get restaurant menus
    result = await db.execute(
        select(Menu)
        .where(Menu.restaurant_id == restaurant.id, Menu.status == True),

    )

    menus = result.scalars().all()

    return {
        "success": True,
        "menus": menus
    }
