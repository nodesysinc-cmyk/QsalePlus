from src.services.email_services import send_menu_ready_email, send_menu_failed_email
import io
import os
import time
import asyncio
import json
import uuid
import csv

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
)

from src.services.pdf_service import convert_pdf_to_images
from src.services.image_service import convert_image_to_png
from src.db.models import Restaurant, MenuImage, Menu, PageCoordinates, MenuItem, MenuItemCompleteVariant, MenuItemComplete
from src.db import get_db

# -----------------------------------------------------------------------
# TODO: apni async session-maker ka ASAL naam yahan import karein.
# Ye wahi cheez hai jo "src/db/__init__.py" (ya jahan bhi get_db
# defined hai) mein banai gayi hoti hai, jaisay:
#     AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, ...)
# Background tasks request khatam hone ke BAAD chalte hain, isliye
# unhein purani "db" session use nahi karni chahiye (wo band ho chuki
# hoti hai) - inhein apni nayi session khud banani hoti hai, isi liye
# ye import zaroori hai.
# -----------------------------------------------------------------------
# <-- isay apne project ke mutabiq theek karein
from src.db import AsyncSessionLocal
from src.services.csv_exports import build_items_csv_bytes

from src.menu_handler.text_extraction import MenuDataExtractorSimple
from src.menu_handler.image_coordinate_handle import MenuImageExtractor
from src.menu_handler.text_extraction_image import MenuDataExtractor
from src.menu_handler.fallback_menu_images import MissingPhotoFallback

from pydantic import BaseModel
from typing import Optional, List, Any


class MenuItemUpdate(BaseModel):
    product_name: Optional[str] = None
    arabic_name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None


router = APIRouter()

BASE_URL = os.getenv("BASE_URL")


# -----------------------------------------------------------------------
# SHARED HELPERS (unchanged from before)
# -----------------------------------------------------------------------

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


async def get_next_menu_number(
    restaurant_id: int,
    db: AsyncSession,
) -> int:
    result = await db.execute(
        select(func.max(Menu.menu_number)).where(
            Menu.restaurant_id == restaurant_id
        )
    )

    last_number = result.scalar()

    return (last_number or 0) + 1


# -----------------------------------------------------------------------
# BACKGROUND PIPELINE — coordinates + Claude extraction.
# Ye SIRF response bhej diye jane ke BAAD chalta hai, isliye:
#   - apni khud ki nayi db session banata hai (purani request wali
#     session is waqt tak band ho chuki hoti hai)
#   - sirf plain data (paths, ids) leta hai, koi live UploadFile object
#     nahi (wo bhi is waqt tak band ho chuka hota hai)
#   - end mein menu.status ko True (ya fail hone par False) set karta hai
# -----------------------------------------------------------------------


async def run_heavy_pipeline(menu_id: int, all_pages: list):
    async with AsyncSessionLocal() as db:
        try:
            # -------------------------
            # STEP 1: coordinates nikalo + DB mein save karo
            # 3-3 images ke batches mein
            # -------------------------
            image_extractor = MenuImageExtractor()
            batch_size = 3

            all_items = []
            for i in range(0, len(all_pages), batch_size):
                batch = all_pages[i:i + batch_size]
                print(
                    f"[BATCH] Processing {i + 1}-{i + len(batch)} "
                    f"of {len(all_pages)} page(s) ..."
                )

                batch_result = await image_extractor.process_images(
                    pages=batch,
                    db=db,
                )
                all_items.extend(batch_result["items"])

            print(len(all_items))

            # -------------------------
            # STEP 2: Claude vision extraction + DB-based image linking
            # -------------------------
            pages_with_ids = [
                (item["path"], item["menu_image_id"]) for item in all_pages
            ]

            extractor = MenuDataExtractor()
            extraction_result = await extractor.extract(db, pages=pages_with_ids, menu_id=menu_id)

            # -------------------------
            # DONE — menu ka status True karo
            # -------------------------
            menu = await db.get(Menu, menu_id)
            menu.status = True
            await db.commit()

            print(f"[BACKGROUND DONE] menu_id={menu_id} status=True")

            # -------------------------
            # EMAIL — restaurant ko inform karo menu ready hai
            # -------------------------
            restaurant = await db.get(Restaurant, menu.restaurant_id)
            if restaurant:
                items_before_merge = (extraction_result or {}).get(
                    "items_before_merge", [])
                csv_bytes = build_items_csv_bytes(
                    items_before_merge, base_url=BASE_URL)

                await send_menu_ready_email(
                    to_email=restaurant.email,
                    restaurant_name=restaurant.name,
                    menu_id=menu.id,
                    menu_number=menu.menu_number,
                    csv_bytes=csv_bytes,
                )

        except Exception as e:
            print(f"[BACKGROUND ERROR] menu_id={menu_id}: {e}")
            menu = await db.get(Menu, menu_id)
            if menu:
                menu.status = False
                await db.commit()

                # -------------------------
                # EMAIL — restaurant ko fail hone ka inform karo
                # -------------------------
                restaurant = await db.get(Restaurant, menu.restaurant_id)
                if restaurant:
                    await send_menu_failed_email(
                        to_email=restaurant.email,
                        restaurant_name=restaurant.name,
                        menu_id=menu.id,
                        menu_number=menu.menu_number,
                    )


# -----------------------------------------------------------------------
# InMemoryUploadFile — sirf /images-to-png ke liye chahiye, kyunki uske
# convert_image_to_png ko ek UploadFile-jaisa object chahiye hota hai,
# aur asal UploadFile response jate hi band ho jata hai. PDF wale
# convert_pdf_to_images ko ye nahi chahiye - wo file synchronously,
# request ke andar hi read kar leta hai.
# -----------------------------------------------------------------------

class InMemoryUploadFile:
    def __init__(self, content: bytes, filename: str, content_type: str):
        self.filename = filename
        self.content_type = content_type
        self.file = io.BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> None:
        self.file.seek(offset)


# -----------------------------------------------------------------------
# /pdf-to-images
# -----------------------------------------------------------------------

@router.post("/pdf-to-images")
async def pdf_to_images(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    # -------------------------
    # CHECK PDF
    # -------------------------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    # -------------------------
    # GET AUTH CODE FROM HEADER
    # -------------------------
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid authorization header")

    auth_code = authorization.replace("Bearer ", "", 1).strip()

    if not auth_code:
        raise HTTPException(
            status_code=401, detail="Authorization token is missing")

    # -------------------------
    # FIND RESTAURANT
    # -------------------------
    result = await db.execute(
        select(Restaurant).where(Restaurant.verification_token == auth_code)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(status_code=401, detail="Invalid auth code")

    menu_number = await get_next_menu_number(restaurant.id, db)

    # -------------------------
    # CONVERT PDF — ye fast hai (page-split + save), isliye
    # synchronous hi rakha hai. Menu record yahin ban jata hai aur
    # menu_id turant frontend ko mil jata hai.
    # -------------------------
    result = await convert_pdf_to_images(
        file=file,
        restaurant_id=restaurant.id,
        menu_type="pdf",
        menu_number=menu_number,
        db=db,
    )
    # result["images"] = [{"page":..., "filename":..., "path":..., "menu_image_id":...}, ...]

    # -------------------------
    # Slow hissa (coordinates + Claude extraction) background mein
    # -------------------------
    background_tasks.add_task(
        run_heavy_pipeline, result["menu_id"], result["images"]
    )

    return {
        "success": True,
        "menu_id": result["menu_id"],
        "total_pages": result["total_pages"],
        "status": "processing, you will recive email once it done",
    }


# -----------------------------------------------------------------------
# /images-to-png
# -----------------------------------------------------------------------

@router.post("/images-to-png")
async def images_to_png(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one image"
        )

    restaurant = await get_restaurant_from_auth(authorization, db)
    menu_number = await get_next_menu_number(restaurant.id, db)

    menu = Menu(
        restaurant_id=restaurant.id,
        menu_type="image",
        menu_number=menu_number,
        path="",
        status=False,  # abhi processing baaki hai
    )

    db.add(menu)
    await db.commit()
    await db.refresh(menu)

    # -------------------------
    # Files ke bytes ABHI padh lete hain (request ke andar) - kyunki
    # response bhej diye jane ke baad UploadFile ka stream band ho
    # jata hai aur background task mein read() fail ho jayega.
    # -------------------------
    file_payloads = [
        {
            "filename": f.filename,
            "content": await f.read(),
            "content_type": f.content_type,
        }
        for f in files
    ]

    background_tasks.add_task(
        run_images_pipeline, menu.id, file_payloads
    )

    return {
        "success": True,
        "menu_id": menu.id,
        "status": "processing, you will recive email once it done",
    }


async def run_images_pipeline(menu_id: int, file_payloads: list[dict]):
    """
    /images-to-png ke liye poora background flow:
    pehle format-convert (jo pehle synchronous tha), phir wahi
    heavy pipeline jo /pdf-to-images bhi use karta hai.
    """
    async with AsyncSessionLocal() as db:
        try:
            results = []
            for index, payload in enumerate(file_payloads, start=1):
                fake_file = InMemoryUploadFile(
                    payload["content"], payload["filename"], payload["content_type"]
                )
                result = await convert_image_to_png(
                    file=fake_file,
                    menu_id=menu_id,
                    image_number=index,
                    db=db,
                )
                results.append(result)

            await db.commit()

        except Exception as e:
            print(f"[BACKGROUND ERROR - convert] menu_id={menu_id}: {e}")
            menu = await db.get(Menu, menu_id)
            if menu:
                menu.status = False
                await db.commit()
            return

    # convert step ki apni session band ho chuki, ab heavy pipeline
    # (coordinates + Claude) apni nayi session khud banayega
    await run_heavy_pipeline(menu_id, results)


# -----------------------------------------------------------------------
# /menu/{menu_id}/status — frontend isay poll karega jab tak "done" na aaye
# -----------------------------------------------------------------------

@router.get("/menu/{menu_id}/status")
async def get_menu_status(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Menu).where(Menu.id == menu_id))
    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(status_code=404, detail="Menu not found")

    return {"status": "done" if menu.status else "processing"}


# -----------------------------------------------------------------------
# BAAKI ENDPOINTS — bilkul unchanged
# -----------------------------------------------------------------------

@router.patch("/menu-item/{item_id}")
async def update_menu_item(
    item_id: int,
    payload: MenuItemUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MenuItem).where(MenuItem.id == item_id)
    )

    item = result.scalar_one_or_none()

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )

    data = payload.model_dump(exclude_unset=True)

    for field, value in data.items():
        setattr(item, field, value)

    await db.commit()

    await db.refresh(item)

    return {
        "success": True,
        "item": item
    }


@router.get("/menu/{menu_id}/items")
async def get_menu_items(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(Menu).where(Menu.id == menu_id)
    )
    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(status_code=404, detail="Menu not found")

    result = await db.execute(
        select(MenuItem)
        .where(MenuItem.menu_id == menu_id)
        .options(selectinload(MenuItem.variants))
    )
    items = result.scalars().all()

    print(f"Total items found: {len(items)}")

    items_data = []

    for item in items:
        print(
            f"Item ID: {item.id}, Name: {item.product_name}, Variants count: {len(item.variants)}")

        if item.variants:
            for v in item.variants:
                print(f"  -> Variant: size={v.size}, price={v.price}")
                items_data.append({
                    "id": item.id,
                    "product_name": item.product_name,
                    "arabic_name": item.arabic_name,
                    "category": item.category,
                    "description": item.description,
                    "unit": item.unit,
                    "priceLabel": f"{v.size}: {v.price}",
                    "has_photo_on_menu": item.has_photo_on_menu,
                    "image_file": item.image_file,
                    "product_name_confidence": item.product_name_confidence,
                    "price_confidence": item.price_confidence,
                    "arabic_name_confidence": item.arabic_name_confidence,
                    "menu_image_id": item.menu_image_id

                })
        else:
            print(f"  -> NO VARIANTS for item {item.id}")
    print(items_data)
    return {
        "success": True,
        "menu": items_data,
        "debug_total_items": len(items)  # temporary, testing ke liye
    }


@router.patch("/menu-item/complete/{item_id}")
async def update_menu_complete_item(
    item_id: int,
    payload: MenuItemUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MenuItemComplete).where(MenuItemComplete.id == item_id)
    )

    item = result.scalar_one_or_none()

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )

    data = payload.model_dump(exclude_unset=True)

    for field, value in data.items():
        setattr(item, field, value)

    await db.commit()

    await db.refresh(item)

    return {
        "success": True,
        "item": item
    }


@router.get("/menu/complete/{menu_id}/items")
async def get_menu_complete_items(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(Menu).where(Menu.id == menu_id)
    )
    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(status_code=404, detail="Menu not found")

    result = await db.execute(
        select(MenuItemComplete)
        .where(MenuItemComplete.menu_id == menu_id)
        .options(selectinload(MenuItemComplete.variants))
    )
    items = result.scalars().all()

    print(f"Total items found: {len(items)}")

    items_data = []

    for item in items:
        print(
            f"Item ID: {item.id}, Name: {item.product_name}, Variants count: {len(item.variants)}")

        if item.variants:
            for v in item.variants:
                print(f"  -> Variant: size={v.size}, price={v.price}")
                items_data.append({
                    "id": item.id,
                    "product_name": item.product_name,
                    "arabic_name": item.arabic_name,
                    "category": item.category,
                    "description": item.description,
                    "unit": item.unit,
                    "priceLabel": f"{v.size}: {v.price}",
                    "has_photo_on_menu": item.has_photo_on_menu,
                    "image_file": item.image_file,
                    "product_name_confidence": item.product_name_confidence,
                    "price_confidence": item.price_confidence,
                    "arabic_name_confidence": item.arabic_name_confidence,
                    "menu_image_id": item.menu_image_id
                })
        else:
            print(f"  -> NO VARIANTS for item {item.id}")
    print(items_data)
    return {
        "success": True,
        "menu": items_data,
        "debug_total_items": len(items)  # temporary, testing ke liye
    }


@router.get("/menu/{menu_id}/export")
async def export_menu_csv(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):
    # Check menu exists
    result = await db.execute(
        select(Menu).where(
            Menu.id == menu_id
        )
    )

    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(
            status_code=404,
            detail="Menu not found"
        )

    # Get menu items with variants
    result = await db.execute(
        select(MenuItem)
        .options(
            selectinload(MenuItem.variants)
        )
        .where(
            MenuItem.menu_id == menu_id
        )
    )

    items = result.scalars().all()

    # Create CSV in memory
    buffer = io.StringIO()

    writer = csv.writer(buffer)

    writer.writerow([
        "Product Name",
        "Arabic Name",
        "Category",
        "Description",
        "Unit",
        "Price",
        "Image URL"
    ])

    for item in items:

        variants = item.variants or []

        # MenuItemVariant objects
        price_label = " · ".join(
            (
                f"{variant.size}: {variant.price}"
                if variant.size
                else f"{variant.price}"
            )
            for variant in variants
        )

        # Build image URL
        image_url = ""

        if item.image_file:
            if item.image_file.startswith("http"):
                image_url = item.image_file
            else:
                image_url = f"{BASE_URL}/{item.image_file}"

        writer.writerow([
            item.product_name or "",
            item.arabic_name or "",
            item.category or "",
            item.description or "",
            item.unit or "",
            price_label,
            image_url
        ])

    buffer.seek(0)

    filename = f"menu_{menu_id}_export.csv"

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/menu/complete/{menu_id}/export")
async def export_menu_csv(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):
    # Check menu exists
    result = await db.execute(
        select(Menu).where(Menu.id == menu_id)
    )
    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(
            status_code=404,
            detail="Menu not found"
        )

    # Get menu items with variants
    result = await db.execute(
        select(MenuItemComplete)
        .options(selectinload(MenuItemComplete.variants))
        .where(MenuItemComplete.menu_id == menu_id)
    )
    items = result.scalars().all()

    # Create CSV in memory
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow([
        "Product ID/Barcode",
        "Name*",
        "Unit*",
        "Product Category ID*",
        "Product Type*",
        "Sales Price",
        "Purchase Price",
        "Cost Price",
        "Description",
        "Arabic Name",
        "Scan by Price(Y/N)",
        "Scan by Weight(Y/N)",
        "Is Digital Menu(Y/N)",
        "Is Salable(Y/N)",
        "Is Price Editable(Y/N)",
        "Terminal",
        "Brand",
        "Discount",
        "Start Date(YYYY-MM-DD)",
        "End Date(YYYY-MM-DD)",
        "Supplier Code",
        "Product Reference",
        "Is Serialized",
        "Start No.",
        "Prefix",
        "Is MRP Enabled?",
        "MRP Price",
        "Is Batch",
        "Batch Start No",
        "Batch Prefix",
        "Min Ord Qty",
        "Low Stock Alert",
    ])

    def build_row(product_id, name, unit, category, description, arabic_name, price):
        return [
            product_id,             # Product ID/Barcode
            name,                   # Name*
            unit or "",             # Unit*
            category or "",         # Product Category ID*
            "Item",                     # Product Type*
            price if price is not None else "",  # Sales Price
            0,                     # Purchase Price
            0,                     # Cost Price
            description or "",      # Description
            arabic_name or "",      # Arabic Name
            "N",                    # Scan by Price(Y/N)
            "N",                    # Scan by Weight(Y/N)
            "Y",                    # Is Digital Menu(Y/N)
            "Y",                    # Is Salable(Y/N)
            "N",                    # Is Price Editable(Y/N)
            "",                     # Terminal
            "",                     # Brand
            0,                     # Discount
            "",                     # Start Date
            "",                     # End Date
            "",                     # Supplier Code
            "",                     # Product Reference
            "N",                    # Is Serialized
            "",                     # Start No.
            "",                     # Prefix
            "N",                    # Is MRP Enabled?
            0,                     # MRP Price
            "N",                    # Is Batch
            "",                     # Batch Start No
            "",                     # Batch Prefix
            "",                     # Min Ord Qty
            "",                     # Low Stock Alert
        ]

    for item in items:
        variants = item.variants or []

        if not variants:
            writer.writerow(build_row(
                product_id=item.id,
                name=item.product_name or "",
                unit=item.unit,
                category=item.category,
                description=item.description,
                arabic_name=item.arabic_name,
                price=None,
            ))
        else:
            for variant in variants:
                # Size wise naam alag, aur ID bhi unique rakhne ke liye
                # variant id append kiya (agar variant model mein id hai)
                name = item.product_name or ""
                if variant.size:
                    name = f"{name} ({variant.size})"

                product_id = item.id

                writer.writerow(build_row(
                    product_id=product_id,
                    name=name,
                    unit=item.unit,
                    category=item.category,
                    description=item.description,
                    arabic_name=item.arabic_name,
                    price=variant.price,
                ))

    buffer.seek(0)
    filename = f"menu_{menu_id}_export.csv"

    csv_content = "\ufeff" + buffer.getvalue()

    return StreamingResponse(
        iter([csv_content.encode("utf-8-sig")]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/menu/{menu_id}/images")
async def get_menu_images(
    menu_id: int,
    db: AsyncSession = Depends(get_db),
):
    # Check menu exists
    result = await db.execute(
        select(Menu).where(Menu.id == menu_id)
    )
    menu = result.scalar_one_or_none()

    if not menu:
        raise HTTPException(status_code=404, detail="Menu not found")

    # Get all images linked to this menu
    result = await db.execute(
        select(MenuImage).where(MenuImage.menu_id == menu_id)
    )
    images = result.scalars().all()

    images_data = []

    for image in images:
        # Build full image URL (agar already http(s) hai to as-is chorho)
        if image.path and image.path.startswith("http"):
            image_url = image.path
        elif image.path:
            image_url = f"{BASE_URL}/{image.path.lstrip('/')}"
        else:
            image_url = None

        images_data.append({
            "id": image.id,
            "menu_id": image.menu_id,
            "image_url": image_url,
        })

    return {
        "success": True,
        "menu_id": menu_id,
        "images": images_data,
        "total_images": len(images_data),
    }
