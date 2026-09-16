import os
import uuid
import fitz

from PIL import Image
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Menu, MenuImage


UPLOAD_DIR = "uploads/pdfs"
OUTPUT_DIR = "uploads/images/pdf_pages"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def convert_pdf_to_images(
    file: UploadFile,
    restaurant_id: int,
    menu_type: str,
    menu_number: int,
    db: AsyncSession,
):
    # Unique ID for this PDF
    file_id = str(uuid.uuid4())

    # -------------------------
    # SAVE PDF
    # -------------------------

    pdf_filename = f"{file_id}.pdf"

    pdf_path = os.path.join(
        UPLOAD_DIR,
        pdf_filename
    )

    content = await file.read()

    with open(pdf_path, "wb") as buffer:
        buffer.write(content)

    # -------------------------
    # SAVE MENU IN DATABASE
    # -------------------------

    menu = Menu(
        restaurant_id=restaurant_id,
        menu_type=menu_type,
        menu_number=menu_number,
        path=pdf_path,
    )

    db.add(menu)
    await db.flush()

    # -------------------------
    # OPEN PDF
    # -------------------------

    document = fitz.open(pdf_path)

    images = []

    # PDF-render zoom aur JPEG quality - .env se control ho sakte hain
    render_zoom = float(os.getenv("PDF_RENDER_ZOOM", "1.5"))
    jpeg_quality = int(os.getenv("PDF_IMAGE_JPEG_QUALITY", "85"))

    # -------------------------
    # CONVERT PAGES TO IMAGES
    # -------------------------

    for page_number, page in enumerate(
        document,
        start=1
    ):

        pix = page.get_pixmap(
            matrix=fitz.Matrix(render_zoom, render_zoom)
        )

        # PNG ki jagah JPEG - same content, kaafi chhoti file size
        image_name = (
            f"{file_id}_page_{page_number}.jpg"
        )

        image_path = os.path.join(
            OUTPUT_DIR,
            image_name
        )

        # PyMuPDF ka pix.save() JPEG ke liye direct quality-control
        # nahi deta reliably - isliye PIL se convert + compress karte
        # hain (poora control milta hai)
        img = Image.frombytes(
            "RGB" if pix.n < 4 else "RGBA",
            [pix.width, pix.height],
            pix.samples,
        )
        if img.mode == "RGBA":
            img = img.convert("RGB")  # JPEG alpha-channel support nahi karta

        img.save(image_path, "JPEG", quality=jpeg_quality, optimize=True)

        # -------------------------
        # SAVE IMAGE IN DATABASE
        # -------------------------

        menu_image = MenuImage(
            menu_id=menu.id,
            image_number=page_number,
            path=image_path,
        )

        db.add(menu_image)
        await db.flush()
        images.append({
            "page": page_number,
            "filename": image_name,
            "path": image_path,
            "menu_image_id": menu_image.id
        })

    document.close()

    # -------------------------
    # COMMIT EVERYTHING
    # -------------------------

    await db.commit()

    await db.refresh(menu)

    return {
        "success": True,
        "pdf_id": file_id,
        "menu_id": menu.id,
        "total_pages": len(images),
        "pdf_path": pdf_path,
        "images": images,
    }
