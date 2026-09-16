import os
import uuid

from PIL import Image
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import MenuImage


OUTPUT_DIR = "uploads/images/pdf_pages"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


async def convert_image_to_png(
    file: UploadFile,
    menu_id: int,
    image_number: int,
    db: AsyncSession,
):
    file_id = str(uuid.uuid4())

    original_filename = file.filename.lower()

    output_filename = f"{file_id}_page_{image_number}.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    content = await file.read()

    temp_path = os.path.join(
        OUTPUT_DIR,
        f"temp_{file_id}"
    )

    with open(temp_path, "wb") as buffer:
        buffer.write(content)

    converted = False

    # Already PNG
    if original_filename.endswith(".png"):

        with open(output_path, "wb") as output:
            output.write(content)

    # Other image formats -> PNG
    else:
        converted = True

        with Image.open(temp_path) as image:

            if image.mode in ("RGBA", "LA"):
                image.save(output_path, "PNG")

            else:
                image.convert("RGB").save(
                    output_path,
                    "PNG"
                )

    os.remove(temp_path)

    # -------------------------
    # SAVE IMAGE IN DATABASE
    # -------------------------

    menu_image = MenuImage(
        menu_id=menu_id,
        image_number=image_number,
        path=output_path,
    )

    db.add(menu_image)
    await db.flush()  # <-- yeh line add ki, taake menu_image.id generate ho jaye

    return {
        "original_filename": file.filename,
        "converted": converted,
        "format": "PNG",
        "filename": output_filename,
        "path": output_path,
        "image_number": image_number,
        "menu_image_id": menu_image.id,  # <-- ab yeh key available hogi
    }
