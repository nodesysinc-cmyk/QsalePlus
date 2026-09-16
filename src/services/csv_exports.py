import io
import csv


def build_items_csv_bytes(items: list[dict], base_url: str | None = None) -> bytes:
    """
    items_before_merge_final (ya koi bhi list of dict jiski shape extraction
    items jaisi ho) ko CSV bytes mein convert karta hai — email attachment
    ke liye ready.

    Expected dict shape (agar key na mile to khaali chhod deta hai):
        {
            "product_name": str,
            "arabic_name": str,
            "category": str,
            "description": str,
            "unit": str,
            "variants": [{"size": str, "price": float}, ...],
            "image_file": str,
        }
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow([
        "Product Name",
        "Arabic Name",
        "Category",
        "Description",
        "Unit",
        "Price",
        "Image URL",
    ])

    for item in items or []:
        variants = item.get("variants") or []

        price_label = " · ".join(
            (
                f"{v.get('size')}: {v.get('price')}"
                if v.get("size")
                else f"{v.get('price')}"
            )
            for v in variants
        )

        image_file = item.get("image_file") or ""
        image_url = ""
        if image_file:
            if image_file.startswith("http"):
                image_url = image_file
            elif base_url:
                image_url = f"{base_url}/{image_file}"
            else:
                image_url = image_file

        writer.writerow([
            item.get("product_name") or "",
            item.get("arabic_name") or "",
            item.get("category") or "",
            item.get("description") or "",
            item.get("unit") or "",
            price_label,
            image_url,
        ])

    # UTF-8 BOM taake Excel Arabic text sahi dikhaye
    return buffer.getvalue().encode("utf-8-sig")
