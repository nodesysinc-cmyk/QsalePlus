"""
MERGED VERSION: Vision AI Structured Extraction
  + DB-based Image Linking (PageCoordinates table)
  + LOCAL EMBEDDING DEDUP (offline, no extra Claude call)
  + DB SAVE (MenuItem + MenuItemVariant tables)
===========================================================================
... (docstring wesa hi, sirf ye add: ab final (post-dedup) items JSON
     return karne ke sath-sath seedha `MenuItem` + `MenuItemVariant`
     DB tables mein bhi save ho jate hain. `extract()` ab `menu_id`
     bhi leta hai - taake har MenuItem row us menu se link ho.)
"""

import asyncio
import base64
import copy
import json
import os
from urllib.parse import urljoin
import io
import base64
from PIL import Image
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# --- DB imports: project ke actual pattern ke mutabiq ---
from src.db.models import PageCoordinates, MenuItem, MenuItemVariant, MenuItemComplete, MenuItemCompleteVariant
from src.db import get_db  # sirf __main__/standalone test ke liye chahiye hoga

load_dotenv()


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

IMAGE_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# EXTRACTION PROMPT (wesa hi, koi change nahi)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert menu-digitization AI. You will be shown an image of one page from a restaurant/cafe menu (photo or scanned PDF page).

Carefully read the ENTIRE page and extract every product/menu item you can see, following these exact rules:

1. Product Name: extract exactly as printed (English). If the page itself is only in Arabic and no English name is printed anywhere, transliterate/translate the Arabic into a natural English product name yourself — product_name must NEVER be null or empty either. Lower its confidence accordingly if you had to translate it.
2. Arabic Name: extract exactly as printed if present. If Arabic is NOT printed anywhere on this page for this item, YOU must generate an accurate, natural Arabic translation of the product name yourself. arabic_name must NEVER be null, empty, or omitted — always provide a real value, either read from the page or translated by you.
3. Category: the section heading this item falls under (e.g. "Burgers", "Beverages"). If no explicit heading, infer the most likely category from context.
4. Description: extract if present; otherwise null.
5. Sizes & Prices: if a product has multiple sizes (Small/Medium/Large/Regular etc.), return ONE product entry with a "variants" array containing each size and its price — do NOT create separate products for each size.
   If there is only one size, still use a "variants" array with a single entry, size = "Regular".
6. Unit: default to "Each" unless explicitly stated otherwise.
7. has_photo_on_menu: true if this specific item has a real photo printed next to it on this page, false otherwise.
8. Detect the page's language(s) automatically. Keep English and Arabic as separate fields — never merge them into one string.
9. For each of these fields, give a confidence score 0-100 based on how clearly it was printed/legible: product_name, price, arabic_name.
10. If unsure about a value, still provide your best guess but lower the confidence score accordingly — never omit a field.
11. MANDATORY — confidence for translated/generated fields: if product_name or arabic_name was NOT printed on the page and you generated/translated/transliterated it yourself, that field's confidence score MUST ALWAYS be below 50 (e.g. 20-49), with NO exceptions, regardless of how confident you feel about the translation quality. This is a hard rule, not a suggestion. Only if the value was ACTUALLY printed on the page and clearly legible should it receive a normal/high score. Never assign 50 or above to a translated/generated name.

12. IMAGE LINKING (compare against pre-detected candidates):
    After this instruction, you will also receive a JSON array called "candidate_photos" — this lists photo detections that were ALREADY found on this same page by a separate detection pipeline (object detector + OCR), independent of your own reading. Each candidate has:
      - "image_file": the full URL to an already-cropped photo file
      - "matched_text": the single closest text-line that was found near that photo
      - "raw_text": a combined string of ALL nearby text found around that photo (may include the item's name, price, and description together)
      - "detected_label": what the detector classified that region as (e.g. "food photo")

    This "candidate_photos" list was built independently of your own extraction, using a different method (object detection + OCR, not vision-language reading). Your job is to CROSS-CHECK your own extraction against it:

    For every item where you set has_photo_on_menu to true, compare BOTH your extracted "product_name" AND your extracted "description" against each candidate's "matched_text" and "raw_text":
      a) If your product_name and/or description clearly correspond to (appear in, or are a close paraphrase/substring match of) a candidate's matched_text or raw_text, that candidate is a match.
      b) Matching on description content counts just as much as matching on product_name — sometimes the name is short/generic (e.g. "Burger") but the description text (ingredients, style) is what confirms the correct candidate among several similar ones.
      c) If more than one candidate seems plausible, pick the one with the strongest textual overlap across both product_name and description combined.
      d) If no candidate has a clear, confident textual correspondence to this item's product_name/description, do NOT force a match — leave it unmatched.

    Only if a candidate is confidently matched: set this item's "image_file" field to EXACTLY that candidate's "image_file" value (copy the URL as given, do not alter it).
    If has_photo_on_menu is false: ALWAYS set "image_file" to null, and do not attempt any comparison for that item.
    If has_photo_on_menu is true but no candidate confidently matches: set "image_file" to null.
    Each candidate's "image_file" should generally be used for at most ONE item — do not assign the same "image_file" to multiple different products unless you are confident the page genuinely repeats the same photo.

13. FINAL VERIFICATION PASS: Before returning your answer, re-scan the ENTIRE page image one more time from top to bottom, section by section, and cross-check against the item list you have built so far. Specifically check for:
    a) Any item, dish, or drink name printed anywhere on the page (including small print, corners, side panels, or combo/set sections) that is NOT yet in your "items" list — add it if missing.
    b) Any category/section on the page that has zero items extracted from it — go back and re-read that section carefully.
    c) Any item where you can see a price on the page but it wasn't captured in "variants" — go back and add it.
    d) Do not skip faint, small, or partially-obscured text — attempt to read it before giving up.
    Only after this verification pass is complete should you finalize and return the JSON. Do not mention this verification process in your output — just ensure the final "items" list is as complete and accurate as possible.

Return ONLY valid JSON (no markdown fences, no commentary, no preamble) in exactly this shape:

{
  "items": [
    {
      "product_name": "Chicken Burger",
      "arabic_name": "برجر دجاج",
      "category": "Burgers",
      "description": "Grilled chicken breast with lettuce and mayo",
      "variants": [
        {"size": "Regular", "price": 18.00},
        {"size": "Large", "price": 22.00}
      ],
      "unit": "Each",
      "has_photo_on_menu": true,
      "image_file": "http://localhost:8000/uploads/images/itemImages/chicken_burger.jpg",
      "confidence": {
        "product_name": 96,
        "price": 98,
        "arabic_name": 62
      }
    }
  ]
}
"""


class MenuDataExtractor:
    """
    Har page-image + uske "menu_image_id" ki jodi leta hai, Claude se
    structured items nikalta hai, local-embedding dedup karta hai, aur
    final items seedha `MenuItem` + `MenuItemVariant` DB tables mein
    save kar deta hai.
    """

    _client = None
    _embedder = None

    CROSS_CASE_THRESHOLD = 0.81
    ARABIC_PRINTED_THRESHOLD = 50
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "[ERROR] .env mein ANTHROPIC_API_KEY missing hai.\n"
                "        Isi folder mein .env file mein ye add karo:\n"
                "        ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx"
            )

        if MenuDataExtractor._client is None:
            MenuDataExtractor._client = AsyncAnthropic(api_key=api_key)

        self.client = MenuDataExtractor._client

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _encode_image(path: str) -> str:
        img = Image.open(path)

        # Claude ke liye RGB JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Large PDF page images ko resize karo
        max_dimension = 3000

        if max(img.size) > max_dimension:
            img.thumbnail(
                (max_dimension, max_dimension),
                Image.Resampling.LANCZOS
            )

        # Compress until comfortably below Claude's 10 MB limit
        quality = 85

        while True:
            buffer = io.BytesIO()

            img.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True
            )

            size_mb = buffer.tell() / (1024 * 1024)

            # 7 MB target rakho, 10 MB limit se kaafi neeche
            if size_mb <= 7 or quality <= 40:
                break

            quality -= 5

        print(
            f"[IMAGE] {path} -> "
            f"{img.width}x{img.height}, "
            f"{size_mb:.2f} MB, "
            f"quality={quality}"
        )

        return base64.standard_b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    @staticmethod
    def _media_type_for(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    @staticmethod
    def _to_full_url(path_or_url: str | None) -> str | None:
        if not path_or_url:
            return None
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        normalized = path_or_url.replace("\\", "/").lstrip("/")
        return urljoin(IMAGE_BASE_URL.rstrip("/") + "/", normalized)

    @staticmethod
    def _truncate(value, max_len: int):
        """DB column ki max-length se zyada aana par truncate kar dete
        hain, taake DB insert error na aaye."""
        if value is None:
            return None
        value = str(value)
        return value if len(value) <= max_len else value[:max_len]

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Loading local embedding model '{cls.EMBED_MODEL_NAME}' "
                  f"(pehli baar thoda time lagega)...")
            cls._embedder = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls._embedder

    # -----------------------------------------------------------------
    # STEP: DB se candidate_photos fetch karna
    # -----------------------------------------------------------------

    async def _fetch_candidate_photos(self, db: AsyncSession, menu_image_id: int) -> list:
        result = await db.execute(
            select(PageCoordinates).where(
                PageCoordinates.menu_image_id == menu_image_id
            )
        )
        rows = result.scalars().all()

        candidates = [
            {
                "image_file": self._to_full_url(row.image_file),
                "matched_text": row.matched_text,
                "raw_text": row.raw_text,
                "detected_label": row.detected_label,
            }
            for row in rows
        ]
        print(f"[..] {len(candidates)} candidate photo(s) found "
              f"for menu_image_id={menu_image_id}")
        return candidates

    # -----------------------------------------------------------------
    # STEP: ek page Claude ko bhejna
    # -----------------------------------------------------------------

    async def _extract_one(self, image_path: str, menu_image_id: int, candidates: list) -> list:
        candidates_json = json.dumps(
            {"candidate_photos": candidates}, indent=2, ensure_ascii=False
        )

        media_type = "image/jpeg"
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": self._encode_image(image_path),
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "text",
                            "text": (
                                "Here is the candidate_photos data for THIS "
                                "page, to use for image linking as described "
                                f"in rule 12:\n\n{candidates_json}"
                            ),
                        },
                    ],
                }
            ],
        )

        raw_text = "".join(
            block.text for block in message.content if block.type == "text")
        cleaned = raw_text.strip().removeprefix(
            "```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            print(
                f"[WARN] JSON parse fail hui '{image_path}' ke liye, raw text save kar rahe")
            parsed = {"items": [], "_raw_error_text": raw_text}

        for item in parsed.get("items", []):
            item["_source_page_path"] = image_path
            item["_menu_image_id"] = menu_image_id

            has_photo = bool(item.get("has_photo_on_menu"))
            if not has_photo:
                item["image_file"] = None
            elif item.get("image_file"):
                item["image_file"] = self._to_full_url(item["image_file"])
            else:
                item["image_file"] = None

            if not item.get("arabic_name"):
                item.setdefault("confidence", {})["arabic_name"] = 0
            if not item.get("product_name"):
                item.setdefault("confidence", {})["product_name"] = 0

        return parsed.get("items", [])

    # -----------------------------------------------------------------
    # STEP: field-level merge
    # -----------------------------------------------------------------

    @staticmethod
    def _merge_two_items(primary: dict, duplicate: dict) -> dict:
        p_conf = dict(primary.get("confidence") or {})
        d_conf = dict(duplicate.get("confidence") or {})

        for field in ("product_name", "arabic_name"):
            p_val = primary.get(field)
            d_val = duplicate.get(field)
            p_score = p_conf.get(field) or 0
            d_score = d_conf.get(field) or 0

            if not p_val and d_val:
                primary[field] = d_val
                p_conf[field] = d_score
            elif p_val and d_val and d_score > p_score:
                primary[field] = d_val
                p_conf[field] = d_score

        for field in ("description", "category", "unit"):
            if not primary.get(field) and duplicate.get(field):
                primary[field] = duplicate[field]

        if not primary.get("image_file") and duplicate.get("image_file"):
            primary["image_file"] = duplicate["image_file"]

        primary["has_photo_on_menu"] = bool(
            primary.get("has_photo_on_menu")) or bool(duplicate.get("has_photo_on_menu"))

        if not primary.get("has_photo_on_menu"):
            primary["image_file"] = None

        primary_has_prices = any(
            v.get("price") is not None for v in primary.get("variants") or [])
        duplicate_has_prices = any(
            v.get("price") is not None for v in duplicate.get("variants") or [])
        if not primary_has_prices and duplicate_has_prices:
            primary["variants"] = duplicate["variants"]

        if "price" in d_conf and (d_conf.get("price") or 0) > (p_conf.get("price") or 0):
            p_conf["price"] = d_conf["price"]

        primary["confidence"] = p_conf

        primary_pages = primary.get("_source_page_paths") or (
            [primary["_source_page_path"]] if primary.get("_source_page_path") else [])
        duplicate_pages = duplicate.get("_source_page_paths") or (
            [duplicate["_source_page_path"]] if duplicate.get("_source_page_path") else [])
        primary["_source_page_paths"] = list(
            dict.fromkeys(primary_pages + duplicate_pages))
        primary.pop("_source_page_path", None)

        return primary

    # -----------------------------------------------------------------
    # STEP: LOCAL embedding-based dedup
    # -----------------------------------------------------------------

    def _merge_duplicates_by_embedding(self, all_items: list) -> list:
        for item in all_items:
            if "_source_page_paths" not in item:
                item["_source_page_paths"] = [
                    item.pop("_source_page_path", None)]

        if len(all_items) < 2:
            return all_items

        embedder = self._get_embedder()

        arabic_names = [
            f"query: {(item.get('arabic_name') or '').strip()}"
            for item in all_items
        ]
        embeddings = embedder.encode(
            arabic_names,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        import numpy as np
        sim_matrix = embeddings @ embeddings.T

        n = len(all_items)

        def overall_conf(item):
            c = item.get("confidence") or {}
            vals = [c.get("product_name") or 0,
                    c.get("arabic_name") or 0, c.get("price") or 0]
            return sum(vals) / len(vals)

        def arabic_conf(item):
            return (item.get("confidence") or {}).get("arabic_name") or 0

        merged_away = set()

        for i in range(n):
            if i in merged_away:
                continue
            for j in range(i + 1, n):
                if j in merged_away:
                    continue

                conf_i = arabic_conf(all_items[i])
                conf_j = arabic_conf(all_items[j])

                i_is_printed = conf_i >= self.ARABIC_PRINTED_THRESHOLD
                j_is_printed = conf_j >= self.ARABIC_PRINTED_THRESHOLD

                if not i_is_printed and not j_is_printed:
                    continue
                if i_is_printed and j_is_printed:
                    continue

                required_sim = self.CROSS_CASE_THRESHOLD

                if sim_matrix[i][j] >= required_sim:
                    item_i, item_j = all_items[i], all_items[j]
                    if overall_conf(item_i) >= overall_conf(item_j):
                        primary_idx, dup_idx = i, j
                    else:
                        primary_idx, dup_idx = j, i

                    self._merge_two_items(
                        all_items[primary_idx], all_items[dup_idx])
                    merged_away.add(dup_idx)

        final_items = [item for idx, item in enumerate(
            all_items) if idx not in merged_away]
        return final_items

    # -----------------------------------------------------------------
    # STEP: final cleanup
    # -----------------------------------------------------------------

    @staticmethod
    def _finalize_item(item: dict) -> dict:
        item = dict(item)
        item["menu_image_id"] = item.pop("_menu_image_id", None)
        item.pop("_source_page_path", None)
        item.pop("_source_page_paths", None)
        return item

    @classmethod
    def _finalize_items(cls, items: list) -> list:
        return [cls._finalize_item(item) for item in items]

    # -----------------------------------------------------------------
    # STEP: DB SAVE - final items ko MenuItem + MenuItemVariant table
    # mein save karna
    # -----------------------------------------------------------------

    async def _save_items_to_db(self, db: AsyncSession, items: list, menu_id: int) -> list:
        """
        Har item ke liye:
          1) `MenuItem` row banti hai (confidence dict se teeno
             confidence-fields nikal ke apne column mein), db.add()
             ke baad `flush()` (taake `id` mil jaye, `commit()` abhi
             nahi - poore batch ke liye ek hi commit aakhir mein hota
             hai)
          2) uske "variants" array se `MenuItemVariant` rows banti
             hain (`menu_item_id` abhi-mile id se set ho kar)

        Return: saved items ki list - har item mein DB-assigned "id"
        aur DB-saved "variants" (id ke sath) add kiya hua.
        """
        saved_items = []

        for item in items:
            confidence = item.get("confidence") or {}

            menu_item = MenuItem(
                menu_id=menu_id,
                menu_image_id=item.get("menu_image_id"),
                product_name=self._truncate(
                    item.get("product_name"), 255) or "Unnamed Item",
                arabic_name=self._truncate(item.get("arabic_name"), 255),
                category=self._truncate(item.get("category"), 100),
                description=item.get("description"),
                unit=self._truncate(item.get("unit"), 50),
                has_photo_on_menu=bool(item.get("has_photo_on_menu")),
                image_file=self._truncate(item.get("image_file"), 500),
                product_name_confidence=confidence.get("product_name"),
                price_confidence=confidence.get("price"),
                arabic_name_confidence=confidence.get("arabic_name"),
                source=item.get("_source_page_path")
            )
            db.add(menu_item)
            await db.flush()  # menu_item.id yahan se milega

            variant_rows = []
            for variant in item.get("variants") or []:
                variant_row = MenuItemVariant(
                    menu_item_id=menu_item.id,
                    size=self._truncate(variant.get("size"), 100),
                    price=variant.get("price"),
                )
                db.add(variant_row)
                variant_rows.append(variant_row)

            if variant_rows:
                await db.flush()  # variant ids bhi mil jayen

            saved_item = dict(item)
            saved_item["id"] = menu_item.id
            saved_item["variants"] = [
                {"id": v.id, "size": v.size, "price": v.price}
                for v in variant_rows
            ]
            saved_items.append(saved_item)

        await db.commit()  # poora batch ek sath commit

        print(f"[DB] {len(saved_items)} menu item(s) saved to DB "
              f"(menu_id={menu_id})")
        return saved_items

    async def _save_items_to_complete_db(self, db: AsyncSession, items: list, menu_id: int) -> list:
        """
        Har item ke liye:
          1) `MenuItem` row banti hai (confidence dict se teeno
             confidence-fields nikal ke apne column mein), db.add()
             ke baad `flush()` (taake `id` mil jaye, `commit()` abhi
             nahi - poore batch ke liye ek hi commit aakhir mein hota
             hai)
          2) uske "variants" array se `MenuItemVariant` rows banti
             hain (`menu_item_id` abhi-mile id se set ho kar)

        Return: saved items ki list - har item mein DB-assigned "id"
        aur DB-saved "variants" (id ke sath) add kiya hua.
        """
        saved_items = []

        for item in items:
            confidence = item.get("confidence") or {}

            menu_item = MenuItemComplete(
                menu_id=menu_id,
                menu_image_id=item.get("menu_image_id"),
                product_name=self._truncate(
                    item.get("product_name"), 255) or "Unnamed Item",
                arabic_name=self._truncate(item.get("arabic_name"), 255),
                category=self._truncate(item.get("category"), 100),
                description=item.get("description"),
                unit=self._truncate(item.get("unit"), 50),
                has_photo_on_menu=bool(item.get("has_photo_on_menu")),
                image_file=self._truncate(item.get("image_file"), 500),
                product_name_confidence=confidence.get("product_name"),
                price_confidence=confidence.get("price"),
                arabic_name_confidence=confidence.get("arabic_name"),
                source=item.get("_source_page_path")

            )
            db.add(menu_item)
            await db.flush()  # menu_item.id yahan se milega

            variant_rows = []
            for variant in item.get("variants") or []:
                variant_row = MenuItemCompleteVariant(
                    menu_item_id=menu_item.id,
                    size=self._truncate(variant.get("size"), 100),
                    price=variant.get("price"),
                )
                db.add(variant_row)
                variant_rows.append(variant_row)

            if variant_rows:
                await db.flush()  # variant ids bhi mil jayen

            saved_item = dict(item)
            saved_item["id"] = menu_item.id
            saved_item["variants"] = [
                {"id": v.id, "size": v.size, "price": v.price}
                for v in variant_rows
            ]
            saved_items.append(saved_item)

        await db.commit()  # poora batch ek sath commit

        print(f"[DB] {len(saved_items)} menu item(s) saved to DB "
              f"(menu_id={menu_id})")
        return saved_items

    # -----------------------------------------------------------------
    # MAIN: batch - saari pages, dedup, phir DB save
    # -----------------------------------------------------------------

    async def extract(self, db: AsyncSession, menu_id: int, pages: list[tuple[str, int]],
                      output_path: str | None = None,
                      merge_duplicates: bool = True) -> dict:
        """
        db: router se aayi request-scoped AsyncSession.
        menu_id: is extraction-run ka parent `Menu` record ka id -
            har saved `MenuItem` isi se link hoga.
        pages: list of (image_path, menu_image_id) tuples.
        output_path: (optional) result JSON file mein save karna ho to.
        merge_duplicates: True (default) -> local embedding dedup pass.

        Return value (dict):
            "items"              -> final list, DB-saved (id + variant
                                     ids ke sath)
            "items_before_merge" -> raw list, dedup se PEHLE (DB mein
                                     save NAHI hoti, sirf reference ke
                                     liye)
        """
        missing = [p for p, _ in pages if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"[ERROR] Ye image path(s) nahi milin: {missing}"
            )

        # PHASE 1 - candidates fetch SEQUENTIALLY
        candidates_per_page = []
        for path, menu_image_id in pages:
            candidates = await self._fetch_candidate_photos(db, menu_image_id)
            candidates_per_page.append(candidates)

        # PHASE 2 - Claude calls CONCURRENTLY
        results = await asyncio.gather(
            *[
                self._extract_one(path, menu_image_id, candidates)
                for (path, menu_image_id), candidates
                in zip(pages, candidates_per_page)
            ]
        )
        all_items = [item for sub in results for item in sub]
        print(f"\nTotal items extracted (pre-merge): {len(all_items)}")

        for item in all_items:
            if "_source_page_paths" not in item:
                item["_source_page_paths"] = [
                    item.pop("_source_page_path", None)]

        items_before_merge = copy.deepcopy(all_items)

        if merge_duplicates:
            all_items = self._merge_duplicates_by_embedding(all_items)
            print(f"Total items after local-embedding dedup: {len(all_items)}")

        final_items = self._finalize_items(all_items)
        items_before_merge_final = self._finalize_items(items_before_merge)

        # --- DB SAVE: final (post-dedup) items ---
        saved_items = await self._save_items_to_db(db, final_items, menu_id)
        saved_items_complete = await self. _save_items_to_complete_db(db, items_before_merge_final, menu_id)

        result = {
            "items": saved_items,
            "items_before_merge": items_before_merge_final,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved -> {output_path}")

        return result


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def main():
        pages = [
            ("page_2.png", 101),
            ("page_4.png", 102),
        ]
        menu_id = 1  # test ke liye

        extractor = MenuDataExtractor()

        async for db in get_db():
            result = await extractor.extract(
                db, menu_id, pages, output_path="output/merged_result.json")
            break

        print("\n--- FINAL (post-merge, image-linked, DB-saved) ---")
        print(json.dumps(result["items"], indent=2, ensure_ascii=False))

        print("\n--- ORIGINAL (pre-merge) ---")
        print(json.dumps(result["items_before_merge"],
              indent=2, ensure_ascii=False))

    asyncio.run(main())
