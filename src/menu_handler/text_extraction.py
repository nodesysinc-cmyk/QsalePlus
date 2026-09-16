"""
SIMPLE VERSION v4: Vision AI Structured Extraction + LOCAL EMBEDDING DEDUP
===========================================================================
v3 se tabdeeli (v4):

  MERGE AB SIRF "CROSS-CASE" (Printed + Translated) MEIN HOTA HAI
  -----------------------------------------------------------------------
  v3 mein "Printed vs Printed" pairs bhi merge ho jate the agar arabic_name
  similarity high thi. User ne notice kiya ke isse kabhi kabhi do
  GENUINELY ALAG items (dono ka arabic_name page par asal mein printed
  tha) bhi merge ho rahe the - jo galat hai, kyunke agar dono taraf real
  printed text match kar raha hai to bhi auto-merge karna risky hai
  (do alag entries ho sakti hain jo sirf naam/spelling mein milti-julti
  hain). Is liye ab Printed-vs-Printed pairs ko BILKUL SKIP kar diya
  jata hai - merge hi nahi hota, similarity kitni bhi ho.

  Ab sirf EK case mein merge hota hai:
    - Printed vs Translated -> compare + merge hota hai (ye hi asli
      cross-language duplicate case hai jo catch karna tha - English-only
      page ka item jiska arabic_name Claude ne khud translate kiya, aur
      Arabic-only page ka wahi dish jiska arabic_name asal mein printed
      hai). Threshold = CROSS_CASE_THRESHOLD (0.90), strict rakha gaya
      hai kyunke ek side ab bhi guess/translation hai.

  Skip hone wale do cases:
    - Printed vs Printed       -> SKIP (dono real text hain, lekin phir
                                   bhi auto-merge nahi karte - false-merge
                                   ka risk, do genuinely alag items ho
                                   sakti hain)
    - Translated vs Translated -> SKIP (dono guess hain, unreliable
                                   generic-wording match, false-positive
                                   ka sabse bada source)

  Baaki sab kuch pehle jaisa hi hai: concurrency cap nahi, translation
  fallback prompt mein hi hai, dedup local/offline embeddings se
  (koi extra Claude API call nahi).

  NAYI TABDEELI (is patch mein): ab extract() function final
  (post-merge) items ke sath sath ORIGINAL list (yani merge/dedup se
  PEHLE wali raw extracted items list, jo har image se seedha aayi
  thi) bhi return karta hai. Dono lists alag keys mein milti hain:
      result["items"]              -> final, dedup ke baad
      result["items_before_merge"] -> original, dedup se pehle (raw)
  Agar output_path diya jaye to JSON file mein bhi dono save hoti
  hain.

Model intekhab: `intfloat/multilingual-e5-small` - lightweight
multilingual sentence-embedding model (~470MB, 118M params - bge-m3
se kaafi halka), phir bhi Arabic samet multilingual MTEB benchmarks
par accuracy bge-m3 ke qareeb-qareeb hai. Poora offline/local chalta
hai (sentence-transformers ke zariye). Note: e5-family models "query: "
prefix ke sath train hue hain, is liye behtar accuracy ke liye har
text ko encode karne se pehle "query: " prefix lagaya jata hai (code
mein already handled).

.env mein zaroori:
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx

Setup:
  pip install anthropic python-dotenv sentence-transformers
"""

import asyncio
import base64
import copy
import json
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# EXTRACTION PROMPT (same as pehle - translation fallback rule intact)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert menu-digitization AI. You will be shown an image of one page from a restaurant/cafe menu (photo or scanned PDF page).

Carefully read the ENTIRE page and extract every product/menu item you can see, following these exact rules:

1. Product Name: extract exactly as printed (English). If the page itself is only in Arabic and no English name is printed anywhere, transliterate/translate the Arabic into a natural English product name yourself — product_name must NEVER be null or empty either. Lower its confidence accordingly if you had to translate it.
2. Arabic Name: extract exactly as printed if present on the page. If Arabic is NOT printed anywhere on this page for this item, YOU must generate an accurate, natural Arabic translation of the product name yourself. arabic_name must NEVER be null, empty, or omitted — always provide a real value, either read from the page or translated by you.
3. Category: the section heading this item falls under (e.g. "Burgers", "Beverages"). If no explicit heading, infer the most likely category from context.
4. Description: extract if present; otherwise null.
5. Sizes & Prices: if a product has multiple sizes (Small/Medium/Large/Regular etc.), return ONE product entry with a "variants" array containing each size and its price — do NOT create separate products for each size.
   If there is only one size, still use a "variants" array with a single entry, size = "Regular".
6. Unit: default to "Each" unless explicitly stated otherwise.
7. has_photo_on_menu: true if this specific item has a real photo printed next to it on this page, false otherwise.
8. Detect the page's language(s) automatically. Keep English and Arabic as separate fields — never merge them into one string.
9. For each of these fields, give a confidence score 0-100 based on how clearly it was printed/legible: product_name, price, arabic_name.
10. If unsure about a value, still provide your best guess but lower the confidence score accordingly — never omit a field.
11. IMPORTANT — confidence for translated fields: if product_name or arabic_name was NOT printed on the page and you generated/translated it yourself, that field's confidence score must be below 50 (to clearly mark it as inferred, not read). If it WAS printed on the page and clearly legible, score it normally (high if clear).

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
      "confidence": {
        "product_name": 96,
        "price": 98,
        "arabic_name": 62
      }
    }
  ]
}
"""


class MenuDataExtractorSimple:
    """
    Sirf image paths ki list leta hai, har image Claude (vision) ko
    UNLIMITED-PARALLEL bhejta hai, structured menu-data extract karta
    hai, aur duplicates ko LOCAL embedding similarity (arabic_name
    par, multilingual-e5-small) se merge kar ke return karta hai. Koi
    doosri Claude call dedup ke liye nahi lagti.

    v4 guard rule: merge SIRF cross-case (ek item ka arabic_name
    printed + doosre ka translated) mein hota hai. Printed-vs-Printed
    aur Translated-vs-Translated dono cases mein merge SKIP hota hai.

    extract() ab dedup se PEHLE wali raw list bhi return karta hai
    (result["items_before_merge"]), taake user chahe to original
    (un-merged) items bhi dekh/compare kar sake.
    """

    _client = None
    _embedder = None  # lazy-loaded, sirf tab load hota hai jab extract() chale

    # cross-case (ek printed + ek translated) ke liye threshold - yehi
    # ek case hai jahan merge allowed hai, isliye thoda strict rakha
    # gaya hai kyunke ek side ab bhi guess/generic translation hai
    CROSS_CASE_THRESHOLD = 0.81

    # is se kam confidence wala arabic_name "translated/guessed" maana
    # jata hai (EXTRACTION_PROMPT rule #11 ke mutabiq)
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

        if MenuDataExtractorSimple._client is None:
            MenuDataExtractorSimple._client = AsyncAnthropic(api_key=api_key)

        self.client = MenuDataExtractorSimple._client

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _media_type_for(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    @classmethod
    def _get_embedder(cls):
        """Local embedding model ek hi baar load hoti hai (class-level cache)."""
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Loading local embedding model '{cls.EMBED_MODEL_NAME}' "
                  f"(pehli baar thoda time lagega, download/model-load ho raha hai)...")
            cls._embedder = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls._embedder

    # -----------------------------------------------------------------
    # STEP: ek image Claude ko bhejna (true async I/O, no semaphore -
    # jitni images utni parallel requests, koi cap nahi)
    # -----------------------------------------------------------------

    async def _extract_one(self, image_path: str) -> list:
        media_type = self._media_type_for(image_path)

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
            item["_source_image"] = image_path
            # safety net: agar model ne phir bhi bhool se null/empty
            # chhoda ho (prompt ki wajah se normally aana nahi chahiye)
            if not item.get("arabic_name"):
                item.setdefault("confidence", {})["arabic_name"] = 0
            if not item.get("product_name"):
                item.setdefault("confidence", {})["product_name"] = 0

        return parsed.get("items", [])

    # -----------------------------------------------------------------
    # STEP: field-level merge (jaisa pehle tha - behtar confidence
    # wali value jeetti hai, khali fields bharti hain)
    # -----------------------------------------------------------------

    @staticmethod
    def _merge_two_items(primary: dict, duplicate: dict) -> dict:
        """primary ko duplicate ke behtar/missing fields se update karta hai."""
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

        primary_has_prices = any(
            v.get("price") is not None for v in primary.get("variants") or [])
        duplicate_has_prices = any(
            v.get("price") is not None for v in duplicate.get("variants") or [])
        if not primary_has_prices and duplicate_has_prices:
            primary["variants"] = duplicate["variants"]

        primary["has_photo_on_menu"] = bool(
            primary.get("has_photo_on_menu")) or bool(duplicate.get("has_photo_on_menu"))

        if "price" in d_conf and (d_conf.get("price") or 0) > (p_conf.get("price") or 0):
            p_conf["price"] = d_conf["price"]

        primary["confidence"] = p_conf

        primary_pages = primary.get("_source_pages") or (
            [primary["_source_image"]] if primary.get("_source_image") else [])
        duplicate_pages = duplicate.get("_source_pages") or (
            [duplicate["_source_image"]] if duplicate.get("_source_image") else [])
        primary["_source_pages"] = list(
            dict.fromkeys(primary_pages + duplicate_pages))
        primary.pop("_source_image", None)

        return primary

    # -----------------------------------------------------------------
    # STEP: LOCAL embedding-based dedup (arabic_name par, cosine sim)
    #   - koi Claude call NAHI, poora local/offline
    #   - har item ka arabic_name embed hota hai (ek batch call)
    #   - phir NxN cosine similarity matrix banta hai
    #   - GUARD RULE (v4): merge SIRF cross-case mein hota hai, yani
    #     jab ek item ka arabic_name PRINTED ho (confidence >=
    #     ARABIC_PRINTED_THRESHOLD) aur doosre ka TRANSLATED/guessed ho
    #     (confidence < threshold). Baaki dono cases -
    #     Printed-vs-Printed aur Translated-vs-Translated - hamesha
    #     SKIP hote hain, similarity kitni bhi high ho.
    # -----------------------------------------------------------------

    def _merge_duplicates_by_embedding(self, all_items: list) -> list:
        for item in all_items:
            if "_source_pages" not in item:
                item["_source_pages"] = [item.pop("_source_image", None)]

        if len(all_items) < 2:
            return all_items

        embedder = self._get_embedder()

        # e5-family models "query: " prefix ke sath train hue hain -
        # symmetric similarity (item vs item) ke liye dono taraf ye
        # prefix lagana behtar accuracy deta hai (e5 docs/recommendation)
        arabic_names = [
            f"query: {(item.get('arabic_name') or '').strip()}"
            for item in all_items
        ]
        embeddings = embedder.encode(
            arabic_names,
            normalize_embeddings=True,   # taake dot-product = cosine similarity ho
            show_progress_bar=False,
        )

        import numpy as np
        sim_matrix = embeddings @ embeddings.T  # NxN cosine similarity

        n = len(all_items)

        # confidence ke hisaab se "kaun zyada reliable hai" decide karne
        # ke liye ek chhota overall-confidence score
        def overall_conf(item):
            c = item.get("confidence") or {}
            vals = [c.get("product_name") or 0,
                    c.get("arabic_name") or 0, c.get("price") or 0]
            return sum(vals) / len(vals)

        def arabic_conf(item):
            return (item.get("confidence") or {}).get("arabic_name") or 0

        merged_away = set()

        # NxN pairs check - har item ko har doosre se compare (upper
        # triangle kaafi hai, symmetric matrix hai)
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

                # === GUARD RULE (v4) ===
                # Translated vs Translated -> SKIP (dono guess hain,
                # unreliable generic-wording match)
                if not i_is_printed and not j_is_printed:
                    continue

                # Printed vs Printed -> ab bhi SKIP (dono asal text hain,
                # lekin phir bhi auto-merge nahi karte - genuinely alag
                # do items ho sakti hain, false-merge ka risk)
                if i_is_printed and j_is_printed:
                    continue

                # Sirf ek case bacha: ek printed + ek translated
                # (asli cross-language duplicate case) -> ye hi merge
                # hota hai, strict threshold ke sath
                required_sim = self.CROSS_CASE_THRESHOLD

                if sim_matrix[i][j] >= required_sim:
                    item_i, item_j = all_items[i], all_items[j]
                    # zyada reliable wala primary banega
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
    # MAIN: batch - saari images ek sath (UNLIMITED parallel), phir
    # local embedding dedup pass
    # -----------------------------------------------------------------

    async def extract(self, image_paths: list, output_path: str | None = None,
                      merge_duplicates: bool = True) -> dict:
        """
        image_paths: sirf image files ke addresses ki list.
        output_path: (optional) result JSON file mein save karna ho to.
        merge_duplicates: True (default) -> local embedding-similarity
            pass jo arabic_name ke semantic match par duplicates merge
            karta hai (v4 guard rule ke sath: merge SIRF cross-case
            mein hota hai - ek printed + ek translated arabic_name.
            Printed-vs-Printed aur Translated-vs-Translated dono
            skip hote hain). Koi extra Claude call nahi lagti -
            bilkul local/offline.

        Return value (dict) mein DO lists hoti hain:
            "items"              -> final list, dedup/merge ke baad
            "items_before_merge" -> original raw list, dedup se
                                     PEHLE (jaisi seedha Claude se
                                     har image ke liye aayi thi,
                                     koi merge nahi hua)
        """
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"[ERROR] Ye image path(s) nahi milin: {missing}"
            )

        # jitni images utni parallel Claude calls - koi semaphore/cap nahi
        results = await asyncio.gather(
            *[self._extract_one(p) for p in image_paths]
        )
        all_items = [item for sub in results for item in sub]
        print(f"\nTotal items extracted (pre-merge): {len(all_items)}")

        # normalize _source_image -> _source_pages BEFORE snapshotting,
        # taake original list ka shape final list jaisa hi rahe
        # (sirf "kya merge hua" ka farak ho, field-naming ka nahi)
        for item in all_items:
            if "_source_pages" not in item:
                item["_source_pages"] = [item.pop("_source_image", None)]

        # yahan dedup/merge se PEHLE ki deep copy le lete hain - ye
        # wahi "original list" hai jo user ko chahiye, merge se
        # bilkul untouched
        items_before_merge = copy.deepcopy(all_items)

        if merge_duplicates:
            all_items = self._merge_duplicates_by_embedding(all_items)
            print(f"Total items after local-embedding dedup: {len(all_items)}")

        result = {
            "items": all_items,
            "items_before_merge": items_before_merge,
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
        image_paths = ["page_2.png", "page_4.png"]  # apni list yahan do

        extractor = MenuDataExtractorSimple()
        result = await extractor.extract(image_paths)

        print("\n--- FINAL (post-merge) ---")
        print(json.dumps(result["items"], indent=2, ensure_ascii=False))

        print("\n--- ORIGINAL (pre-merge) ---")
        print(json.dumps(result["items_before_merge"],
              indent=2, ensure_ascii=False))

    asyncio.run(main())
