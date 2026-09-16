"""
STEP 3 (CLASS-BASED + ASYNC): Missing-Photo Fallback
=========================================================================================
Jin items ki has_photo_on_menu false hai (ya jinhe Step 2 mein image_file
nahi mila), unke liye SerpAPI (Google Images) se photo dhoondta hai, CLIP
se verify karta hai, aur agar match na mile to OpenAI se image GENERATE
karta hai.

IMPORTANT: ye class Step 2 (MenuDataExtractor -> step2_matched.json) ke
POORA save hone ke baad hi chalni chahiye - isay input file chahiye hoti
hai jisme "has_photo_on_menu" field har item par set ho chuki ho.

ASYNC BEHAVIOUR - 2 tarah ke kaam yahan hain, dono alag treat hote hain:

  1. TRUE I/O-bound (genuinely parallel async):
       - SerpAPI search (network call)
       - Image download (network call)
       - OpenAI image generation (network call)
     Ye seedha async/await se parallel chalte hain - koi thread-offload
     ki zaroorat nahi, kyunki ye "wait karna" hai, "compute karna" nahi.

  2. CPU/GPU-bound (thread-offload zaroori):
       - CLIP scoring/inference
     Isay `loop.run_in_executor()` se background thread mein bhejte hain
     (bilkul DINO/OCR class jaisa), warna event-loop block ho jayega.
     Saath mein class-level SHARED semaphore (_clip_semaphore) hai taake
     multiple requests mile kar bhi GPU/CPU overload na karein.

.env mein zaroori:
  SERPAPI_KEY=your_serpapi_key
  OPENAI_API_KEY=sk-...

Setup:
  pip install aiohttp openai transformers torch pillow python-dotenv
"""

import asyncio
import base64
import io
import json
import os
import re
import threading

import aiohttp
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


class MissingPhotoFallback:
    """
    Un items ke liye jinki apni photo nahi hai, Google-image-search +
    CLIP-verification + AI-generation fallback flow.

    IMPORTANT: is class ko chalane se PEHLE Step 2 (MenuDataExtractor)
    poora complete ho chuka hona chahiye - iska input file
    (step2_matched.json ya similar) mein har item par "has_photo_on_menu"
    set hona zaroori hai.
    """

    # class-level shared model state - saare instances/requests ke beech
    # shared (CLIP ek hi baar load hota hai, chahe kitne bhi instances banein)
    _clip_model = None
    _clip_processor = None
    _clip_load_lock = threading.Lock()

    # class-level shared semaphores - poore system mein globally
    # concurrency control karte hain (jaisa pichli classes mein tha) -
    # taake multiple requests mile kar bhi GPU/API quota overload na karein
    _clip_semaphore = None
    _item_semaphore = None
    _semaphore_lock = threading.Lock()

    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.openai_api_key = os.getenv(
            "OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

        missing = [name for name, val in [
            ("SERPAPI_KEY", self.serpapi_key),
            ("OPENAI_API_KEY / OPEN_API_KEY", self.openai_api_key),
        ] if not val]
        if missing:
            raise RuntimeError(
                f"[ERROR] .env mein ye missing hain: {', '.join(missing)}\n"
                f"        Isi folder mein .env file banao/check karo."
            )

        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)

        # --- env-driven config ---
        self.clip_model_id = os.getenv(
            "CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        self.clip_match_threshold = _env_float("CLIP_MATCH_THRESHOLD", 0.27)
        self.search_num_results = _env_int("GOOGLE_SEARCH_NUM_RESULTS", 5)
        self.search_timeout = _env_int("GOOGLE_SEARCH_TIMEOUT", 10)
        self.download_timeout = _env_int("IMAGE_DOWNLOAD_TIMEOUT", 3)
        self.max_concurrent_items = _env_int("MAX_CONCURRENT_ITEMS", 6)
        self.max_concurrent_clip = _env_int("MAX_CONCURRENT_CLIP", 2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # class-level semaphores - ek hi baar banaye jaate hain, phir
        # saare instances/requests unhi ko share karte hain
        if MissingPhotoFallback._clip_semaphore is None:
            with MissingPhotoFallback._semaphore_lock:
                if MissingPhotoFallback._clip_semaphore is None:
                    MissingPhotoFallback._clip_semaphore = asyncio.Semaphore(
                        self.max_concurrent_clip)
                    MissingPhotoFallback._item_semaphore = asyncio.Semaphore(
                        self.max_concurrent_items)
        self._clip_semaphore = MissingPhotoFallback._clip_semaphore
        self._item_semaphore = MissingPhotoFallback._item_semaphore

    # -----------------------------------------------------------------
    # MODEL LOADING (lazy, class-level cached, thread-safe)
    # -----------------------------------------------------------------

    def _load_clip(self):
        if MissingPhotoFallback._clip_model is None:
            with MissingPhotoFallback._clip_load_lock:
                if MissingPhotoFallback._clip_model is None:
                    print(
                        f"[CLIP] Model load ho raha hai ({self.clip_model_id}) "
                        f"on device={self.device} ...")
                    MissingPhotoFallback._clip_processor = CLIPProcessor.from_pretrained(
                        self.clip_model_id)
                    MissingPhotoFallback._clip_model = CLIPModel.from_pretrained(
                        self.clip_model_id).to(self.device)
                    MissingPhotoFallback._clip_model.eval()
        return MissingPhotoFallback._clip_processor, MissingPhotoFallback._clip_model

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unnamed_item"

    # -----------------------------------------------------------------
    # STEP 1: SerpAPI search - TRUE I/O-bound (genuinely async)
    # -----------------------------------------------------------------

    async def _google_image_search(self, session: aiohttp.ClientSession, query: str) -> list:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.serpapi_key,
            "num": self.search_num_results,
            "safe": "active",
        }
        url = "https://serpapi.com/search.json"
        try:
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=self.search_timeout)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    print(
                        f"[SERPAPI][ERROR] status={resp.status} query='{query}'")
                    return []
        except Exception as e:
            print(f"[SERPAPI][ERROR] '{query}': {type(e).__name__}: {e!r}")
            return []

        items = data.get("images_results", [])
        urls = [item["original"]
                for item in items[:self.search_num_results] if "original" in item]
        print(f"[SERPAPI] '{query}' -> {len(urls)} candidate url(s)")
        return urls

    # -----------------------------------------------------------------
    # STEP 2: Parallel download - TRUE I/O-bound
    # -----------------------------------------------------------------

    async def _download_one(self, session: aiohttp.ClientSession, url: str):
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=self.download_timeout)
            ) as resp:
                if resp.status != 200:
                    return None
                raw = await resp.read()
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    async def _download_candidates(self, session: aiohttp.ClientSession, urls: list) -> list:
        images = await asyncio.gather(*[self._download_one(session, u) for u in urls])
        return [(u, img) for u, img in zip(urls, images) if img is not None]

    # -----------------------------------------------------------------
    # STEP 3: CLIP verification - CPU/GPU-bound (thread-offload zaroori)
    # -----------------------------------------------------------------

    def _clip_scores_sync(self, images: list, text: str) -> list:
        processor, model = self._load_clip()
        inputs = processor(text=[text], images=images,
                           return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        image_embeds = outputs.image_embeds / \
            outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / \
            outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (image_embeds @ text_embeds.T).squeeze(-1)
        return scores.cpu().tolist()

    async def _clip_best_match(self, candidates: list, query_text: str):
        if not candidates:
            return None, None, None

        urls, images = zip(*candidates)
        loop = asyncio.get_running_loop()

        async with self._clip_semaphore:
            scores = await loop.run_in_executor(
                None, self._clip_scores_sync, list(images), query_text)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return urls[best_idx], images[best_idx], scores[best_idx]

    # -----------------------------------------------------------------
    # STEP 4 (fallback): OpenAI image generation - TRUE I/O-bound
    # -----------------------------------------------------------------

    async def _generate_ai_image(self, prompt: str):
        try:
            resp = await self.openai_client.images.generate(
                model="gpt-image-1", prompt=prompt, size="1024x1024",
            )
            image_base64 = resp.data[0].b64_json
        except Exception as e:
            print(f"[OPENAI][ERROR] generation failed: {e}")
            return None

        try:
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"[OPENAI][ERROR] decode failed: {e}")
            return None

    # -----------------------------------------------------------------
    # PER-ITEM PIPELINE
    # -----------------------------------------------------------------

    async def _process_item(self, session: aiohttp.ClientSession, item: dict,
                            output_dir: str) -> dict:
        async with self._item_semaphore:
            name = (item.get("product_name") or item.get("name") or "").strip()
            description = (item.get("description") or "").strip()
            query_text = f"{name} {description}".strip()

            filename = self._sanitize_filename(name)
            out_path = os.path.join(output_dir, f"{filename}.jpg")

            urls = await self._google_image_search(session, query_text)
            candidates = await self._download_candidates(session, urls)
            best_url, best_image, best_score = await self._clip_best_match(
                candidates, query_text)

            result = {
                "source": name, "clip_score": best_score,
                "clip_source_url": None, "match_method": None, "image_file": None,
            }

            if best_score is not None and best_score >= self.clip_match_threshold:
                best_image.save(out_path)
                result.update({"match_method": "google",
                              "clip_source_url": best_url, "image_file": out_path})
                print(
                    f"[MATCH] \"{name}\" -> GOOGLE (score={best_score:.3f}) -> {out_path}")
            else:
                gen_prompt = f"A professional, appetizing food photo of {name}. {description}".strip(
                )
                gen_image = await self._generate_ai_image(gen_prompt)
                if gen_image is not None:
                    gen_image.save(out_path)
                    result.update(
                        {"match_method": "generated", "image_file": out_path})
                    print(f"[MATCH] \"{name}\" -> GENERATED -> {out_path}")
                else:
                    result["match_method"] = "failed"
                    print(f"[MATCH] \"{name}\" -> FAILED")

            return result

    # -----------------------------------------------------------------
    # BATCH: saare missing-photo items PARALLEL process karna
    # -----------------------------------------------------------------

    async def process_items(self, items: list, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self._process_item(session, item, output_dir) for item in items]
            )
        return list(results)

    # -----------------------------------------------------------------
    # MAIN: JSON file load -> fallback chalao -> merge -> save
    # -----------------------------------------------------------------

    async def fill_missing_photos_in_json(self, input_json_path: str,
                                          output_json_path: str,
                                          output_dir: str,
                                          limit: int | None = 3) -> str:
        """
        IMPORTANT: input_json_path (Step 2 ka output) poora complete
        aur exist karna chahiye - warna clear error dega.

        limit: ==== SIRF DEVELOPMENT/TESTING KE LIYE ====
            Agar diya jaye (jaise limit=3), to sirf pehle N missing-photo
            items hi process honge (baaki as-is chhod diye jaate hain,
            unki has_photo_on_menu false hi rahegi). Isse SerpAPI/OpenAI
            cost testing ke waqt control mein rehti hai.
            PRODUCTION mein ise None hi rakhna (ya call karte waqt is
            parameter ko bilkul mat do) - taake SAARE missing items
            process hon.
        """
        if not os.path.exists(input_json_path):
            raise RuntimeError(
                f"[ERROR] '{input_json_path}' nahi mili.\n"
                "        Pehle Step 2 (MenuDataExtractor) poora chala "
                "kar is file ko generate karo."
            )

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_items = data["items"] if isinstance(
            data, dict) and "items" in data else data

        missing_items = [it for it in all_items if not it.get(
            "has_photo_on_menu", False)]

        # ==== DEV-ONLY LIMIT - isay hatana ho to poora ye if-block hata do ====
        if limit is not None:
            print(
                f"[DEV MODE] limit={limit} laga hai - sirf pehle {limit} "
                f"missing item(s) process honge (total missing: "
                f"{len(missing_items)}). Production ke liye limit=None "
                f"karo ya parameter hi mat do.")
            missing_items = missing_items[:limit]
        # ==== DEV-ONLY LIMIT END ====

        print(
            f"[INFO] {len(missing_items)} / {len(all_items)} items ki "
            f"photo missing hai - process kar rahe hain")

        if missing_items:
            results = await self.process_items(missing_items, output_dir)
            results_by_name = {r["source"]: r for r in results}
            for item in all_items:
                name = item.get("product_name") or item.get("name")
                if not item.get("has_photo_on_menu", False) and name in results_by_name:
                    r = results_by_name[name]
                    item["has_photo_on_menu"] = r["match_method"] in (
                        "google", "generated")
                    item["image_file"] = r["image_file"]
                    item["image_match_method"] = r["match_method"]
                    item["image_clip_score"] = r["clip_score"]

        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        out_payload = {"items": all_items} if isinstance(
            data, dict) and "items" in data else all_items
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

        print(f"\nSaved -> {output_json_path}")
        return output_json_path
