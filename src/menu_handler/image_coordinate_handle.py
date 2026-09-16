"""
NO-LLM, NO-COST, PIP-ONLY PIPELINE (CLASS-BASED + ASYNC):
Grounding DINO (LOCAL) + EasyOCR (LOCAL) + Python position-matching
=========================================================================================
... (docstring wesa hi, sirf ye add: ab results JSON file ki jagah
     DIRECTLY `PageCoordinates` table mein DB save hote hain, cropped
     images `upload/images/itemImages/` mein jaati hain, aur DINO ko
     di jaane wali image ko CPU-speed ke liye resize kiya ja sakta hai
     - `DINO_MAX_DIM` env-var se control hota hai; `0` set karne se
     resize OFF ho jata hai [GPU/production ke liye])
"""

import asyncio
import os
import re
import threading

import torch
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import easyocr

from src.db.models import PageCoordinates

load_dotenv()


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _env_list(key: str, default: list) -> list:
    raw = os.getenv(key)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


ITEM_IMAGES_DIR = os.getenv(
    "ITEM_IMAGES_DIR", os.path.join("uploads", "images", "itemImages"))


class MenuImageExtractor:
    """
    Ek menu-page image se dish-photos crop + name-match karne ka poora
    flow. Results JSON file ki jagah seedha `PageCoordinates` DB table
    mein save hote hain.
    """

    _dino_processor = None
    _dino_model = None
    _ocr_reader = None
    _load_lock = threading.Lock()

    def __init__(self, output_root: str = "output_no_llm"):
        self.output_root = output_root

        self.model_id = os.getenv(
            "GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
        self.dino_query = os.getenv(
            "DINO_QUERY", "food photo. dish photo.")
        self.box_threshold = _env_float("BOX_THRESHOLD", 0.35)
        self.text_threshold = _env_float("TEXT_THRESHOLD", 0.25)

        self.ocr_languages = _env_list("OCR_LANGUAGES", ["en"])
        self.ocr_min_confidence = _env_float("OCR_MIN_CONFIDENCE", 0.35)
        self.ocr_canvas_size = _env_int("OCR_CANVAS_SIZE", 1280)

        self.min_photo_size = _env_int("MIN_PHOTO_SIZE", 40)
        self.max_match_distance = _env_int("MAX_MATCH_DISTANCE", 250)
        self.food_label_keywords = _env_list(
            "FOOD_LABEL_KEYWORDS", ["food", "dish"])

        # DINO ko dene se pehle image ko is max-dimension tak resize
        # karte hain (CPU pe fast karne ke liye). `0` set karne se
        # resize OFF ho jata hai - production/GPU pe yehi karna hai,
        # taake full-resolution + best accuracy mile.
        self.dino_max_dim = _env_int("DINO_MAX_DIM", 900)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------------------

    def _load_dino(self):
        if MenuImageExtractor._dino_model is None:
            with MenuImageExtractor._load_lock:
                if MenuImageExtractor._dino_model is None:
                    print(
                        f"[DINO] Model load ho raha hai ({self.model_id}) "
                        f"on device={self.device} ...")
                    MenuImageExtractor._dino_processor = AutoProcessor.from_pretrained(
                        self.model_id)
                    MenuImageExtractor._dino_model = (
                        AutoModelForZeroShotObjectDetection
                        .from_pretrained(self.model_id)
                        .to(self.device)
                    )
                    MenuImageExtractor._dino_model.eval()
        return MenuImageExtractor._dino_processor, MenuImageExtractor._dino_model

    def _load_ocr(self):
        if MenuImageExtractor._ocr_reader is None:
            with MenuImageExtractor._load_lock:
                if MenuImageExtractor._ocr_reader is None:
                    print(
                        f"[OCR] EasyOCR reader load ho raha hai "
                        f"(langs={self.ocr_languages}) ...")
                    MenuImageExtractor._ocr_reader = easyocr.Reader(
                        self.ocr_languages, gpu=(self.device == "cuda"), quantize=False)
        return MenuImageExtractor._ocr_reader

    # -----------------------------------------------------------------
    # STEP 1: Grounding DINO (ab optional resize ke sath - CPU-speed)
    # -----------------------------------------------------------------

    def _detect_photo_boxes_sync(self, image_path: str) -> list:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[ERROR] Image nahi mili: {image_path}")

        processor, model = self._load_dino()
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # --- resize (agar dino_max_dim > 0 hai) ---
        if self.dino_max_dim > 0:
            scale = min(1.0, self.dino_max_dim / max(orig_w, orig_h))
        else:
            scale = 1.0

        dino_input_image = (
            image.resize((int(orig_w * scale), int(orig_h * scale)))
            if scale < 1.0 else image
        )

        inputs = processor(images=dino_input_image, text=self.dino_query,
                           return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[dino_input_image.size[::-1]],
        )[0]

        raw_labels = results.get("text_labels", results.get("labels", []))

        boxes = []
        skipped = 0
        for box, score, label in zip(
            results["boxes"].tolist(), results["scores"].tolist(), raw_labels
        ):
            # resized-image ke coordinates -> wapas ORIGINAL image scale
            x0, y0, x1, y1 = [v / scale for v in box]

            if (x1 - x0) < self.min_photo_size or (y1 - y0) < self.min_photo_size:
                continue

            label_text = str(label).lower()
            is_food_label = any(
                kw in label_text for kw in self.food_label_keywords)
            if not is_food_label:
                skipped += 1
                continue

            boxes.append({"bbox": [x0, y0, x1, y1],
                         "confidence": score, "label": label})

        print(
            f"[DINO] {len(boxes)} photo box(es) kept in {image_path} "
            f"({skipped} non-food label(s) skipped, resize_scale={scale:.2f})")
        return boxes

    async def detect_photo_boxes(self, image_path: str) -> list:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_photo_boxes_sync, image_path)

    # -----------------------------------------------------------------
    # STEP 2: EasyOCR
    # -----------------------------------------------------------------

    def _extract_text_blocks_sync(self, image_path: str) -> list:
        reader = self._load_ocr()
        try:
            raw_results = reader.readtext(
                image_path, canvas_size=self.ocr_canvas_size)
        except RuntimeError as e:
            if "not enough memory" in str(e).lower() or "alloc" in str(e).lower():
                raise RuntimeError(
                    "[ERROR] EasyOCR ke liye RAM kam pad gayi.\n"
                    f"        OCR_CANVAS_SIZE={self.ocr_canvas_size} hai - "
                    ".env mein isay kam karo (e.g. 960)."
                ) from e
            raise

        text_blocks = []
        for idx, (quad_points, text, confidence) in enumerate(raw_results, start=1):
            text = text.strip()
            if not text or confidence < self.ocr_min_confidence:
                continue
            xs = [p[0] for p in quad_points]
            ys = [p[1] for p in quad_points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            text_blocks.append({
                "id": f"t{idx}", "text": text, "bbox": bbox, "confidence": confidence,
            })

        print(f"[OCR] {len(text_blocks)} text line(s) found in {image_path}")
        return text_blocks

    async def extract_text_blocks(self, image_path: str) -> list:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_text_blocks_sync, image_path)

    # -----------------------------------------------------------------
    # STEP 3: Position matching
    # -----------------------------------------------------------------

    @staticmethod
    def _y_overlap(bbox1: list, bbox2: list) -> float:
        y0 = max(bbox1[1], bbox2[1])
        y1 = min(bbox1[3], bbox2[3])
        return max(0.0, y1 - y0)

    @staticmethod
    def _horizontal_gap(bbox1: list, bbox2: list) -> float:
        if bbox1[2] < bbox2[0]:
            return bbox2[0] - bbox1[2]
        if bbox2[2] < bbox1[0]:
            return bbox1[0] - bbox2[2]
        return 0.0

    def _find_matching_text(self, photo_bbox: list, text_blocks: list):
        if not text_blocks:
            return None
        same_line = []
        for block in text_blocks:
            overlap = self._y_overlap(photo_bbox, block["bbox"])
            if overlap > 0:
                gap = self._horizontal_gap(photo_bbox, block["bbox"])
                same_line.append((gap, block))
        if same_line:
            same_line.sort(key=lambda t: t[0])
            best_gap, best_block = same_line[0]
            return best_block if best_gap <= self.max_match_distance else None

        photo_cy = (photo_bbox[1] + photo_bbox[3]) / 2
        best_block, best_dist = None, float("inf")
        for block in text_blocks:
            block_cy = (block["bbox"][1] + block["bbox"][3]) / 2
            dist = abs(photo_cy - block_cy)
            if dist < best_dist:
                best_dist = dist
                best_block = block
        return best_block if best_dist <= self.max_match_distance else None

    def _find_nearby_texts(self, photo_bbox: list, text_blocks: list, top_n: int = 6) -> list:
        if not text_blocks:
            return []
        photo_cy = (photo_bbox[1] + photo_bbox[3]) / 2
        scored = []
        for block in text_blocks:
            overlap = self._y_overlap(photo_bbox, block["bbox"])
            if overlap > 0:
                score = self._horizontal_gap(photo_bbox, block["bbox"])
            else:
                block_cy = (block["bbox"][1] + block["bbox"][3]) / 2
                score = self.max_match_distance + abs(photo_cy - block_cy)
            scored.append((score, block))
        scored.sort(key=lambda t: t[0])

        nearby = []
        for score, block in scored:
            if score > self.max_match_distance * 2:
                continue
            nearby.append(block)
            if len(nearby) >= top_n:
                break
        return nearby

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unnamed_item"

    # -----------------------------------------------------------------
    # STEP 4: Crop + save (hamesha ORIGINAL full-res image se - resize
    # ka koi asar crop-quality pe nahi padta)
    # -----------------------------------------------------------------

    @staticmethod
    def _crop_and_save(image_path: str, bbox: list, out_dir: str, out_name: str):
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(image_path)
        x0, y0, x1, y1 = bbox
        x0, x1 = sorted([max(0, x0), min(img.width, x1)])
        y0, y1 = sorted([max(0, y0), min(img.height, y1)])
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        cropped = img.crop((x0, y0, x1, y1))
        out_path = os.path.join(out_dir, out_name)
        cropped.save(out_path)
        return out_path

    # -----------------------------------------------------------------
    # DB SAVE
    # -----------------------------------------------------------------

    @staticmethod
    def _truncate(value, max_len: int):
        if value is None:
            return None
        value = str(value)
        return value if len(value) <= max_len else value[:max_len]

    async def _save_to_db(self, db: AsyncSession, items: list) -> list:
        rows = [
            PageCoordinates(
                menu_image_id=item["menu_image_id"],
                matched_text=self._truncate(item.get("matched_text"), 255),
                raw_text=self._truncate(item.get("raw_text"), 2000),
                detected_label=self._truncate(item.get("detected_label"), 255),
                source_image=self._truncate(item.get("source_image"), 500),
                source_image_path=self._truncate(
                    item.get("source_image_path"), 1000),
                photo_bbox=item.get("photo_bbox"),
                confidence=item.get("confidence"),
                image_file=self._truncate(item.get("image_file"), 1000),
            )
            for item in items
        ]

        if rows:
            db.add_all(rows)
            await db.commit()
            for row in rows:
                await db.refresh(row)

        return rows

    @staticmethod
    def _row_to_dict(row: PageCoordinates) -> dict:
        return {
            "id": row.id,
            "menu_image_id": row.menu_image_id,
            "matched_text": row.matched_text,
            "raw_text": row.raw_text,
            "detected_label": row.detected_label,
            "source_image": row.source_image,
            "source_image_path": row.source_image_path,
            "photo_bbox": row.photo_bbox,
            "confidence": row.confidence,
            "image_file": row.image_file,
        }

    # -----------------------------------------------------------------
    # MAIN: ek image ke liye poora flow
    # -----------------------------------------------------------------

    async def process_image(self, image_path: str, file_name: str, menu_image_id: int) -> list:
        source_image_name = os.path.basename(image_path)

        photo_boxes, text_blocks = await asyncio.gather(
            self.detect_photo_boxes(image_path),
            self.extract_text_blocks(image_path),
        )

        used_names = set()
        results = []

        for idx, photo in enumerate(photo_boxes, start=1):
            matched_block = self._find_matching_text(
                photo["bbox"], text_blocks)
            matched_text = matched_block["text"] if matched_block else None

            nearby_blocks = self._find_nearby_texts(
                photo["bbox"], text_blocks)
            raw_text = " | ".join(
                b["text"] for b in nearby_blocks) if nearby_blocks else None

            base_name = self._sanitize_filename(
                matched_text) if matched_text else f"unmatched_{idx}"
            final_name = base_name
            suffix = 2
            while final_name in used_names:
                final_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(final_name)

            out_path = self._crop_and_save(
                image_path, photo["bbox"], ITEM_IMAGES_DIR,
                f"{file_name}_{final_name}.jpg")

            results.append({
                "menu_image_id": menu_image_id,
                "matched_text": matched_text,
                "raw_text": raw_text,
                "detected_label": photo.get("label"),
                "source_image": source_image_name,
                "source_image_path": image_path,
                "photo_bbox": photo["bbox"],
                "confidence": photo.get("confidence"),
                "image_file": out_path,
            })
            print(
                f"[MATCH] photo#{idx} -> \"{matched_text}\" -> {out_path} "
                f"(source: {source_image_name})")

        return results

    async def process_images(self, pages: list, db: AsyncSession) -> dict:
        """
        pages: [{"path": ..., "filename": ..., "menu_image_id": ...}, ...]

        Jo bhi `pages` list milti hai, sab EK SATH (parallel) process
        hoti hain - batching (agar chahiye) caller (endpoint) ki
        zimmedari hai, is method ki nahi.
        """
        os.makedirs(ITEM_IMAGES_DIR, exist_ok=True)

        all_results_nested = await asyncio.gather(
            *[
                self.process_image(
                    page["path"],
                    file_name=page["filename"],
                    menu_image_id=page["menu_image_id"],
                )
                for page in pages
            ]
        )
        all_results = [item for sub in all_results_nested for item in sub]

        saved_rows = await self._save_to_db(db, all_results)
        saved_items = [self._row_to_dict(r) for r in saved_rows]

        print(f"\nTotal photos matched & saved to DB: {len(saved_items)}")
        return {"items": saved_items}
