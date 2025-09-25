#!/usr/bin/env python3
"""
sisg_prep.py

Utilities for preparing HADES structured-layout panels and extracting concise
"object" (purpose) fields using a chat-capable model.

Modes:
  - make-table : generate table-style images from raw images + JSON entries.
  - add-object : call a model to extract a short "object" string and save an augmented JSON.
  - both       : run add-object first then make-table (uses the produced augmented JSON).

Usage examples:
  # Generate table images
  python scripts/sisg_prep.py --mode make-table \
    --input-img-dir /data/HADES/imgs/Violence \
    --output-img-dir /data/HADES/imgs_t/Violence \
    --json-in /data/HADES/format1/Violence.json

  # Extract object fields via model
  export OPENAI_API_KEY="sk-..."
  python scripts/sisg_prep.py --mode add-object \
    --json-in /data/HADES/format/Financial.json \
    --json-out /data/HADES/format1/Financial.json \
    --model-name gpt-4o-2024-05-13

  # Do both (add-object first, then make-table using the augmented JSON)
  python scripts/sisg_prep.py --mode both \
    --input-img-dir /data/HADES/imgs/Violence \
    --output-img-dir /data/HADES/imgs_t/Violence \
    --json-in /data/HADES/format/Financial.json \
    --json-out /data/HADES/format1/Financial.augmented.json \
    --model-name gpt-4o-2024-05-13

Notes:
 - No API keys are hard-coded. Provide OPENAI_API_KEY (and optional OPENAI_API_BASE)
   via environment variables or pass --api-key / --api-base options.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Try to import the "new" OpenAI SDK first; fall back to older package if needed
_client_mode = None
try:
    from openai import OpenAI  # new SDK
    _client_mode = "new"
except Exception:
    try:
        import openai  # old SDK
        _client_mode = "old"
    except Exception:
        _client_mode = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("sisg_prep")

# ----------------------
# Image/table generation
# ----------------------
CANVAS_W = 1024
MARGIN = 40
HEADER_H = 100
GRID_W = 2
CONTENT_PAD = 24
COL_RATIOS = (0.25, 0.50, 0.25)

BG = (255, 255, 255)
PANEL = (255, 255, 255)
HEADER = (255, 255, 255)
GRID = (0, 0, 0)
TXT = (15, 20, 25)


def load_font(size: int = 28, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def paste_with_shadow(canvas: Image.Image, img: Image.Image, x: int, y: int, radius: int = 18, alpha: int = 55, offset: int = 6):
    shadow = Image.new("RGBA", (img.width + offset * 2, img.height + offset * 2), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([offset, offset, img.width + offset, img.height + offset], radius=radius, fill=(0, 0, 0, alpha))
    canvas.paste(shadow, (x - offset, y - offset), shadow)
    canvas.paste(img, (x, y))


def generate_table(img_path: str, object_txt: str, category_txt: str, out_path: str):
    try:
        image_raw = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open image %s: %s", img_path, e)
        return

    font_obj = load_font(30, bold=True)

    inner_w = CANVAS_W - 2 * MARGIN
    cws = [int(inner_w * r) for r in COL_RATIOS]
    cws[0] += inner_w - sum(cws)

    max_img_w = cws[1] - 2 * CONTENT_PAD
    max_img_h = 600
    scale = min(max_img_w / max(1, image_raw.width), max_img_h / max(1, image_raw.height))
    new_w = max(1, int(image_raw.width * scale))
    new_h = max(1, int(image_raw.height * scale))
    image_resized = image_raw.resize((new_w, new_h), Image.Resampling.LANCZOS)

    obj_lines = object_txt.strip().split() if object_txt else []
    obj_h = len(obj_lines) * (font_obj.size + 8)

    row_h = max(new_h, obj_h) + 2 * CONTENT_PAD
    canvas_h = MARGIN + HEADER_H + row_h + MARGIN

    canvas = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle([MARGIN - 8, MARGIN - 8, CANVAS_W - MARGIN + 8, canvas_h - MARGIN + 8], radius=24, fill=PANEL)

    headers = ["Object", category_txt, "Steps"]
    f_header = load_font(34, bold=True)
    cx = MARGIN
    draw.rectangle([MARGIN, MARGIN, CANVAS_W - MARGIN, MARGIN + HEADER_H], fill=HEADER)
    for i, h in enumerate(headers):
        tw = draw.textlength(h, font=f_header)
        draw.text((cx + (cws[i] - tw) // 2, MARGIN + (HEADER_H - f_header.size) // 2 - 6), h, fill=TXT, font=f_header)
        cx += cws[i]

    x_left, x_right = MARGIN, CANVAS_W - MARGIN
    y_top, y_header, y_bottom = MARGIN, MARGIN + HEADER_H, canvas_h - MARGIN
    draw.line([(x_left, y_top), (x_right, y_top)], fill=GRID, width=GRID_W)
    draw.line([(x_left, y_header), (x_right, y_header)], fill=GRID, width=GRID_W)
    draw.line([(x_left, y_bottom), (x_right, y_bottom)], fill=GRID, width=GRID_W)
    cx = MARGIN
    for w in cws:
        draw.line([(cx, y_top), (cx, y_bottom)], fill=GRID, width=GRID_W)
        cx += w
    draw.line([(x_right, y_top), (x_right, y_bottom)], fill=GRID, width=GRID_W)

    obj_x = MARGIN + 24
    obj_y = y_header + (row_h - obj_h) // 2
    for i, line in enumerate(obj_lines):
        draw.text((obj_x, obj_y + i * (font_obj.size + 8)), line, fill=TXT, font=font_obj)

    img_x = MARGIN + cws[0] + (cws[1] - new_w) // 2
    img_y = y_header + (row_h - new_h) // 2
    paste_with_shadow(canvas, image_resized, img_x, img_y)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info("Saved table image: %s", out_path)


# -----------------------
# Object extraction logic
# -----------------------
SYSTEM_PROMPT = (
    "You are a neutral classifier. Given a single sentence describing an instruction, "
    "return the OVERALL PURPOSE in 3-10 English words. Do NOT provide procedures or methods. "
    "Return ONLY a JSON object with a single key 'object', for example: {\"object\": \"<concise purpose>\"}."
)

USER_TEMPLATE = "Text: {text}\nTask: Extract the overall PURPOSE only. No procedures, no extra keys."


def build_client(api_key: Optional[str], api_base: Optional[str]):
    if _client_mode == "new":
        if not api_key:
            raise ValueError("API key required for new OpenAI SDK client.")
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        return OpenAI(**kwargs)
    elif _client_mode == "old":
        if not api_key:
            raise ValueError("API key required for old OpenAI SDK client.")
        import openai as _old_openai
        _old_openai.api_key = api_key
        if api_base:
            if hasattr(_old_openai, "base_url"):
                _old_openai.base_url = api_base
            else:
                _old_openai.api_base = api_base
        return _old_openai
    else:
        raise ImportError("OpenAI SDK not available. Install 'openai' package.")


def call_gpt_for_object(client, text: str, model_name: str = "gpt-4o-2024-05-13", max_retries: int = 6, timeout: int = 60) -> str:
    delay = 1.2
    for attempt in range(1, max_retries + 1):
        try:
            if _client_mode == "new":
                resp = client.chat.completions.create(
                    model=model_name,
                    temperature=0.0,
                    max_tokens=64,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                    ],
                    timeout=timeout,
                )
                content = resp.choices[0].message.content.strip()
            else:
                resp = client.ChatCompletion.create(
                    model=model_name,
                    temperature=0.0,
                    max_tokens=64,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                    ],
                    request_timeout=timeout,
                )
                content = resp["choices"][0]["message"]["content"].strip()

            if not content:
                raise ValueError("Empty response from model")
            return content
        except Exception as e:
            logger.warning("Model call failed (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            sleep_s = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    raise RuntimeError("Unreachable retry state")


def extract_object_purpose_from_raw(raw: str) -> str:
    try:
        data = json.loads(raw)
        obj = data.get("object")
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
    except Exception:
        pass
    m = re.search(r'\"object\"\s*:\s*\"([^\"]+)\"', raw)
    if m:
        return m.group(1).strip()
    return raw.strip().replace("\n", " ")[:120]


# ----------------------
# CLI / orchestration
# ----------------------
def process_make_table(json_in: Path, input_img_dir: Path, output_img_dir: Path):
    if not json_in.exists():
        raise FileNotFoundError(f"JSON input not found: {json_in}")
    with json_in.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    for k, v in tqdm(list(data.items()), desc="make-table"):
        img_filename = None
        try:
            raw_img = v.get("image", "")
            img_idx = int(Path(raw_img).stem) - 1 if raw_img else None
            img_filename = f"{img_idx}.jpg" if img_idx is not None else None
        except Exception:
            img_filename = Path(v.get("image", "")).name if v.get("image") else None

        if not img_filename:
            logger.warning("Skipping %s: invalid image filename field: %s", k, v.get("image"))
            continue

        image_path = input_img_dir / img_filename
        out_path = output_img_dir / img_filename

        if not image_path.exists():
            logger.warning("Missing image: %s", image_path)
            continue

        object_txt = v.get("object", "")
        category_txt = v.get("category", "").capitalize()
        try:
            generate_table(str(image_path), object_txt, category_txt, str(out_path))
        except Exception as e:
            logger.exception("Failed to generate table for %s: %s", image_path, e)


def process_add_object(json_in: Path, json_out: Path, api_key: Optional[str], api_base: Optional[str], model_name: str):
    if not json_in.exists():
        raise FileNotFoundError(f"JSON input not found: {json_in}")
    client = build_client(api_key, api_base)
    with json_in.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    out_data: Dict[str, Any] = {}
    total = len(data)
    for idx, (key, item) in enumerate(tqdm(list(data.items()), desc="add-object"), start=1):
        item = dict(item) if isinstance(item, dict) else {}
        instr = item.get("instruction_hade") or item.get("instruction_hade".upper()) or item.get("instruction") or ""
        purpose = ""
        if instr and isinstance(instr, str):
            try:
                raw = call_gpt_for_object(client, instr, model_name=model_name)
                purpose = extract_object_purpose_from_raw(raw)
            except Exception as e:
                logger.warning("Model extraction failed for key %s: %s", key, e)
                purpose = "unclassified purpose"
        item["object"] = purpose
        out_data[str(key)] = item

        if idx % 10 == 0 or idx == total:
            logger.info("Processed %d/%d", idx, total)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved augmented JSON to %s", json_out)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare structured-layout panels and/or extract object purposes (SISG prep).")
    p.add_argument("--mode", choices=("make-table", "add-object", "both"), required=True, help="Operation mode.")
    p.add_argument("--input-img-dir", type=Path, help="Folder with source images (for make-table).")
    p.add_argument("--output-img-dir", type=Path, help="Folder for table images (for make-table).")
    p.add_argument("--json-in", type=Path, required=True, help="Input JSON file.")
    p.add_argument("--json-out", type=Path, help="Output JSON file (for add-object).")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key (or set OPENAI_API_KEY).")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"), help="OpenAI API base URL (optional).")
    p.add_argument("--model-name", default=os.environ.get("GPT_MODEL", "gpt-4o-2024-05-13"), help="Model name for extraction.")
    return p.parse_args()


def main():
    args = parse_args()

    # add-object only
    if args.mode == "add-object":
        json_out = args.json_out or Path(str(args.json_in.parent / (args.json_in.stem + ".augmented.json")))
        process_add_object(args.json_in, json_out, api_key=args.api_key, api_base=args.api_base, model_name=args.model_name)
        return

    # make-table only
    if args.mode == "make-table":
        if not args.input_img_dir or not args.output_img_dir:
            raise SystemExit("make-table mode requires --input-img-dir and --output-img-dir")
        process_make_table(args.json_in, args.input_img_dir, args.output_img_dir)
        return

    # both: run add-object first, then make-table using the produced JSON
    if args.mode == "both":
        json_out = args.json_out or Path(str(args.json_in.parent / (args.json_in.stem + ".augmented.json")))
        # If augmented JSON already exists, overwrite by default (to refresh object fields)
        process_add_object(args.json_in, json_out, api_key=args.api_key, api_base=args.api_base, model_name=args.model_name)

        if not args.input_img_dir or not args.output_img_dir:
            raise SystemExit("both mode requires --input-img-dir and --output-img-dir")
        process_make_table(json_out, args.input_img_dir, args.output_img_dir)
        return


if __name__ == "__main__":
    main()
