#!/usr/bin/env python3
"""
generate.py — GitHub-only, always-different analogue-horror PPT/VHS video generator.

Key features:
- Robust config loading with ${ENV_VAR} resolution + safe seed parsing (fixes '${SEED}' ValueError)
- Robust text truncation (fixes empty range for randint when strings are short)
- "Brain" selects a theme key (topic) to drive scraping + story
- Scrapes text from Wikipedia + images from Wikimedia Commons (plus uses local repo images)
- Mixes normal wellness slides with horror protocol / Jane Doe / fatal error intermissions
- Adds up to N pop-up jump images (max 3) lasting ~0.5s
- TTS with espeak (multiple voice profiles), plus VHS noise bed + abrupt stabs
- Renders MP4 using ffmpeg muxing

Run:
  python generate.py --config config.yaml --out out.mp4

In GitHub Actions you can pass SEED env var or leave it auto.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write


# ----------------------------
# Utilities: config + env
# ----------------------------

ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def check_system_deps() -> None:
    """Ensure required external binaries are present."""
    required = ["ffmpeg", "espeak"]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        print(f"ERROR: Missing system dependencies: {', '.join(missing)}")
        print("Please install them (e.g., 'apt-get install ffmpeg espeak' or 'brew install ffmpeg espeak').")
        sys.exit(1)


def resolve_env_value(v: Any) -> Any:
    """Resolve strings like '${SEED}' from environment. Otherwise return as-is."""
    if isinstance(v, str):
        m = ENV_VAR_PATTERN.match(v.strip())
        if m:
            name = m.group(1)
            return os.environ.get(name, "")
    return v


def deep_resolve_env(obj: Any) -> Any:
    """Recursively resolve env values inside dict/list structures."""
    if isinstance(obj, dict):
        return {k: deep_resolve_env(resolve_env_value(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_resolve_env(resolve_env_value(x)) for x in obj]
    return resolve_env_value(obj)


def parse_int_safe(x: Any, default: int) -> int:
    """Parse integer from many inputs (string, env-resolved), with fallback."""
    if x is None:
        return default
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("auto", "random", "none", "null"):
            return default
        try:
            return int(s)
        except ValueError:
            digits = re.findall(r"-?\d+", s)
            if digits:
                try:
                    return int(digits[0])
                except Exception:
                    return default
    return default


def now_seed() -> int:
    return int(time.time() * 1000) % 2_147_483_647


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    # --- FIX START: Normalize 'web' config if it's a boolean ---
    # If config has "web: true" or "web: false", convert it to a dict
    # so calls like cfg.get("web").get("key") don't crash.
    web_val = cfg.get("web")
    if web_val is not None and not isinstance(web_val, dict):
        cfg["web"] = {"enable": bool(web_val)}
    # --- FIX END ---

    cfg = deep_resolve_env(cfg)
    return cfg


# ----------------------------
# Scraping (Wikipedia + Wikimedia)
# ----------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def http_get_json(url: str, params: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "pptex-github-action/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"WARN: API Request failed: {e}")
        return {}


def wiki_random_titles(rng: random.Random, n: int) -> List[str]:
    data = http_get_json(WIKI_API, {
        "action": "query", "format": "json", "list": "random",
        "rnnamespace": 0, "rnlimit": n,
    })
    items = data.get("query", {}).get("random", []) or []
    titles = [it.get("title", "").strip() for it in items if it.get("title")]
    rng.shuffle(titles)
    return titles


def wiki_extract(title: str, max_chars: int = 1200) -> str:
    data = http_get_json(WIKI_API, {
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": 1, "explaintext": 1, "redirects": 1, "titles": title,
    })
    pages = data.get("query", {}).get("pages", {}) or {}
    for _, p in pages.items():
        txt = (p.get("extract") or "").strip()
        if txt:
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt[:max_chars]
    return ""


def commons_image_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    data = http_get_json(COMMONS_API, {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit,
        "gsrnamespace": 6, "prop": "imageinfo", "iiprop": "url|mime",
    })
    pages = data.get("query", {}).get("pages", {}) or {}
    out: List[Dict[str, Any]] = []
    for _, p in pages.items():
        title = p.get("title", "")
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("url")
        mime = ii.get("mime")
        if url and mime and mime.startswith("image/"):
            out.append({"title": title, "url": url, "mime": mime})
    return out


def download_image(url: str, out_path: Path, timeout: int = 25) -> bool:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "pptex-github-action/1.0"})
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


# ----------------------------
# Text “brain” + encryption
# ----------------------------

def choose_theme_key(rng: random.Random, spec: Dict[str, Any]) -> str:
    keys = spec.get("brain", {}).get("theme_keys") or []
    keys = [k for k in keys if isinstance(k, str) and k.strip()]
    if keys:
        return rng.choice(keys).strip()
    titles = wiki_random_titles(rng, 6)
    return titles[0] if titles else "Memory"


def rot13(s: str) -> str:
    return s.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
    ))


def caesar(s: str, shift: int = 7) -> str:
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def encrypt_text(s: str, mode: str) -> str:
    mode = (mode or "none").lower()
    if mode == "none": return s
    if mode == "rot13": return rot13(s)
    if mode == "base64": return base64.b64encode(s.encode("utf-8")).decode("ascii")
    if mode == "caesar": return caesar(s, shift=7)
    return s


def safe_truncate(rng: random.Random, s: str, lo: int = 28, hi: int = 70) -> str:
    s = (s or "").strip()
    if not s: return ""
    n = len(s)
    upper = min(hi, n)
    if upper <= lo:
        return s[:upper]
    end = rng.randint(lo, upper)
    return s[:end]


# ----------------------------
# Rendering / VHS effects
# ----------------------------

@dataclass
class RenderSpec:
    width: int
    height: int
    fps: int
    sr: int
    slide_seconds: float
    popup_max: int
    popup_seconds: float


def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = im.size
    s = max(w / iw, h / ih)
    nw = int(iw * s)
    nh = int(ih * s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0 = (nw - w) // 2
    y0 = (nh - h) // 2
    return im2.crop((x0, y0, x0 + w, y0 + h)).convert("RGB")


def make_vignette_mask(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (np.sqrt(cx**2 + cy**2) + 1e-6)
    v = np.clip(1 - 0.65 * (r ** 1.7), 0.28, 1.0).astype(np.float32)
    return v[..., None]


def make_scan_mask(w: int, h: int) -> np.ndarray:
    m = np.ones((h, w, 1), dtype=np.float32)
    m[::2] = 0.83
    return m


def chroma_bleed(arr: np.ndarray, amt: int = 3) -> np.ndarray:
    out = arr.copy()
    out[:, :, 0] = np.roll(out[:, :, 0], -amt, axis=1)
    out[:, :, 2] = np.roll(out[:, :, 2], amt, axis=1)
    return out


def noise(arr: np.ndarray, level: int = 18) -> np.ndarray:
    n = np.random.randint(-level, level + 1, arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16) + n, 0, 255).astype(np.uint8)


def hslice_glitch(arr: np.ndarray, bands: int = 3, maxshift: int = 52) -> np.ndarray:
    h, w = arr.shape[:2]
    out = arr.copy()
    for _ in range(bands):
        y = random.randint(0, max(1, h - 20))
        hh = random.randint(6, 22)
        shift = random.randint(-maxshift, maxshift)
        out[y:y + hh] = np.roll(out[y:y + hh], shift, axis=1)
    return out


def tracking_line(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    out = arr.copy()
    y = random.randint(int(h * 0.55), max(int(h * 0.55) + 1, h - 10))
    hh = random.randint(4, 10)
    out[y:y + hh] = np.clip(out[y:y + hh].astype(np.int16) + random.randint(40, 85), 0, 255).astype(np.uint8)
    out[y:y + hh] = np.roll(out[y:y + hh], random.randint(-60, 60), axis=1)
    return out


def redactions(arr: np.ndarray, count: int = 2) -> np.ndarray:
    h, w = arr.shape[:2]
    out = arr.copy()
    for _ in range(count):
        bw = random.randint(int(w * 0.28), int(w * 0.70))
        bh = random.randint(18, 30)
        x = random.randint(18, max(19, w - bw - 18))
        y = random.randint(70, max(71, h - bh - 70))
        out[y:y + bh, x:x + bw] = 0
    return out


def wave_warp_face(arr: np.ndarray, t: float) -> np.ndarray:
    h, w = arr.shape[:2]
    out = arr.copy()
    for _ in range(4):
        x0 = random.randint(0, max(1, w - 140))
        ww = random.randint(60, 160)
        shift = int(10 * math.sin(t * 2.0 + x0 * 0.02))
        out[:, x0:x0 + ww] = np.roll(out[:, x0:x0 + ww], shift, axis=0)
    y0 = int(h * 0.30) + random.randint(-8, 8)
    y0 = max(0, min(h - 40, y0))
    out[y0:y0 + 34] = np.roll(out[y0:y0 + 34], random.randint(-18, 18), axis=1)
    return out


def alpha_over(bg: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    a = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(bg.astype(np.float32) * (1 - a) + fg_rgba[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)


def color_bars(w: int, h: int) -> np.ndarray:
    bars = np.zeros((h, w, 3), dtype=np.uint8)
    cols = [(235, 235, 235), (235, 235, 20), (20, 235, 235), (20, 235, 20),
            (235, 20, 235), (235, 20, 20), (20, 20, 235)]
    bw = w // len(cols)
    for i, c in enumerate(cols):
        bars[:, i * bw:(i + 1) * bw] = c
    return bars


def timecode_overlay(arr: np.ndarray, frame_index: int, fps: int) -> np.ndarray:
    h, w = arr.shape[:2]
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    
    # Try preferred font, else fallback
    font = None
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        for fallback in ["Arial", "Courier New", "LiberationMono-Regular"]:
            try:
                font = ImageFont.truetype(f"{fallback}.ttf", 16)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    secs = frame_index / fps
    hh = int(secs // 3600)
    mm = int((secs % 3600) // 60)
    ss = int(secs % 60)
    ff = int((secs - int(secs)) * fps)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}   CH03  SP"
    d.rectangle((10, h - 34, 300, h - 12), fill=(0, 0, 0))
    d.text((16, h - 32), tc, fill=(255, 255, 255), font=font)
    return np.array(im, dtype=np.uint8)


def slide_ui(w: int, h: int, title: str, body: str, theme: str, aesthetic: str) -> np.ndarray:
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Font handling
    fontT = fontB = fontM = None
    try:
        fontT = ImageFont.truetype("DejaVuSans.ttf", 26)
        fontB = ImageFont.truetype("DejaVuSans.ttf", 18)
        fontM = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        try:
            fontT = ImageFont.truetype("Arial.ttf", 26)
            fontB = ImageFont.truetype("Arial.ttf", 18)
            fontM = ImageFont.truetype("Courier New.ttf", 16)
        except Exception:
            fontT = fontB = fontM = ImageFont.load_default()

    if aesthetic == "ppt90s":
        if theme == "fatal": hdr = (255, 40, 40, 235)
        elif theme in ("protocol", "janedoe", "facefull"): hdr = (255, 220, 40, 235)
        else: hdr = (60, 210, 255, 235)
    else:
        hdr = (245, 245, 245, 235)

    d.rectangle((0, 0, w, 56), fill=hdr)
    d.text((16, 14), title, fill=(10, 10, 10, 255), font=fontT)

    px0, py0, px1, py1 = 20, 86, int(w * 0.74), 86 + 250
    d.rounded_rectangle((px0, py0, px1, py1), radius=14, fill=(255, 255, 255, 220),
                        outline=(0, 0, 0, 80), width=2)

    y = py0 + 14
    for line in body.split("\n"):
        d.text((px0 + 16, y), line, fill=(10, 10, 10, 255), font=fontB)
        y += 26

    d.rectangle((0, h - 46, w, h), fill=(0, 0, 0, 120))
    d.text((16, h - 36), "VHS TRAINING ARCHIVE  //  DO NOT DUPLICATE",
           fill=(255, 255, 255, 255), font=fontM)

    if theme == "janedoe":
        d.rectangle((px0 + 16, py1 - 52, px1 - 16, py1 - 20), fill=(230, 230, 230, 235))
        d.text((px0 + 22, py1 - 48), "FILE: JANE_DOE.TAPE  //  ACCESS: DENIED",
               fill=(0, 0, 0, 255), font=fontM)

    return np.array(layer, dtype=np.uint8)


# ----------------------------
# Audio: VHS bed + TTS + stabs
# ----------------------------

def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    y = x[::factor]
    y = np.repeat(y, factor)[:len(x)]
    return y


def render_tts_espeak(text: str, wav_path: Path, voice: str, speed: int, pitch: int, amp: int) -> None:
    cmd = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", str(wav_path), text]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def apply_voice_fx(data: np.ndarray, sr: int, rng: random.Random, intensity: float) -> np.ndarray:
    data = data.astype(np.float32)
    data /= (np.max(np.abs(data)) + 1e-6)

    data = np.tanh(data * (1.6 + 0.4 * intensity)) * 0.70

    t = np.linspace(0, len(data) / sr, len(data), False).astype(np.float32)
    ring = np.sin(2 * np.pi * rng.choice([180, 220, 260]) * t).astype(np.float32)
    data = (0.85 * data + 0.15 * data * ring).astype(np.float32)

    factor = rng.choice([3, 4, 5, 6]) if intensity > 0.6 else rng.choice([2, 3, 4])
    data = bitcrush(data, factor=factor)

    data += (rng.uniform(-1, 1) * 0.001) + (np.random.uniform(-1, 1, len(data)).astype(np.float32) * (0.004 + 0.004 * intensity))
    return data.astype(np.float32)


def mix_in(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int, gain: float) -> None:
    start = int(start_s * sr)
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[:end - start] * gain


# ----------------------------
# Story builder
# ----------------------------

def build_story_slides(rng: random.Random, spec: Dict[str, Any], theme_key: str, scraped: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg_story = spec.get("story", {})
    slide_sec = float(spec.get("render", {}).get("slide_seconds", 3.0))

    paragraphs = scraped.get("paragraphs", []) or []
    if not paragraphs:
        paragraphs = [f"{theme_key} is not stable. Do not describe it."]

    normal_blocks = [
        ("WORKPLACE WELLNESS", "• Hydrate every hour\n• Breathe in for 4, out for 6\n• Check posture\n• Smile (optional)"),
        ("HAPPINESS HABITS", "• Short walk after lunch\n• Call a friend\n• Keep a simple journal\n• Sleep at the same time"),
        ("PRODUCTIVITY TIP", "• One task at a time\n• Reduce distractions\n• Label objects\n• Keep desk tidy"),
    ]

    horror_blocks = [
        ("MEMORY RETENTION", "If you notice a face that feels wrong:\n1) Look away\n2) Do not describe it\n3) Touch an object you can name\n4) Leave immediately"),
        ("IDENTIFICATION FAILURE", "SUBJECT: JANE DOE\nSTATUS: [REDACTED]\nDO NOT ATTEMPT RECOGNITION"),
        ("TRACKING LOST", "FATAL ERROR: MNEMONIC_LEAK\nESC DISABLED\nDO NOT REWIND"),
        ("JANE DOE INTERMISSION", "LAST STABLE MEMORY: 03:17\nWITNESS COUNT: 1\nCOMPLIANCE: PARTIAL"),
    ]

    snippets: List[str] = []
    for p in paragraphs[:8]:
        s = re.sub(r"\s+", " ", p).strip()
        s = safe_truncate(rng, s, lo=28, hi=90)
        if s:
            snippets.append(s)

    encrypt_mode = (spec.get("control", {}).get("encrypt", {}).get("mode") or "none").lower()
    targets = spec.get("control", {}).get("encrypt", {}).get("targets") or ["theme", "easter"]
    targets = set([t.lower().strip() for t in targets if isinstance(t, str)])

    easter = f"{theme_key} / DO NOT RECALL / {rng.randint(100,999)}-{rng.randint(100,999)}"
    if "easter" in targets:
        easter = encrypt_text(easter, encrypt_mode)

    slides: List[Dict[str, Any]] = []

    slides.append({
        "kind": "bars",
        "title": "PLAYBACK",
        "body": f"JOB TRAINING TAPE / {rng.choice([1986, 1987, 1989, 1991])}\nCHANNEL 03  //  TRACKING: OK\nKEY: {encrypt_text(theme_key, encrypt_mode) if 'theme' in targets else theme_key}",
        "bg": None,
        "dur": max(2.0, slide_sec * 0.7),
    })

    for i in range(cfg_story.get("length", 10)):
        if i % 3 == 1:
            title, body = rng.choice(horror_blocks)
            slides.append({"kind": "protocol" if "RETENTION" in title else ("fatal" if "TRACKING" in title else "janedoe"),
                           "title": title,
                           "body": body,
                           "bg": rng.choice(["scene", "object", "face"]),
                           "dur": slide_sec})
        else:
            title, body = rng.choice(normal_blocks)
            if snippets and rng.random() < 0.65:
                body = body + "\n\n" + rng.choice(snippets)
            slides.append({"kind": "normal",
                           "title": title,
                           "body": body,
                           "bg": rng.choice(["scene", "object"]),
                           "dur": slide_sec})

        if rng.random() < 0.25 and snippets:
            slides.append({
                "kind": "normal",
                "title": f"TRAINING NOTE: {theme_key[:32]}",
                "body": rng.choice(snippets) + "\n\n" + rng.choice([
                    "Do not repeat what you read.",
                    "If the text changes, stop reading.",
                    "If you recognize it, look away.",
                ]),
                "bg": rng.choice(["scene", "face"]),
                "dur": slide_sec,
            })

    slides.append({
        "kind": "end",
        "title": "END OF MODULE",
        "body": "Thank you.\nDo not replay this tape.\nThe tape will replay you.\n\n" + easter,
        "bg": "scene",
        "dur": max(2.4, slide_sec * 0.8),
    })

    return slides


# ----------------------------
# Image pool building
# ----------------------------

def list_local_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def build_image_pool(rng: random.Random, spec: Dict[str, Any], theme_key: str, workdir: Path) -> Dict[str, List[Path]]:
    assets_cfg = spec.get("assets", {})
    local_dir = Path(assets_cfg.get("local_dir", "assets/local"))
    local_dir = (workdir / local_dir).resolve()

    faces_local = list_local_images(local_dir / "faces")
    objects_local = list_local_images(local_dir / "objects")
    scenes_local = list_local_images(local_dir / "scenes")
    mixed_local = list_local_images(local_dir / "mixed")

    web_cfg = spec.get("web", {})
    # Using 'web: true' fix logic from load_yaml, web_cfg is definitely a dict now
    enable_web = bool(web_cfg.get("enable", True))
    web_limit = int(web_cfg.get("image_limit", 10))
    cache_dir = workdir / (assets_cfg.get("cache_dir", "assets/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    faces_web: List[Path] = []
    objects_web: List[Path] = []
    scenes_web: List[Path] = []

    if enable_web:
        queries = [
            theme_key,
            f"{theme_key} portrait",
            f"{theme_key} building",
            f"{theme_key} object",
            "CCTV hallway",
            "office desk",
            "passport photo",
        ]
        rng.shuffle(queries)

        downloaded = 0
        for q in queries:
            if downloaded >= web_limit: break
            results = commons_image_search(q, limit=8)
            rng.shuffle(results)
            for item in results[:4]:
                if downloaded >= web_limit: break
                url = item["url"]
                fn = re.sub(r"[^a-zA-Z0-9_-]+", "_", item["title"])[:80]
                outp = cache_dir / f"{fn}_{downloaded}.jpg"
                if outp.exists():
                    downloaded += 1
                    continue
                ok = download_image(url, outp)
                if ok and outp.stat().st_size > 10_000:
                    downloaded += 1
                    qq = q.lower()
                    if "portrait" in qq or "passport" in qq or "face" in qq:
                        faces_web.append(outp)
                    elif "object" in qq or "desk" in qq or "keys" in qq:
                        objects_web.append(outp)
                    else:
                        scenes_web.append(outp)

    faces = faces_local + faces_web
    objects = objects_local + objects_web
    scenes = scenes_local + scenes_web

    if not faces: faces = mixed_local[:]
    if not objects: objects = mixed_local[:]
    if not scenes: scenes = mixed_local[:]

    if not any([faces, objects, scenes]):
        raise RuntimeError("No images found. Add images to assets/local/ or enable web scraping.")

    return {"faces": faces, "objects": objects, "scenes": scenes}


# ----------------------------
# Scrape text for theme
# ----------------------------

def scrape_theme_text(rng: random.Random, theme_key: str, max_paragraphs: int = 10) -> Dict[str, Any]:
    titles = [theme_key]
    titles += wiki_random_titles(rng, 3)

    paragraphs: List[str] = []
    for t in titles:
        ext = wiki_extract(t, max_chars=1600)
        if not ext: continue
        parts = re.split(r"(?<=[.!?])\s+", ext)
        for p in parts:
            p = p.strip()
            if len(p) >= 40:
                paragraphs.append(p)
    rng.shuffle(paragraphs)
    return {"titles": titles, "paragraphs": paragraphs[:max_paragraphs]}


# ----------------------------
# Rendering main
# ----------------------------

def run_ffmpeg_mux(video_path: Path, audio_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def stamp_popup(frame: np.ndarray, popup: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    ph, pw = popup.shape[:2]
    x = random.randint(0, max(0, w - pw))
    y = random.randint(70, max(71, h - ph - 70))
    out = frame.copy()
    out[y:y + ph, x:x + pw] = popup
    out[y:y + 3, x:x + pw] = 0
    out[y + ph - 3:y + ph, x:x + pw] = 0
    out[y:y + ph, x:x + 3] = 0
    out[y:y + ph, x + pw - 3:x + pw] = 0
    return out


def make_popup_from_image(im: Image.Image, w: int, h: int, rng: random.Random) -> np.ndarray:
    arr = np.array(cover_resize(im, w, h), dtype=np.uint8)
    x0 = rng.randint(0, max(1, w - 220))
    y0 = rng.randint(0, max(1, h - 220))
    crop = arr[y0:y0 + 200, x0:x0 + 200]
    pim = Image.fromarray(crop)

    if rng.random() < 0.6:
        pim = ImageOps.grayscale(pim)
        pim = ImageOps.autocontrast(ImageOps.posterize(pim, 3))
        pim = pim.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
    else:
        pim = ImageOps.autocontrast(pim)
        pim = pim.filter(ImageFilter.GaussianBlur(radius=0.8))

    pim = pim.resize((180, 180), Image.Resampling.NEAREST).convert("RGB")
    return np.array(pim, dtype=np.uint8)


def render_video_and_audio(
    rng: random.Random,
    spec: Dict[str, Any],
    theme_key: str,
    slides: List[Dict[str, Any]],
    pools: Dict[str, List[Path]],
    out_path: Path,
    workdir: Path,
) -> Path:
    render_cfg = spec.get("render", {})
    control_cfg = spec.get("control", {})

    rs = RenderSpec(
        width=int(render_cfg.get("width", 640)),
        height=int(render_cfg.get("height", 480)),
        fps=int(render_cfg.get("fps", 15)),
        sr=int(render_cfg.get("sr", 44100)),
        slide_seconds=float(render_cfg.get("slide_seconds", 3.0)),
        popup_max=int(render_cfg.get("popup_max", 3)),
        popup_seconds=float(render_cfg.get("popup_seconds", 0.5)),
    )

    aesthetic = (control_cfg.get("aesthetic") or "ppt90s").lower()
    glitch_level = float(control_cfg.get("effects", {}).get("glitch_level", 0.85))
    redact_level = float(control_cfg.get("effects", {}).get("redact_level", 0.75))
    face_level = float(control_cfg.get("effects", {}).get("face_uncanny", 0.9))
    enable_web = bool(spec.get("web", {}).get("enable", True))

    def pick_bg(kind: str) -> Image.Image:
        if kind in ("janedoe", "facefull"):
            p = rng.choice(pools["faces"])
        elif kind == "normal":
            p = rng.choice(pools["scenes"] if pools["scenes"] else pools["objects"])
        elif kind == "protocol":
            p = rng.choice(pools["objects"] if pools["objects"] else pools["scenes"])
        else:
            p = rng.choice((pools["scenes"] + pools["objects"] + pools["faces"]) or pools["faces"])
        return Image.open(p).convert("RGB")

    vignette = make_vignette_mask(rs.width, rs.height)
    scan = make_scan_mask(rs.width, rs.height)

    silent_video = workdir / "render_silent.mp4"
    audio_wav = workdir / "render_audio.wav"
    final_mp4 = out_path

    popup_sources: List[Image.Image] = []
    for p in (pools["faces"][:3] + pools["objects"][:4] + pools["scenes"][:3]):
        try:
            popup_sources.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not popup_sources:
        popup_sources.append(pick_bg("normal"))

    popup_pool = [make_popup_from_image(im, rs.width, rs.height, rng) for im in popup_sources]

    total_duration = float(sum(s.get("dur", rs.slide_seconds) for s in slides))
    total_frames = int(total_duration * rs.fps)
    popup_events: List[int] = []
    if rs.popup_max > 0 and total_frames > 10:
        for _ in range(rs.popup_max):
            popup_events.append(rng.randint(int(0.12 * total_frames), total_frames - 2))
        popup_events = sorted(set(popup_events))[:rs.popup_max]
    popup_len_frames = max(1, int(rs.popup_seconds * rs.fps))

    flash_frames: set[int] = set()
    for _ in range(10):
        flash_frames.add(rng.randint(int(0.10 * total_frames), total_frames - 2))

    writer = imageio.get_writer(str(silent_video), fps=rs.fps, codec="libx264", bitrate="2400k")

    frame_idx = 0
    slide_starts: List[float] = []
    t_cursor = 0.0

    for s in slides:
        dur = float(s.get("dur", rs.slide_seconds))
        nF = max(1, int(dur * rs.fps))
        slide_starts.append(t_cursor)
        t_cursor += dur

        kind = s["kind"]
        title = s["title"]
        body = s["body"]

        theme = "normal"
        if kind in ("protocol", "facefull"): theme = "protocol"
        elif kind == "fatal": theme = "fatal"
        elif kind == "janedoe": theme = "janedoe"

        ui = slide_ui(rs.width, rs.height, title, body, theme=theme, aesthetic=aesthetic)

        for fi in range(nF):
            t = frame_idx / rs.fps

            if kind == "bars":
                frame = color_bars(rs.width, rs.height)
            elif kind == "fatal":
                frame = np.full((rs.height, rs.width, 3), (20, 60, 170), dtype=np.uint8)
                if rng.random() < 0.15 * glitch_level:
                    frame = (0.6 * frame + 0.4 * color_bars(rs.width, rs.height)).astype(np.uint8)
            else:
                bg = cover_resize(pick_bg(kind), rs.width, rs.height)
                frame = np.array(bg, dtype=np.uint8)

            if fi % 4 == 0:
                frame = np.roll(frame, rng.randint(-2, 2), axis=1)

            if kind in ("janedoe", "facefull") and pools["faces"]:
                face_im = cover_resize(Image.open(rng.choice(pools["faces"])).convert("RGB"), rs.width, rs.height)
                face_arr = np.array(face_im, dtype=np.uint8)
                face_arr = wave_warp_face(face_arr, t)
                if face_level > 0.6:
                    face_arr = chroma_bleed(face_arr, amt=5)
                alpha = 0.78 if kind == "janedoe" else 0.65
                frame = np.clip(frame.astype(np.float32) * (1 - alpha) + face_arr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                if rng.random() < 0.55 * redact_level:
                    frame = redactions(frame, count=2 if kind == "facefull" else 3)

            frame = alpha_over(frame, ui)

            for ev in popup_events:
                if ev <= frame_idx < ev + popup_len_frames:
                    pop = rng.choice(popup_pool)
                    if rng.random() < 0.35: pop = 255 - pop
                    frame = stamp_popup(frame, pop)

            if frame_idx in flash_frames and rng.random() < 0.85:
                flash_src = np.array(cover_resize(pick_bg(rng.choice(["normal", "protocol", "facefull"])), rs.width, rs.height), dtype=np.uint8)
                flash_src = hslice_glitch(chroma_bleed(noise(flash_src, 35), 5), bands=5, maxshift=75)
                frame = flash_src

            frame = chroma_bleed(frame, amt=3 if rng.random() < 0.8 else 5)
            if rng.random() < 0.35 * glitch_level: frame = tracking_line(frame)
            if rng.random() < 0.22 * glitch_level: frame = hslice_glitch(frame, bands=3, maxshift=62)
            frame = noise(frame, level=18)
            frame = (frame.astype(np.float32) * scan).astype(np.uint8)
            frame = (frame.astype(np.float32) * vignette).astype(np.uint8)

            if fi < 3 or fi > nF - 4:
                frame = hslice_glitch(frame, bands=4, maxshift=75)
                if rng.random() < 0.45: frame = 255 - frame

            frame = timecode_overlay(frame, frame_idx, rs.fps)
            writer.append_data(frame)
            frame_idx += 1

    writer.close()

    dur_sec = frame_idx / rs.fps
    total_samples = int(dur_sec * rs.sr)
    t = np.linspace(0, dur_sec, total_samples, False).astype(np.float32)

    audio = (np.random.uniform(-1, 1, total_samples).astype(np.float32) * 0.06)
    audio += 0.03 * np.sin(2 * np.pi * 55 * t) + 0.018 * np.sin(2 * np.pi * 110 * t)
    audio += 0.02 * np.sin(2 * np.pi * 30 * t)
    audio *= (0.82 + 0.18 * np.sin(2 * np.pi * 0.18 * t)).astype(np.float32)

    stab_count = int(spec.get("audio", {}).get("stabs", 18))
    for _ in range(stab_count):
        p0 = rng.randint(0, max(1, total_samples - 2))
        span = rng.randint(int(0.01 * rs.sr), int(0.06 * rs.sr))
        p1 = min(total_samples, p0 + span)
        burst = (np.random.uniform(-1, 1, p1 - p0).astype(np.float32) * rng.uniform(0.25, 0.55))
        audio[p0:p1] += burst

    for sec in [12, 20, 26]:
        p0 = int(sec * rs.sr)
        if p0 < total_samples:
            p1 = min(total_samples, p0 + int(0.18 * rs.sr))
            tt = np.linspace(0, (p1 - p0) / rs.sr, p1 - p0, False).astype(np.float32)
            audio[p0:p1] += 0.22 * np.sin(2 * np.pi * 880 * tt).astype(np.float32)

    voice_cfg = spec.get("tts", {})
    enable_tts = bool(voice_cfg.get("enable", True))
    voice_profiles = voice_cfg.get("profiles") or []
    if not voice_profiles:
        voice_profiles = [
            {"voice": "en-us", "speed": 155, "pitch": 30, "amp": 170},
            {"voice": "en", "speed": 145, "pitch": 18, "amp": 165},
            {"voice": "en-us", "speed": 165, "pitch": 45, "amp": 175},
        ]
    fx_intensity = float(voice_cfg.get("fx_intensity", 0.85))

    narrations = build_narrations_for_slides(rng, theme_key, slides, spec)

    tts_dir = workdir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    if enable_tts:
        for i, text in enumerate(narrations):
            prof = rng.choice(voice_profiles)
            wav_path = tts_dir / f"tts_{i:02d}.wav"
            render_tts_espeak(
                text=text,
                wav_path=wav_path,
                voice=str(prof.get("voice", "en-us")),
                speed=int(prof.get("speed", 155)),
                pitch=int(prof.get("pitch", 30)),
                amp=int(prof.get("amp", 170)),
            )
            sr2, data = wav_read(str(wav_path))
            if data.ndim > 1: data = data.mean(axis=1)
            data = data.astype(np.float32)
            if sr2 != rs.sr:
                x = np.linspace(0, 1, len(data), False)
                x2 = np.linspace(0, 1, int(len(data) * rs.sr / sr2), False)
                data = np.interp(x2, x, data).astype(np.float32)

            data = apply_voice_fx(data, rs.sr, rng, intensity=fx_intensity)
            start_s = slide_starts[min(i, len(slide_starts) - 1)] + 0.25
            mix_in(audio, data, start_s=start_s, sr=rs.sr, gain=0.95)

    audio /= (np.max(np.abs(audio)) + 1e-6)
    wav_write(str(audio_wav), rs.sr, (audio * 32767).astype(np.int16))

    run_ffmpeg_mux(silent_video, audio_wav, final_mp4)
    return final_mp4


def build_narrations_for_slides(rng: random.Random, theme_key: str, slides: List[Dict[str, Any]], spec: Dict[str, Any]) -> List[str]:
    encrypt_mode = (spec.get("control", {}).get("encrypt", {}).get("mode") or "none").lower()
    targets = spec.get("control", {}).get("encrypt", {}).get("targets") or ["easter"]
    targets = set([t.lower().strip() for t in targets if isinstance(t, str)])

    hooks = [
        "Please follow along.",
        "If you feel anxious, breathe slowly.",
        "Do not attempt to interpret the face.",
        "If the slide changes, stop reading.",
        "Do not repeat what you heard.",
    ]

    narr: List[str] = []
    for s in slides:
        kind = s["kind"]
        title = s["title"]
        body = s["body"]

        if kind == "bars":
            line = f"Playback. Job training tape. Theme key: {theme_key}. {rng.choice(hooks)}"
        elif kind == "normal":
            wrong = rng.choice([
                "If the journal writes back, stop.",
                "If you recognize the face, look away.",
                "If you hear your name, do not answer.",
            ])
            line = f"{title}. {body.replace(chr(10), ' ')}. {wrong}"
        elif kind == "protocol":
            line = f"{title}. {body.replace(chr(10), ' ')}. Do not describe what you saw."
        elif kind == "fatal":
            line = "Tracking lost. Fatal error. Mnemonic leak detected. Escape is disabled. Do not rewind."
        elif kind == "janedoe":
            line = "Jane Doe intermission. Identity redacted. Last stable memory, zero three seventeen. Compliance partial."
        else:
            line = f"{title}. {body.replace(chr(10), ' ')}."

        if kind in ("janedoe", "fatal") and rng.random() < 0.55 and "easter" in targets:
            secret = f"{theme_key}::{rng.randint(1000,9999)}::{rng.choice(['DO NOT RECALL','LOOK AWAY','DO NOT NAME IT'])}"
            secret = encrypt_text(secret, encrypt_mode)
            line += f" {secret}."

        narr.append(re.sub(r"\s+", " ", line).strip())
    return narr


# ----------------------------
# CLI / Main
# ----------------------------

def main() -> None:
    check_system_deps()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--seed", default=None, help="Override seed (int).")
    ap.add_argument("--theme", default=None, help="Override theme key")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    cfg_seed_raw = cfg.get("seed", "auto")
    if args.seed is not None:
        seed = parse_int_safe(args.seed, default=now_seed())
    else:
        seed = parse_int_safe(cfg_seed_raw, default=now_seed())

    rng = random.Random(seed)

    theme_key = (args.theme or cfg.get("brain", {}).get("theme") or "").strip()
    if not theme_key:
        theme_key = choose_theme_key(rng, cfg)

    workdir = Path(cfg.get("workdir", ".work")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    web_section = cfg.get("web")
    if isinstance(web_section, dict):
        text_para = int(web_section.get("text_paragraphs", 10))
    else:
        text_para = 10

    scraped = scrape_theme_text(rng, theme_key, max_paragraphs=text_para)
    pools = build_image_pool(rng, cfg, theme_key, workdir=workdir)

    slides = build_story_slides(rng, cfg, theme_key, scraped)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final = render_video_and_audio(rng, cfg, theme_key, slides, pools, out_path, workdir=workdir)
    print(f"OK: wrote {final} (seed={seed}, theme={theme_key})")


if __name__ == "__main__":
    main()

