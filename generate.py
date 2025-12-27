#!/usr/bin/env python3
"""
pptex - Always-different analogue/PPT horror video generator (GitHub friendly)

Features:
- "Brain" picks a theme key per run (seeded) and scrapes content from:
  * Wikipedia REST summary API
  * Wikimedia Commons (random media via MediaWiki API)
- Mixes normal/wellness slides with scary/protocol/JaneDoe/fatal-error slides
- VHS artifacts, glitches, redactions, 90s PPT gradients, transitions
- Up to 3 popup "flash" images for ~0.5 seconds
- Audio: harsh-noise bed + beeps + optional local music + per-slide espeak TTS (varied voices)
- Outputs mp4 (H.264 + AAC) via ffmpeg mux
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import imageio.v2 as imageio
from scipy.io.wavfile import write as write_wav, read as read_wav


# -----------------------------
# Utilities
# -----------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"[pptex] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def run(cmd: List[str], check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    if not quiet:
        print("[pptex] $", " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None
    )

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")

ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def resolve_env_placeholders(value: str) -> str:
    """
    Turns "${SEED}" into env SEED (if present), else leaves it as-is.
    Also supports strings like "seed_${SEED}_x".
    """
    def repl(m: re.Match) -> str:
        k = m.group(1)
        return os.environ.get(k, m.group(0))
    return ENV_VAR_PATTERN.sub(repl, value)

def cfg_get(cfg: dict, key: str, default):
    v = cfg.get(key, default)
    if isinstance(v, str):
        v = resolve_env_placeholders(v)
    return v


# -----------------------------
# Web scraping (safe-ish sources)
# -----------------------------

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_RANDOM  = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
MW_API       = "https://commons.wikimedia.org/w/api.php"

UA = {"User-Agent": "pptex/1.0 (github-actions)"}  # important for APIs

def fetch_json(url: str, timeout: float = 20.0) -> dict:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_text_theme(seed_rng: random.Random, min_len: int = 400) -> Tuple[str, str, List[str]]:
    """
    Choose a theme ("key") from a random Wikipedia page, return:
    (title, extract, keywords)
    """
    data = fetch_json(WIKI_RANDOM)
    title = data.get("title") or "UNKNOWN"
    extract = data.get("extract") or ""
    # sometimes random summaries are super short; retry a bit
    for _ in range(4):
        if len(extract) >= min_len:
            break
        data = fetch_json(WIKI_RANDOM)
        title = data.get("title") or title
        extract = data.get("extract") or extract

    # simple keyword extraction (no heavy NLP)
    words = re.findall(r"[A-Za-z]{4,}", extract.lower())
    stop = set("""
        this that with from have were been will would there their about which
        into your you and the for are not but they them then than when what
        where while also because between after before during without within
        """.split())
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    kw = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keywords = [k for k, _ in kw[:12]]
    seed_rng.shuffle(keywords)
    keywords = keywords[:8]
    return title, extract, keywords

def mw_random_image_urls(seed_rng: random.Random, n: int = 8) -> List[str]:
    """
    Get random media from Wikimedia Commons. We filter for "imageinfo" url.
    """
    urls: List[str] = []
    attempts = 0
    while len(urls) < n and attempts < n * 4:
        attempts += 1
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "6",  # File:
            "grnlimit": "1",
            "prop": "imageinfo",
            "iiprop": "url",
        }
        r = requests.get(MW_API, params=params, headers=UA, timeout=20)
        r.raise_for_status()
        data = r.json()
        pages = (data.get("query") or {}).get("pages") or {}
        for _, p in pages.items():
            infos = p.get("imageinfo") or []
            if not infos:
                continue
            u = infos[0].get("url")
            if u and any(u.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                urls.append(u)
    # de-dupe, shuffle
    urls = list(dict.fromkeys(urls))
    seed_rng.shuffle(urls)
    return urls[:n]

def download_image(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


# -----------------------------
# Visual engine
# -----------------------------

@dataclass
class VideoSpec:
    width: int
    height: int
    fps: int
    slide_seconds: float
    popups_max: int
    popup_seconds: float
    seed: int
    theme: str
    theme_keywords: List[str]
    vibe: str

@dataclass
class Toggles:
    vhs: bool
    glitch: bool
    redactions: bool
    timecode: bool
    ppt90s: bool
    faces: bool

def load_font(prefer_mono: bool = False, size: int = 18) -> ImageFont.FreeTypeFont:
    # DejaVu is available on ubuntu runners
    try:
        if prefer_mono:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def cover_resize(im: Image.Image, W: int, H: int) -> Image.Image:
    iw, ih = im.size
    s = max(W/iw, H/ih)
    nw, nh = int(iw*s), int(ih*s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0 = (nw - W)//2
    y0 = (nh - H)//2
    return im2.crop((x0, y0, x0+W, y0+H))

def make_gradient_bg(rng: random.Random, W: int, H: int) -> np.ndarray:
    c1 = np.array([rng.randint(40,255), rng.randint(40,255), rng.randint(40,255)], dtype=np.float32)
    c2 = np.array([rng.randint(40,255), rng.randint(40,255), rng.randint(40,255)], dtype=np.float32)
    gx = np.linspace(0, 1, W, dtype=np.float32)[None,:,None]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:,None,None]
    mix = np.clip(0.7*gx + 0.3*gy, 0, 1)
    img = c1*(1-mix) + c2*mix
    return np.clip(img, 0, 255).astype(np.uint8)

def alpha_over(bg: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    a = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    return np.clip(bg.astype(np.float32)*(1-a) + fg_rgba[:,:,:3].astype(np.float32)*a, 0, 255).astype(np.uint8)

def chroma_bleed(arr: np.ndarray, amt: int = 3) -> np.ndarray:
    out = arr.copy()
    out[:,:,0] = np.roll(out[:,:,0], -amt, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  amt, axis=1)
    return out

def scanlines(arr: np.ndarray, strength: float = 0.84) -> np.ndarray:
    out = arr.astype(np.float32)
    out[::2] *= strength
    return np.clip(out, 0, 255).astype(np.uint8)

def vignette(arr: np.ndarray, W: int, H: int, power: float = 1.7) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / np.sqrt(cx**2 + cy**2)
    v = np.clip(1 - 0.65*(r**power), 0.28, 1.0).astype(np.float32)
    out = (arr.astype(np.float32) * v[...,None])
    return np.clip(out, 0, 255).astype(np.uint8)

def noise(arr: np.ndarray, level: int = 18) -> np.ndarray:
    n = np.random.randint(-level, level+1, arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16) + n, 0, 255).astype(np.uint8)

def hslice_glitch(arr: np.ndarray, rng: random.Random, bands: int = 3, maxshift: int = 64) -> np.ndarray:
    H = arr.shape[0]
    out = arr.copy()
    for _ in range(bands):
        y = rng.randint(0, H-18)
        hh = rng.randint(6, 22)
        shift = rng.randint(-maxshift, maxshift)
        out[y:y+hh] = np.roll(out[y:y+hh], shift, axis=1)
    return out

def redaction_bars(arr: np.ndarray, rng: random.Random, count: int = 2) -> np.ndarray:
    H, W = arr.shape[:2]
    out = arr.copy()
    for _ in range(count):
        ww = rng.randint(int(W*0.25), int(W*0.75))
        hh = rng.randint(14, 28)
        x = rng.randint(12, W-ww-12)
        y = rng.randint(60, H-60)
        out[y:y+hh, x:x+ww] = 0
    return out

def timecode_overlay(arr: np.ndarray, fps: int, frame_index: int) -> np.ndarray:
    H, W = arr.shape[:2]
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    font = load_font(prefer_mono=True, size=16)
    secs = frame_index / fps
    hh = int(secs//3600); mm = int((secs%3600)//60); ss = int(secs%60); ff = int((secs - int(secs))*fps)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}  CH03  SP"
    d.rectangle((10, H-34, 285, H-12), fill=(0,0,0))
    d.text((16, H-32), tc, fill=(255,255,255), font=font)
    return np.array(im, dtype=np.uint8)

def wave_warp_face(arr: np.ndarray, rng: random.Random, t: float) -> np.ndarray:
    H, W = arr.shape[:2]
    out = arr.copy()
    # vertical strip wobble
    for _ in range(5):
        x0 = rng.randint(0, W-140)
        ww = rng.randint(60, 170)
        shift = int(10*math.sin(t*2.0 + x0*0.02))
        out[:, x0:x0+ww] = np.roll(out[:, x0:x0+ww], shift, axis=0)
    # eye-band shear
    y0 = int(H*0.30) + rng.randint(-10, 10)
    out[y0:y0+36] = np.roll(out[y0:y0+36], rng.randint(-22, 22), axis=1)
    return out

def pop_color_boost(im: Image.Image) -> Image.Image:
    # 90s PPT: oversaturated + contrast
    im = ImageEnhance.Color(im).enhance(1.35)
    im = ImageEnhance.Contrast(im).enhance(1.10)
    return im

def make_slide_ui(
    rng: random.Random,
    W: int,
    H: int,
    title: str,
    body_lines: List[str],
    footer: str,
    theme: str,
    toggles: Toggles
) -> np.ndarray:
    # 90s PPT overlay: neon header + rounded boxes
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer)
    fontT = load_font(False, 28)
    fontB = load_font(False, 18)
    fontM = load_font(True, 16)

    if theme == "fatal":
        hdr = (255, 60, 60, 235)
    elif theme in ("protocol", "janedoe"):
        hdr = (255, 230, 70, 235)
    else:
        hdr = (70, 215, 255, 235)

    d.rectangle((0,0,W,60), fill=hdr)
    d.text((16, 16), title[:70], fill=(10,10,10,255), font=fontT)

    px0, py0, px1, py1 = 22, 94, int(W*0.75), 94 + 260
    d.rounded_rectangle((px0,py0,px1,py1), radius=14, fill=(255,255,255,220), outline=(0,0,0,80), width=2)

    y = py0 + 14
    for ln in body_lines[:10]:
        d.text((px0+16, y), ln[:72], fill=(10,10,10,255), font=fontB)
        y += 26

    d.rectangle((0, H-52, W, H), fill=(0,0,0,120))
    d.text((16, H-40), footer, fill=(255,255,255,255), font=fontM)

    # extra: redaction overlays for janedoe/protocol
    if toggles.redactions and theme in ("janedoe", "protocol") and rng.random() < 0.9:
        for _ in range(rng.randint(2, 4)):
            ww = rng.randint(180, 460)
            hh = rng.randint(16, 28)
            x = rng.randint(18, W-ww-18)
            yb = rng.randint(80, H-90)
            d.rectangle((x, yb, x+ww, yb+hh), fill=(0,0,0,255))

    # fatal: add BSOD-ish lines
    if theme == "fatal":
        d.rectangle((0, 60, W, H), fill=(20, 60, 170, 210))
        lines = [
            "A fatal exception has occurred.",
            "The system has been halted to prevent memory damage.",
            "Do not restart. Do not recall. Do not describe.",
            f"ERROR: {rng.choice(['MNEMONIC_LEAK','ANCHOR_FAIL','ID_NULL'])} (0x0000F4)",
            "Press ESC to continue (ESC is disabled).",
        ]
        y2 = 88
        for ln in lines:
            d.text((22, y2), ln, fill=(255,255,255,255), font=fontM)
            y2 += 26

    return np.array(layer, dtype=np.uint8)

def make_popup_from_image(rng: random.Random, src: np.ndarray) -> np.ndarray:
    H, W = src.shape[:2]
    x0 = rng.randint(0, W-240) if W >= 240 else 0
    y0 = rng.randint(0, H-240) if H >= 240 else 0
    crop = src[y0:y0+220, x0:x0+220] if W >= 240 and H >= 240 else src
    im = Image.fromarray(crop)
    # “uncanny wrongness” variants
    if rng.random() < 0.6:
        im = ImageOps.grayscale(im)
        im = ImageOps.autocontrast(ImageOps.posterize(im, 3))
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
        im = im.convert("RGB")
    else:
        im = pop_color_boost(im)
        im = im.filter(ImageFilter.GaussianBlur(radius=0.9))
    im = im.resize((200,200), Image.Resampling.NEAREST)
    return np.array(im, dtype=np.uint8)

def stamp_popup(frame: np.ndarray, rng: random.Random, popup: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    ph, pw = popup.shape[:2]
    x = rng.randint(0, W-pw)
    y = rng.randint(70, H-ph-70)
    out = frame.copy()
    out[y:y+ph, x:x+pw] = popup
    out[y:y+3, x:x+pw] = 0; out[y+ph-3:y+ph, x:x+pw] = 0
    out[y:y+ph, x:x+3] = 0; out[y:y+ph, x+pw-3:x+pw] = 0
    return out


# -----------------------------
# Audio engine
# -----------------------------

def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    y = x[::factor]
    y = np.repeat(y, factor)[:len(x)]
    return y

def mix_in(dst: np.ndarray, src: np.ndarray, sr: int, start_s: float, gain: float = 0.9) -> None:
    start = int(start_s * sr)
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[:end-start] * gain

def gen_vhs_bed(rng: random.Random, sr: int, dur_s: float) -> np.ndarray:
    n = int(sr * dur_s)
    t = np.linspace(0, dur_s, n, False).astype(np.float32)

    audio = (np.random.uniform(-1,1,n).astype(np.float32) * 0.06)
    audio += 0.03*np.sin(2*np.pi*55*t) + 0.018*np.sin(2*np.pi*110*t)
    audio += 0.02*np.sin(2*np.pi*30*t)
    audio *= (0.82 + 0.18*np.sin(2*np.pi*0.18*t)).astype(np.float32)

    # abrupt pops
    for _ in range(20):
        p0 = rng.randint(0, n-1)
        span = rng.randint(int(0.01*sr), int(0.06*sr))
        p1 = min(n, p0+span)
        burst = (np.random.uniform(-1,1,p1-p0).astype(np.float32) * rng.uniform(0.25, 0.55))
        audio[p0:p1] += burst

    # occasional beeps
    for sec in [rng.uniform(6, dur_s-1), rng.uniform(6, dur_s-1), rng.uniform(6, dur_s-1)]:
        p0 = int(sec*sr)
        p1 = min(n, p0+int(0.18*sr))
        tt = np.linspace(0, (p1-p0)/sr, p1-p0, False).astype(np.float32)
        audio[p0:p1] += 0.22*np.sin(2*np.pi*880*tt).astype(np.float32)

    return audio

def espeak_tts(
    text: str,
    out_wav: Path,
    voice: str,
    speed: int,
    pitch: int,
    amp: int,
) -> None:
    cmd = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", str(out_wav), text]
    run(cmd, check=True, quiet=True)

def load_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    sr, data = read_wav(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    x = data.astype(np.float32)
    x /= (np.max(np.abs(x)) + 1e-6)
    return sr, x


# -----------------------------
# Story / Brain
# -----------------------------

NORMAL_TEMPLATES = [
    "WORKPLACE WELLNESS",
    "HAPPINESS HABITS",
    "PRODUCTIVITY TIP",
    "TEAM CULTURE",
    "HEALTH REMINDER",
]
SCARY_TEMPLATES = [
    "MEMORY RETENTION",
    "ENTITY AVOIDANCE PROTOCOL",
    "JANE DOE INTERMISSION",
    "IDENTIFICATION FAILURE",
    "TRACKING LOST",
    "FATAL ERROR",
]

def encrypt_word(word: str, key: int) -> str:
    # simple Caesar-ish shift for ARG flavor
    out = []
    for ch in word:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch)-97+key) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch)-65+key) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)

def build_story_slides(rng: random.Random, spec: VideoSpec, scraped_extract: str, local_images: List[Path]) -> List[dict]:
    """
    Return list of slide dicts:
    kind: normal/protocol/janedoe/fatal/facefull/bars/end
    title/body/bg_key
    """
    # carve extract into usable chunks
    sentences = re.split(r"(?<=[.!?])\s+", scraped_extract.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 30]
    rng.shuffle(sentences)

    # pick "anchors" (objects) from keywords + a few weird ones
    weird_objects = ["KEYS", "TELEPHONE", "MIRROR", "BADGE", "COFFEE CUP", "FIRE EXIT MAP", "STAPLER", "CLOCK"]
    rng.shuffle(weird_objects)
    anchors = [w.upper() for w in spec.theme_keywords[:3]] + weird_objects[:3]
    rng.shuffle(anchors)

    # encrypted easter egg tokens
    egg_plain = rng.choice(spec.theme_keywords or ["memory", "training", "signal"])
    egg_code  = encrypt_word(egg_plain, key=(spec.seed % 17) + 3)

    # create slide plan (mix normal + scary)
    plan = [
        ("bars",   "PLAYBACK", ["JOB TRAINING TAPE / 1996", f"THEME KEY: {spec.theme.upper()}", "CHANNEL 03  //  TRACKING: OK"], None),
        ("normal", rng.choice(NORMAL_TEMPLATES), [], "SCRAPED"),
        ("normal", rng.choice(NORMAL_TEMPLATES), [], "SCRAPED"),
        ("protocol", "MEMORY RETENTION", [], "SCRAPED"),
        ("facefull", "IDENTIFICATION FAILURE", [f"SUBJECT: JANE DOE", "STATUS: [REDACTED]", "DO NOT ATTEMPT RECOGNITION"], "FACE"),
        ("normal", rng.choice(NORMAL_TEMPLATES), [], "SCRAPED"),
        ("protocol", "ENTITY AVOIDANCE PROTOCOL", [], "SCRAPED"),
        ("fatal", "FATAL ERROR", ["MNEMONIC_LEAK.EXE", "ESC DISABLED", "DO NOT REWIND"], None),
        ("janedoe", "JANE DOE INTERMISSION", ["LAST STABLE MEMORY: 03:17", "WITNESS COUNT: 1", f"EASTER EGG: {egg_code}"], "FACE"),
        ("end", "END OF MODULE", ["Thank you.", "Do not replay this tape.", "The tape will replay you."], "SCRAPED"),
    ]

    # Fill bodies for normal/protocol slides with scraped + anchor nonsense
    slides: List[dict] = []
    for kind, title, body, bg_mode in plan:
        if kind in ("normal", "protocol"):
            # pick 3–5 lines from sentences and twist them
            lines: List[str] = []
            for _ in range(rng.randint(3, 5)):
                s = sentences.pop() if sentences else rng.choice(["Maintain composure.", "Follow policy.", "Do not improvise."])
                # “wrongness”: insert anchors / redact / half-sentence
                if rng.random() < 0.50:
                    s = s[:rng.randint(28, min(70, len(s)))]
                if rng.random() < 0.65:
                    s += f"  //  ANCHOR: {rng.choice(anchors)}"
                if rng.random() < 0.30:
                    s += "  [REDACTED]"
                lines.append(s)

            if kind == "protocol":
                # add numbered steps
                steps = [
                    "1) Look away.",
                    "2) Do not describe it.",
                    "3) Touch an object you can name.",
                    "4) Leave immediately.",
                ]
                rng.shuffle(steps)
                lines = steps[:3] + lines[:2]

            body = lines

        # background selection strategy
        slides.append({
            "kind": kind,
            "title": title,
            "body_lines": body,
            "bg_mode": bg_mode,
        })

    return slides


# -----------------------------
# Main generator
# -----------------------------

def find_local_images(repo_root: Path) -> List[Path]:
    img_dir = repo_root / "assets" / "images"
    if not img_dir.exists():
        return []
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs += list(img_dir.glob(ext))
    return imgs

def scrape_images_into(cache_dir: Path, rng: random.Random, count: int) -> List[Path]:
    ensure_dir(cache_dir)
    urls = mw_random_image_urls(rng, n=count)
    out = []
    for i, u in enumerate(urls):
        p = cache_dir / f"mw_{i:02d}_{sha1(u)[:10]}.img"
        # keep real ext if possible
        ext = Path(u.split("?")[0]).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"
        p = p.with_suffix(ext)
        if download_image(u, p):
            out.append(p)
    return out

def choose_faces_pool(all_imgs: List[Path], rng: random.Random) -> List[Path]:
    # naive: pick a few random images to serve as "faces"
    rng.shuffle(all_imgs)
    return all_imgs[:max(2, min(6, len(all_imgs)))]

def load_as_np_image(path: Path, W: int, H: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im = cover_resize(im, W, H)
    im = pop_color_boost(im)
    return np.array(im, dtype=np.uint8)

def render_video(
    out_silent: Path,
    slides: List[dict],
    images_pool: List[Path],
    faces_pool: List[Path],
    spec: VideoSpec,
    toggles: Toggles,
    rng: random.Random
) -> Tuple[int, int]:
    """
    Returns (total_frames, popups_used)
    """
    W, H, FPS = spec.width, spec.height, spec.fps
    writer = imageio.get_writer(str(out_silent), fps=FPS, codec="libx264", bitrate="2400k")

    # pre-load all images into memory as np arrays (faster in GH)
    loaded_np: List[np.ndarray] = []
    for p in images_pool:
        try:
            loaded_np.append(load_as_np_image(p, W, H))
        except Exception:
            continue
    if not loaded_np:
        # fallback gradient backgrounds
        loaded_np = [make_gradient_bg(rng, W, H)]

    faces_np: List[np.ndarray] = []
    for p in faces_pool:
        try:
            arr = load_as_np_image(p, W, H)
            # push uncanny: posterize a bit
            im = Image.fromarray(arr)
            im = ImageOps.autocontrast(im)
            im = ImageOps.posterize(im, 4)
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
            faces_np.append(np.array(im, dtype=np.uint8))
        except Exception:
            continue
    if not faces_np:
        faces_np = [loaded_np[0]]

    # popup flashes schedule (max 3, ~0.5s each)
    total_frames_est = int(len(slides) * spec.slide_seconds * FPS)
    popups_used = min(spec.popups_max, 3)
    flash_events: List[Tuple[int, int, np.ndarray]] = []
    for _ in range(popups_used):
        start = rng.randint(int(0.10*total_frames_est), max(int(0.12*total_frames_est), total_frames_est-10))
        dur = max(1, int(spec.popup_seconds * FPS))
        img = rng.choice(faces_np if rng.random() < 0.7 else loaded_np)
        flash_events.append((start, start+dur, img))
    flash_events.sort(key=lambda x: x[0])

    frame_idx = 0
    for si, s in enumerate(slides):
        kind = s["kind"]
        nF = int(spec.slide_seconds * FPS)
        if kind in ("bars", "fatal"):
            nF = int((spec.slide_seconds * 0.75) * FPS) if kind == "bars" else int((spec.slide_seconds * 0.95) * FPS)

        theme = "normal"
        if kind in ("protocol", "facefull"): theme = "protocol"
        if kind == "fatal": theme = "fatal"
        if kind == "janedoe": theme = "janedoe"

        footer = f"VHS TRAINING ARCHIVE // {spec.vibe.upper()} // DO NOT DUPLICATE"
        ui = make_slide_ui(rng, W, H, s["title"], s["body_lines"], footer, theme, toggles)

        # choose backgrounds
        bg = rng.choice(loaded_np)
        if kind == "bars":
            # SMPTE-ish bars
            frame_base = np.zeros((H, W, 3), dtype=np.uint8)
            cols = [(235,235,235),(235,235,20),(20,235,235),(20,235,20),(235,20,235),(235,20,20),(20,20,235)]
            bw = W//len(cols)
            for i,c in enumerate(cols):
                frame_base[:, i*bw:(i+1)*bw] = c
        elif kind == "fatal":
            frame_base = np.full((H, W, 3), (20,60,170), dtype=np.uint8)
        else:
            frame_base = bg

        for fi in range(nF):
            t = frame_idx / FPS
            frame = frame_base.copy()

            # mild drift
            if fi % 4 == 0:
                frame = np.roll(frame, rng.randint(-2,2), axis=1)

            # facefull / janedoe: bring face fully visible
            if toggles.faces and kind in ("facefull", "janedoe"):
                face = rng.choice(faces_np)
                warped = wave_warp_face(face, rng, t)
                alpha = 0.72 if kind == "janedoe" else 0.62
                frame = np.clip(frame.astype(np.float32)*(1-alpha) + warped.astype(np.float32)*alpha, 0, 255).astype(np.uint8)
                if toggles.redactions and kind == "janedoe":
                    frame = redaction_bars(frame, rng, count=rng.randint(2,4))

            # UI overlay
            frame = alpha_over(frame, ui)

            # occasional random unrelated object crop popup (not the “flash events”)
            if rng.random() < (0.20 if kind not in ("bars","fatal") else 0.10):
                popup = make_popup_from_image(rng, rng.choice(loaded_np))
                if rng.random() < 0.25:
                    popup = 255 - popup
                frame = stamp_popup(frame, rng, popup)

            # scheduled flash events (max 3)
            for a, b, img in flash_events:
                if a <= frame_idx < b:
                    flash = img.copy()
                    flash = noise(flash, 35)
                    flash = chroma_bleed(flash, amt=5)
                    flash = hslice_glitch(flash, rng, bands=5, maxshift=72)
                    frame = flash
                    break

            # VHS / glitch stack
            if toggles.vhs:
                frame = chroma_bleed(frame, amt=3 if rng.random()<0.8 else 5)
                if rng.random() < 0.35:
                    frame = scanlines(frame, 0.84)
                frame = vignette(frame, W, H)
                frame = noise(frame, 18)

            if toggles.glitch and rng.random() < 0.20:
                frame = hslice_glitch(frame, rng, bands=3, maxshift=62)

            # transitions: heavier at edges
            if toggles.glitch and (fi < 3 or fi > nF-4):
                frame = hslice_glitch(frame, rng, bands=4, maxshift=76)
                if rng.random() < 0.40:
                    frame = 255 - frame

            if toggles.redactions and rng.random() < (0.12 if kind in ("protocol","janedoe") else 0.05):
                frame = redaction_bars(frame, rng, count=1)

            if toggles.timecode:
                frame = timecode_overlay(frame, FPS, frame_idx)

            writer.append_data(frame)
            frame_idx += 1

    writer.close()
    return frame_idx, popups_used

def render_audio(
    out_wav: Path,
    slides: List[dict],
    spec: VideoSpec,
    rng: random.Random,
    local_music_paths: List[Path],
    tts_enabled: bool = True
) -> float:
    """
    Creates WAV and returns duration seconds.
    """
    total_dur = len(slides) * spec.slide_seconds
    sr = SR = 44100
    total_samples = int(SR * total_dur)

    bed = gen_vhs_bed(rng, SR, total_dur)
    audio = bed.copy()

    # mix local music (optional)
    for mp in local_music_paths[:2]:
        try:
            sr2, x = load_wav_mono(mp)
            if sr2 != SR:
                # quick resample
                t1 = np.linspace(0, 1, len(x), False)
                t2 = np.linspace(0, 1, int(len(x)*SR/sr2), False)
                x = np.interp(t2, t1, x).astype(np.float32)
            x = x[:total_samples]
            audio[:len(x)] += x * 0.12
        except Exception:
            pass

    # TTS per slide
    if tts_enabled:
        tts_dir = out_wav.parent / "tts"
        ensure_dir(tts_dir)

        voices = ["en-us", "en", "en-scottish", "en-westindies"]
        for si, s in enumerate(slides):
            # narration: “normal but wrong” corporate tone + theme key / keyword fragments
            key = spec.theme
            kw = spec.theme_keywords
            snippet = " ".join([w for w in kw[:3]]) if kw else "signal memory anchor"
            base = s["title"].lower()

            # voice params (varied)
            voice = rng.choice(voices)
            speed = rng.randint(140, 168)
            pitch = rng.randint(22, 45)
            amp   = rng.randint(150, 190)

            # content: more story-like
            if s["kind"] in ("normal",):
                txt = f"{s['title']}. {rng.choice(['Please follow along.','This training improves wellbeing.','Maintain a calm posture.'])} {rng.choice(s['body_lines'][:2] or ['Continue normally.'])}"
            elif s["kind"] in ("protocol", "facefull"):
                txt = f"{s['title']}. {rng.choice(['Do not stare.','Do not describe it.','Look away immediately.'])} Theme key: {key}. Anchor words: {snippet}."
            elif s["kind"] == "janedoe":
                txt = f"Jane Doe intermission. Identity redacted. If you recognize the face, you may lose a minute. Do not repeat the key."
            elif s["kind"] == "fatal":
                txt = f"Fatal error. Tracking lost. Memory leak detected. Escape is disabled. Do not rewind."
            else:
                txt = f"{s['title']}. Do not replay this tape."

            wav_path = tts_dir / f"tts_{si:02d}.wav"
            espeak_tts(txt, wav_path, voice=voice, speed=speed, pitch=pitch, amp=amp)

            sr2, v = load_wav_mono(wav_path)
            if sr2 != SR:
                t1 = np.linspace(0, 1, len(v), False)
                t2 = np.linspace(0, 1, int(len(v)*SR/sr2), False)
                v = np.interp(t2, t1, v).astype(np.float32)

            # voice FX: bitcrush + light metallic ring
            v = np.tanh(v * 1.8) * 0.72
            v = bitcrush(v, factor=rng.choice([4,5,6]))
            ring = np.sin(2*np.pi*220*np.linspace(0, len(v)/SR, len(v), False).astype(np.float32))
            v = (0.82*v + 0.18*v*ring).astype(np.float32)
            v += (np.random.uniform(-1,1,len(v)).astype(np.float32) * 0.006)

            mix_in(audio, v, SR, start_s=si*spec.slide_seconds + 0.25, gain=0.95)

    # normalize
    audio /= (np.max(np.abs(audio)) + 1e-6)
    write_wav(str(out_wav), SR, (audio * 32767).astype(np.int16))
    return total_dur

def mux_av(video_in: Path, audio_in: Path, out_mp4: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_mp4)
    ], check=True, quiet=True)


# -----------------------------
# CLI / Config
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--seed", default=None, help="Override seed (int). If not set, uses config/env/random.")
    ap.add_argument("--no-web", action="store_true", help="Disable web scraping (uses local images only).")
    ap.add_argument("--no-tts", action="store_true", help="Disable TTS.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = Path(".").resolve()
    cfg_path = Path(args.config)

    if not cfg_path.exists():
        die(f"Missing config: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # ---- FIX for ${SEED} ----
    seed_raw = args.seed if args.seed is not None else cfg_get(cfg, "seed", "1337")
    seed_raw = str(seed_raw).strip()

    # if it's still a placeholder and env didn't resolve, fall back
    if seed_raw.startswith("${") and seed_raw.endswith("}"):
        seed_raw = "1337"

    # allow non-int seeds by hashing (so "nightmare" is ok)
    try:
        seed = int(seed_raw)
    except ValueError:
        seed = int(sha1(seed_raw)[:8], 16)

    rng = random.Random(seed)

    # spec
    W = int(cfg_get(cfg, "width", 640))
    H = int(cfg_get(cfg, "height", 480))
    FPS = int(cfg_get(cfg, "fps", 15))
    slide_seconds = float(cfg_get(cfg, "slide_seconds", 3.2))
    popups_max = int(cfg_get(cfg, "popup_flashes_max", 3))
    popup_seconds = float(cfg_get(cfg, "popup_flash_seconds", 0.5))
    vibe = str(cfg_get(cfg, "vibe", "vhs_found_tape"))

    # toggles (your “control bar” == config toggles)
    toggles_cfg = cfg.get("toggles", {}) or {}
    toggles = Toggles(
        vhs=bool(cfg_get(toggles_cfg, "vhs", True)),
        glitch=bool(cfg_get(toggles_cfg, "glitch", True)),
        redactions=bool(cfg_get(toggles_cfg, "redactions", True)),
        timecode=bool(cfg_get(toggles_cfg, "timecode", True)),
        ppt90s=bool(cfg_get(toggles_cfg, "ppt90s", True)),
        faces=bool(cfg_get(toggles_cfg, "faces", True)),
    )

    # user-chosen words to encrypt / inject
    inject_words = cfg.get("inject_words", []) or []
    encrypt_words = cfg.get("encrypt_words", []) or []
    encrypt_shift = int(cfg_get(cfg, "encrypt_shift", (seed % 17) + 3))

    # assets
    local_images = find_local_images(repo_root)
    local_music = []
    music_dir = repo_root / "assets" / "music"
    if music_dir.exists():
        local_music = list(music_dir.glob("*.wav"))

    # web scraping
    theme_title = "UNKNOWN"
    extract = ""
    keywords: List[str] = []

    use_web = (not args.no_web) and bool(cfg_get(cfg, "web", True))
    scraped_images: List[Path] = []

    if use_web:
        try:
            theme_title, extract, keywords = fetch_text_theme(rng)
        except Exception as e:
            print(f"[pptex] Web text scrape failed, continuing: {e}", file=sys.stderr)

        cache_dir = repo_root / ".cache" / "scraped"
        ensure_dir(cache_dir)

        try:
            scraped_images = scrape_images_into(cache_dir, rng, count=int(cfg_get(cfg, "scrape_images_count", 10)))
        except Exception as e:
            print(f"[pptex] Web image scrape failed, continuing: {e}", file=sys.stderr)

    # build image pools
    all_images = (local_images + scraped_images)
    if not all_images:
        # last resort: generate a dummy image so it still runs
        dummy = repo_root / ".cache" / "dummy.jpg"
        ensure_dir(dummy.parent)
        Image.fromarray(make_gradient_bg(rng, W, H)).save(dummy)
        all_images = [dummy]

    faces_pool = choose_faces_pool(all_images, rng)

    # theme keywords enrichment: inject user words
    if inject_words:
        keywords = (keywords + [w.lower() for w in inject_words])
    keywords = list(dict.fromkeys([k for k in keywords if k]))[:10]
    rng.shuffle(keywords)

    # encrypted words for easter eggs
    enc = []
    for w in encrypt_words:
        enc.append(encrypt_word(str(w), encrypt_shift))
    if enc:
        keywords = keywords + enc

    spec = VideoSpec(
        width=W, height=H, fps=FPS,
        slide_seconds=slide_seconds,
        popups_max=popups_max,
        popup_seconds=popup_seconds,
        seed=seed,
        theme=theme_title,
        theme_keywords=keywords,
        vibe=vibe
    )

    # story build
    if not extract:
        extract = "This training deck is incomplete. Tracking is unstable. Do not attempt recognition."

    slides = build_story_slides(rng, spec, extract, local_images)

    # output paths
    out_mp4 = Path(args.out).resolve()
    out_dir = out_mp4.parent
    ensure_dir(out_dir)

    tmp_dir = out_dir / ".pptex_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    silent_mp4 = tmp_dir / "silent.mp4"
    audio_wav = tmp_dir / "audio.wav"

    # render
    print(f"[pptex] seed={seed} theme='{spec.theme}' keywords={spec.theme_keywords[:6]}")
    total_frames, pop_used = render_video(
        silent_mp4,
        slides,
        images_pool=all_images,
        faces_pool=faces_pool,
        spec=spec,
        toggles=toggles,
        rng=rng
    )
    dur_s = total_frames / spec.fps
    render_audio(audio_wav, slides, spec, rng, local_music, tts_enabled=(not args.no_tts))

    mux_av(silent_mp4, audio_wav, out_mp4)

    # write a small metadata file
    meta = {
        "seed": seed,
        "theme": spec.theme,
        "keywords": spec.theme_keywords,
        "duration_s": dur_s,
        "popups_used": pop_used,
        "slides": [{"kind": s["kind"], "title": s["title"]} for s in slides],
        "timestamp": now_stamp(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[pptex] OK -> {out_mp4}")

if __name__ == "__main__":
    main()
