# pptex — VHS / 90s PPT Found-Tape Horror Generator (GitHub-only)

This repo generates an always-different analogue horror video:
- 90s PowerPoint aesthetics + VHS artifacts + glitches + redactions + timecode
- Mixes normal wellness slides with scary protocol/JaneDoe/fatal-error slides
- Scrapes text + images from Wikipedia/Wikimedia each run
- Adds up to 3 short flash popups (~0.5s) for “wrongness”
- Creates noisy music bed + espeak TTS with varied voice params
- Runs entirely in GitHub Actions and uploads an `.mp4` artifact

## Quick start
1. Put any images you want used in slides here:
   `assets/images/` (jpg/png/webp)

2. (Optional) Put any `.wav` files you want mixed into the noise bed:
   `assets/music/`

3. Go to **Actions → Generate Horror PPT VHS Video → Run workflow**  
   Optionally set a `seed`.

4. Download `out.mp4` from the workflow artifacts.

## Controls (“control bar”)
Edit `config.yaml`:
- `toggles.vhs` / `toggles.glitch` / `toggles.redactions` / `toggles.timecode` etc.
- `inject_words` are inserted into the deck theme vocabulary
- `encrypt_words` become easter-egg ciphers on some slides
- `popup_flashes_max` and `popup_flash_seconds` control the jumpscare flashes

## Local run (optional)
```bash
pip install -r requirements.txt
sudo apt-get install ffmpeg espeak
python generate.py --config config.yaml --out out.mp4
