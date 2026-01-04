# SAM Audio CLI

Memory-optimized command-line tool for text-prompted audio separation using [SAM Audio](https://huggingface.co/facebook/sam-audio-small). Extract or remove specific sounds from audio files using natural language descriptions.

## Features

- Text-prompted audio separation (e.g., "vocals", "drums", "dog barking")
- Memory optimized: ~4-6GB VRAM (down from ~11GB)
- Automatic chunked processing for long audio files
- Interactive and CLI modes
- GPU acceleration with CPU fallback

## Installation

```bash
git clone https://github.com/yourusername/sam-audio-local.git
cd sam-audio-local
pip install torch torchaudio python-dotenv huggingface-hub
pip install git+https://github.com/facebookresearch/sam-audio
```
Add a .env file with the following:
HF_TOKEN=your_token_here

You will ned to request and be granted access to Sam Audio on Huggingface: https://huggingface.co/facebook/sam-audio-base

### Protobuf Fix (Required)

SAM Audio has protobuf compatibility issues. Fix with:

```bash
pip install --upgrade protobuf

# Tweak the directory as needed
cp ../usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py ./

# Downgrade protobuf
pip install protobuf==3.19.4

cp ./builder.py ../usr/local/lib/python3.11/dist-packages/google/protobuf/internal/

```

## Usage

**Interactive mode:**
```bash
python main.py
```

**CLI mode:**
```bash
# Extract vocals
python main.py --input song.mp3 --prompt "vocals" --mode extract

# Remove background noise
python main.py --input audio.wav --prompt "background noise" --mode remove
```

## Options

- `--input`, `-i`: Input audio file
- `--prompt`, `-p`: Sound description to extract/remove
- `--mode`, `-m`: `extract` or `remove` (default: extract)
- `--model`: `small`, `base`, or `large` (default: small)
- `--output`, `-o`: Output directory (default: output)
- `--float32`: Use float32 precision (better quality, more VRAM)
- `--chunk-duration`: Chunk size in seconds (default: 25)

## Output Files

**Extract mode:**
- `{filename}_extracted.wav` - Isolated sound
- `{filename}_residual.wav` - Everything else

**Remove mode:**
- `{filename}_cleaned.wav` - Audio with sound removed
- `{filename}_removed.wav` - Isolated removed sound

## Acknowledgments

Built with Claude AI, referencing the [AudioGhost AI](https://github.com/0x0funky/audioghost-ai/) open source web UI project.

Based on [SAM Audio](https://huggingface.co/facebook/sam-audio-small) by Meta AI.