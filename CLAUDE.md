# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multimodal OCR project using Qwen2.5-Omni-7B to convert financial statement images into semantically correct HTML. The main application processes images through a multimodal LLM that can handle text, audio, image, and video inputs.

## Development Setup

This project uses `uv` for Python package management with Python 3.10+.

**Install dependencies:**
```bash
uv sync
```

**Activate virtual environment:**
```bash
source .venv/bin/activate
```

**Run the main OCR script:**
```bash
python main.py
```

**Lint code:**
```bash
ruff check .
```

**Format code:**
```bash
ruff format .
```

## Architecture

### Core Components

**main.py** - The primary entry point containing the complete OCR pipeline:
1. Loads Qwen2.5-Omni-7B model using HuggingFace Transformers
2. Initializes processor for multimodal inputs
3. Constructs conversation with system prompt and user image
4. Processes image through the model
5. Generates HTML output from financial statement images

### Key Dependencies

- **transformers**: HuggingFace library for the Qwen2.5-Omni model
- **qwen-omni-utils**: Utilities for processing multimodal inputs (process_mm_info)
- **torch/torchvision**: Deep learning framework and vision utilities
- **accelerate**: For efficient model loading across devices
- **av**: Audio/video processing support
- **soundfile**: Audio file I/O

### Model Configuration

The model uses:
- Auto device mapping (`device_map="auto"`)
- Automatic dtype selection (`torch_dtype="auto"`)
- Multimodal conversation format with system/user roles
- Content types: text, image, audio, video

### Input Processing

Images are processed through a conversation structure where:
- System message defines the AI assistant persona
- First user message contains the OCR prompt
- Second user message contains the image reference

The `USE_AUDIO_IN_VIDEO` flag controls whether audio tracks in video inputs are processed (default: False).

### Output

The model generates:
- `text_ids`: Token IDs for generated text
- `audio`: Generated audio output (if applicable)

Only newly generated tokens are decoded (slicing from `input_length` onward) to get the final HTML output.

## Working with Images

Place test images in the project root. The current example uses `test.png`. Update the image path in main.py:143 to process different financial statement images.
