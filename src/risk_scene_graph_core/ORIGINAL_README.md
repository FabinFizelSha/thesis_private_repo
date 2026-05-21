# Risk-Annotated-Scene-Graph
[Thesis Document](https://github.com/CoollPro/Risk-Annotated-Scene-Graph/blob/main/MasterThesis.pdf)

Risk-Annotated-Scene-Graph is a Python pipeline for building and updating a risk-aware scene graph from RGB-D data.

It combines:
- a VLM-based scene-graph correction step,
- optional SAM-based object proposal generation,
- optional CLIP + ChromaDB visual retrieval ("Visual RAG"),
- and a FastAPI service for online ingestion and graph access.

The project can run in two main modes:
- API mode: send frames to `/frame` and read results from `/scenegraph`.
- Dataset mode: feed a local dataset through the queue for batch-like processing.

## Features

- FastAPI endpoints for frame ingestion and status monitoring
- Multiprocess architecture (API worker + queue consumer + optional learning worker)
- Scene graph persistence and visualization output (optional)
- Offline mode (`--vlm-offline`) for local testing without external VLM calls
- Small-VLM mode (`--small-vlm`) to simplify VLM behavior:
  - strips `gt`, `distance`, and `confidence` from VLM input,
  - adds a stricter prompt rule to limit new nodes,
  - filters duplicate nodes after inference
- Optional SAM segmentation + Visual RAG loop for object labeling and incremental memory

## Project Structure

- `main.py`: process orchestration and CLI
- `scripts/Worker.py`: API worker, queue worker, learning worker
- `scripts/VlmHelper.py`: VLM prompt construction + inference + response parsing
- `scripts/SceneGraph.py`: core graph handling and updates
- `scripts/SamSegmenter.py`: SAM-based mask generation
- `scripts/VisualRAP.py`: CLIP embedding + ChromaDB retrieval
- `scripts/LoadDataset.py`: dataset loader (rgb/depth/groundtruth)
- `config/parameters.py`: camera, dataset, and processing config
- `testing/`: helper clients for API tests
- `start.ps1`, `start.sh`: startup examples

## Requirements

- Python 3.10+ recommended
- Optional GPU for faster SAM / CLIP inference
- Optional external VLM endpoint compatible with OpenAI Chat Completions API

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to run without Visual RAG / SAM-related heavy deps, use:

```bash
pip install -r requierement_without_rap.txt
```

## Environment Variables

- `OPENAI_BASE_URL` (optional)
  - Default: `http://0.0.0.0:8000/v1`
  - Used by `VLMHelper` for OpenAI-compatible API calls
- `MODEL` (optional)
  - If not set, the first model from the endpoint model list is used

Example:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export MODEL="your-model-name"
```

PowerShell:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
$env:MODEL = "your-model-name"
```

## Quick Start

### 1) Start service (API mode)

```bash
python main.py
```

Default host/port in code:
- host: `0.0.0.0`
- port: `8005`

API docs:
- `http://localhost:8005/docs`

### 2) Send a sample frame

```bash
python testing/SendSample.py --host 127.0.0.1 --port 8005
```

### 3) Stream full dataset to API

```bash
python testing/DatasetClient.py --host 127.0.0.1 --port 8005
```

## Running Modes and CLI Flags

Main CLI (`python main.py`):

- `--host`: API host (default `0.0.0.0`)
- `--port`: API port (default `8005`)
- `--usedataset`: enables dataset feeder process
- `--path`: dataset directory override (must contain `rgb/`, `depth/`, `groundtruth.txt`)
- `--vlm-offline`: use dummy VLM helper (no external VLM call)
- `--no-sam-rap`: disable SAM + Visual RAG path
- `--save-output`: save intermediate/final output files under `data/tmp`
- `--small-vlm`: enable simplified VLM flow (stripping, stricter prompt, duplicate filtering)

Examples:

```bash
# API only, online VLM
python main.py

# Dataset feeder + online VLM
python main.py --usedataset

# Offline VLM for local functional tests
python main.py --usedataset --vlm-offline

# Disable SAM/RAG
python main.py --usedataset --no-sam-rap

# Save files to data/tmp
python main.py --usedataset --save-output

# Small-VLM mode
python main.py --usedataset --small-vlm
```

## API Endpoints

Base URL: `http://<host>:<port>`

- `GET /health`
  - health check

- `POST /frame`
  - enqueue one frame for processing
  - payload fields:
    - `timestamp` (string)
    - `rgb` (base64 encoded JPG/PNG)
    - `depth` (base64 encoded PNG, expected aligned to RGB)
    - `tx` (translation `[x,y,z]`)
    - `rotM` (3x3 rotation matrix)

- `GET /scenegraph`
  - returns latest graph (`nodes`, `edges`)

- `POST /robot/info`
  - queues robot metadata to be inserted on next processed frame

- `GET /robot`
  - returns current robot node (if present)

- `GET /queue/status`
  - queue stats (`queued`, `processing`, `completed`, `failed`)

- `GET /frame/{frame_id}`
  - status/details for a specific frame

## Dataset Format

Expected dataset directory:

```text
dataset_dir/
  rgb/
    <timestamp>.png
    ...
  depth/
    <timestamp>.png
    ...
  groundtruth.txt
```

`groundtruth.txt` format (space-separated):

```text
timestamp tx ty tz qx qy qz qw
```

Where quaternion is converted to a 3x3 rotation matrix internally.

Default dataset path (from config):
- `data/dataset_test`

## Output Files

When `--save-output` is enabled, output is written under `data/tmp`:

- `inVLM/`: VLM input graph dumps
- `outVLM/`: VLM output graph dumps
- `completeGraph/`: full graph snapshots as JSON
- `outPNG/`: graph visualization images
- `masks/`: all SAM masks visualization
- `masks_detected/`: masks for recognized objects

## Visual RAG Notes

Visual RAG uses:
- CLIP model (`openai/clip-vit-base-patch32`)
- ChromaDB collection (`visual_rag`)
- storage path: `visual_memory/`

`VisualRAP` can auto-start a local Chroma server (`chroma run`) on port `8001` if not already running.

To preload labels/images into memory:

```bash
python LoadDataToRAP.py --path training_data
```

Recommended: 5-15 images per class.

## Configuration

Main static config is in `config/parameters.py`, including:
- camera intrinsics / distortion
- dataset paths and depth scaling
- image size validation
- SAM thresholds
- visual retrieval distance threshold
- VLM input format (`json`, `yaml`, `toon`)

Adjust these values for your sensor and environment.

## Troubleshooting

- API not reachable:
  - Check host/port and open `http://localhost:<port>/docs`
  - Ensure process is running (`python main.py`)

- VLM errors / empty outputs:
  - Verify `OPENAI_BASE_URL` and `MODEL`
  - Try `--vlm-offline` to isolate network/model issues

- SAM startup slow or failing:
  - First run may download the SAM checkpoint (~2.4 GB)
  - Ensure enough disk space and stable network

- ChromaDB issues:
  - Make sure `chromadb[cli]` is installed
  - Check if port `8001` is already in use

- Unexpected image size errors:
  - Verify incoming RGB and depth dimensions match `img_size` in config

## Notes

- `start.ps1` and `start.sh` provide quick launch templates.
- If comments in scripts mention a different default port, trust `main.py` as source of truth (currently `8005`).

