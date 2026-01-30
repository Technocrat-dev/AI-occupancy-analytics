# AI Chair Occupancy Analytics

Real-time chair occupancy detection and analytics using YOLOv11 and DeepSort tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## Features

- **Real-time Detection**: YOLOv11 object detection for chairs and people
- **Person Tracking**: DeepSort re-identification across frames
- **Occupancy Analytics**: Per-chair and per-person usage statistics
- **Multi-Camera Support**: Unified analytics across overlapping camera views
- **Web Dashboard**: Modern glassmorphism UI with Chart.js visualizations
- **REST API**: Full API for video processing and data retrieval
- **Docker Ready**: GPU-accelerated container deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Open http://localhost:8000
```

### Docker Deployment

```bash
# Build and run with GPU support
docker-compose up --build

# Or build manually
docker build -t chair-analytics .
docker run -p 8000:8000 --gpus all chair-analytics
```

## API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/` | Web dashboard |
| `POST` | `/process-video` | Upload and analyze video |
| `GET` | `/api/results/{file_id}` | Get analysis results |
| `GET` | `/api/history` | List all analyses |
| `DELETE` | `/api/history/{file_id}` | Delete analysis |
| `GET` | `/download/{file_id}` | Download processed video |

### Multi-Camera API

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/multi-camera/setup` | Configure cameras |
| `GET` | `/api/multi-camera/setup` | Get current config |
| `DELETE` | `/api/multi-camera/setup` | Clear config |
| `GET` | `/api/multi-camera/stats` | Unified statistics |

## Configuration

Environment variables (for Docker):

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DATABASE_URL` | `sqlite:///./analysis.db` | Database connection |
| `UPLOAD_DIR` | `uploads` | Upload directory |
| `OUTPUT_DIR` | `outputs` | Output directory |
| `MODEL_PATH` | `yolo11n.pt` | YOLO model path |
| `MAX_FILE_SIZE` | `104857600` | Max upload (100MB) |
| `MAX_WORKERS` | `2` | Processing threads |

## Processing Parameters

| Parameter | Range | Default | Description |
|:----------|:------|:--------|:------------|
| `proximity_threshold` | 10-500 | 80 | Person-chair distance (px) |
| `occupancy_frames_threshold` | 1-60 | 5 | Frames to confirm occupancy |
| `motion_blur_threshold` | 10-500 | 100 | Blur detection sensitivity |

## Output Analytics

Each processed video generates:

- **Processed Video**: Annotated with bounding boxes and occupancy status
- **Occupancy Chart**: Frame-by-frame utilization over time
- **Interaction Ledger**: Every person-chair session with timestamps
- **Per-Person Metrics**: Total time, chairs used, session count
- **Per-Chair Metrics**: Total usage time, unique users
- **Peak/Low Activity Windows**: Busiest and quietest periods

## Cleanup Utility

Manage disk space with the cleanup script:

```bash
# Preview what would be deleted (dry run)
python cleanup.py --dry-run

# Delete files older than 7 days
python cleanup.py --retention-days 7

# Keep directory under 5GB
python cleanup.py --max-size-mb 5000
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only config tests (no ML deps needed)
pytest tests/test_occupancy.py::TestConfigValidation -v

# Run API tests
pytest tests/test_api.py -v
```

## Project Structure

```
├── app.py              # FastAPI application
├── config.py           # Environment configuration
├── database.py         # SQLAlchemy setup
├── process_video.py    # ML pipeline (YOLO + DeepSort)
├── cleanup.py          # Output cleanup utility
├── index.html          # Web dashboard
├── api/
│   └── multi_camera.py # Multi-camera endpoints
├── myutils/
│   ├── detector.py     # YOLO wrapper
│   ├── tracker.py      # DeepSort format converter
│   └── panel.py        # Video overlay panel
├── tests/
│   ├── test_api.py     # API tests
│   └── test_occupancy.py # Unit tests
└── docker-compose.yml  # Container orchestration
```

## License

MIT License
