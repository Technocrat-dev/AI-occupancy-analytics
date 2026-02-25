




# AI Chair Occupancy Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time chair occupancy detection and analytics system powered by YOLOv11 and DeepSort tracking. Designed for workspace optimization, facility management, and occupancy monitoring.**

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Screenshots](#-screenshots)
- [Architecture](#%EF%B8%8F-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [Advanced Features](#-advanced-features)
- [Testing](#-testing)
- [Cleanup & Maintenance](#-cleanup--maintenance)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

The **AI Chair Occupancy Analytics Platform** is an enterprise-grade solution for monitoring and analyzing chair usage patterns in real-time. It utilizes state-of-the-art computer vision and deep learning techniques to:

- **Detect and track** chairs and people in video streams
- **Analyze occupancy patterns** across time and space
- **Generate insights** on space utilization and peak usage periods
- **Support multi-camera** setups with intelligent deduplication
- **Provide real-time** WebSocket streams and RESTful APIs

### Use Cases

- **Office Space Management**: Optimize desk allocation and workspace planning
- **Library Analytics**: Monitor study room and seating availability
- **Healthcare Facilities**: Track waiting room occupancy
- **Educational Institutions**: Analyze classroom and cafeteria usage
- **Retail & Hospitality**: Monitor seating areas and customer flow

---

## Key Features

### Core Capabilities

- **YOLOv11 Object Detection**: State-of-the-art real-time detection of chairs and people
- **DeepSort Tracking**: Advanced person re-identification across frames
- **Multi-Camera Support**: Unified analytics with automatic chair deduplication
- **Real-Time Streaming**: WebSocket support for live video and statistics
- **RESTful API**: Complete API for integration with external systems
- **Web Dashboard**: Modern glassmorphism UI with interactive visualizations
- **Batch Processing**: Asynchronous video analysis with progress tracking
- **Database Persistence**: SQLite storage for historical analytics

### Advanced Features

- **Smart Tracking**
  - Chair position interpolation during occlusion
  - Motion blur detection and adaptive enhancement
  - Reflection detection and filtering
  - Person-chair zone-based occupancy detection

- **Comprehensive Analytics**
  - Frame-by-frame occupancy rates
  - Per-person usage metrics (total time, chairs used, session count)
  - Per-chair metrics (total usage, unique users)
  - Peak and low activity window detection
  - Interaction ledger with complete session history

- **Multi-Camera Intelligence**
  - Camera priority zones for overlapping views
  - Automatic duplicate chair resolution
  - Conflict-free unified occupancy counting
  - Configurable overlap regions

- **Production Ready**
  - Docker containerization with GPU support
  - Environment-based configuration
  - Automated cleanup utilities
  - Comprehensive test suite
  - Health check endpoints

---

## Screenshots

### Dashboard Overview

<img width="2047" height="1099" alt="image" src="https://github.com/user-attachments/assets/f85ec0aa-7ac6-4134-a791-a70fe7a029d9" />

### Analytics Visualization

<img width="1839" height="662" alt="image" src="https://github.com/user-attachments/assets/2fca25d6-7141-40ad-8488-1facd03367bf" />
<img width="1807" height="1042" alt="image" src="https://github.com/user-attachments/assets/046c3d06-c30c-42f8-804a-844750ac0699" />
<img width="1896" height="1001" alt="image" src="https://github.com/user-attachments/assets/168d4c51-0ae4-4dcd-9878-74affe110dff" />
<img width="1788" height="835" alt="image" src="https://github.com/user-attachments/assets/8c3b5ffb-cac9-40ee-9ac1-f15d2bf4daba" />

### Multi-Camera Setup

<img width="2026" height="1092" alt="image" src="https://github.com/user-attachments/assets/f4b5af13-afae-4b83-8ac2-d69b3407e627" />

### Live Streaming

<img width="2026" height="1093" alt="image" src="https://github.com/user-attachments/assets/c8950eb6-5522-4903-8542-ffb268c9a341" />

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Dashboard<br/>HTML/CSS/JS]
        API_CLIENT[External API Clients]
        WS_CLIENT[WebSocket Clients]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI Application<br/>app.py]
        WS[WebSocket Handler<br/>websocket_handler.py]
        MULTI[Multi-Camera API<br/>api/multi_camera.py]
    end
    
    subgraph "Service Layer"
        PROCESSOR[Frame Processor<br/>services/processor.py]
        STREAM[Stream Generator<br/>services/stream.py]
        ANALYTICS[Analytics Service<br/>services/analytics.py]
    end
    
    subgraph "ML Pipeline"
        DETECTOR[YOLO Detector<br/>myutils/detector.py]
        TRACKER[DeepSort Tracker<br/>myutils/tracker.py]
        PROCESS[Video Processor<br/>process_video.py]
    end
    
    subgraph "Data Layer"
        DB[(SQLite Database<br/>analysis.db)]
        FILES[File Storage<br/>uploads/outputs/]
        CONFIG[Configuration<br/>config.py]
    end
    
    WEB --> FASTAPI
    API_CLIENT --> FASTAPI
    WS_CLIENT --> WS
    
    FASTAPI --> PROCESSOR
    FASTAPI --> MULTI
    FASTAPI --> DB
    WS --> PROCESSOR
    
    MULTI --> PROCESS
    PROCESSOR --> PROCESS
    STREAM --> PROCESS
    
    PROCESS --> DETECTOR
    PROCESS --> TRACKER
    PROCESS --> ANALYTICS
    
    PROCESS --> FILES
    ANALYTICS --> DB
    
    CONFIG -.-> FASTAPI
    CONFIG -.-> PROCESS
    
    style WEB fill:#e1f5ff
    style FASTAPI fill:#b3e5fc
    style PROCESS fill:#81d4fa
    style DETECTOR fill:#4fc3f7
    style DB fill:#fff59d
```

### Video Processing Pipeline

```mermaid
flowchart LR
    subgraph INPUT[Input Sources]
        VIDEO[Video File]
        WEBCAM[Webcam]
        RTSP[RTSP Stream]
    end
    
    subgraph DETECTION[Detection Phase]
        YOLO[YOLOv11 Model<br/>Chair & Person Detection]
        BLUR[Motion Blur<br/>Detection]
        REFLECT[Reflection<br/>Filtering]
    end
    
    subgraph TRACKING[Tracking Phase]
        DEEPSORT[DeepSort<br/>Person Re-ID]
        CHAIR_SIG[Chair Signature<br/>Matching]
        INTERPOLATE[Position<br/>Interpolation]
    end
    
    subgraph ANALYSIS[Analysis Phase]
        PROXIMITY[Proximity<br/>Calculation]
        OCCUPANCY[Occupancy<br/>Status Update]
        ZONES[Zone-based<br/>Detection]
    end
    
    subgraph OUTPUT[Output Generation]
        ANNOTATE[Frame<br/>Annotation]
        METRICS[Metrics<br/>Calculation]
        VIDEO_OUT[Processed Video]
        JSON[JSON Results]
    end
    
    INPUT --> DETECTION
    DETECTION --> TRACKING
    TRACKING --> ANALYSIS
    ANALYSIS --> OUTPUT
    
    BLUR -.filter.-> TRACKING
    REFLECT -.filter.-> TRACKING
    
    style INPUT fill:#e8f5e9
    style DETECTION fill:#fff9c4
    style TRACKING fill:#b3e5fc
    style ANALYSIS fill:#f8bbd0
    style OUTPUT fill:#d1c4e9
```

### Multi-Camera Architecture

```mermaid
graph TB
    subgraph CAMERAS[Camera Feeds]
        CAM1[Camera 1<br/>Priority: High]
        CAM2[Camera 2<br/>Priority: Medium]
        CAM3[Camera 3<br/>Priority: Low]
    end
    
    subgraph PROCESSING[Individual Processing]
        PROC1[Processor 1]
        PROC2[Processor 2]
        PROC3[Processor 3]
    end
    
    subgraph MANAGER[Multi-Camera Manager]
        ZONES[Zone Authority<br/>Checker]
        DEDUP[Chair<br/>Deduplication]
        CONFLICT[Conflict<br/>Resolution]
        UNIFIED[Unified<br/>Statistics]
    end
    
    subgraph STORAGE[Data Storage]
        SETUP_DB[(Setup Config<br/>Database)]
        RESULTS[Merged<br/>Results]
    end
    
    CAM1 --> PROC1
    CAM2 --> PROC2
    CAM3 --> PROC3
    
    PROC1 --> ZONES
    PROC2 --> ZONES
    PROC3 --> ZONES
    
    ZONES --> DEDUP
    DEDUP --> CONFLICT
    CONFLICT --> UNIFIED
    
    SETUP_DB -.config.-> ZONES
    UNIFIED --> RESULTS
    
    style CAM1 fill:#c8e6c9
    style CAM2 fill:#fff9c4
    style CAM3 fill:#ffccbc
    style MANAGER fill:#b3e5fc
    style RESULTS fill:#d1c4e9
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Processor
    participant YOLO
    participant DeepSort
    participant Database
    participant FileSystem
    
    Client->>FastAPI: Upload Video
    FastAPI->>FileSystem: Save to uploads/
    FastAPI->>Processor: process_video_for_api()
    
    loop For Each Frame
        Processor->>YOLO: Detect Objects
        YOLO-->>Processor: Chairs & People
        Processor->>DeepSort: Track People
        DeepSort-->>Processor: Person IDs
        Processor->>Processor: Calculate Occupancy
        Processor->>Client: WebSocket Progress Update
    end
    
    Processor->>FileSystem: Save Processed Video
    Processor->>FileSystem: Save JSON Results
    Processor-->>FastAPI: Processing Complete
    
    FastAPI->>Database: Save Summary
    Database-->>FastAPI: Confirmation
    FastAPI-->>Client: Success Response
```

### Component Interaction Diagram

```mermaid
graph LR
    subgraph Frontend
        HTML[index.html<br/>Dashboard UI]
    end
    
    subgraph Backend
        APP[app.py<br/>FastAPI Server]
        WS_H[websocket_handler.py<br/>Real-time Updates]
    end
    
    subgraph Services
        PROC_S[processor.py<br/>Frame Generator]
        STREAM_S[stream.py<br/>MJPEG Stream]
        ANAL_S[analytics.py<br/>Statistics]
    end
    
    subgraph Core
        PV[process_video.py<br/>Main Pipeline]
        MCM[MultiCameraManager<br/>Deduplication]
        COT[ChairOccupancyTracker<br/>Tracking Logic]
    end
    
    subgraph Utils
        DET[detector.py<br/>YOLO Wrapper]
        TRK[tracker.py<br/>DeepSort Utils]
        PANEL[panel.py<br/>Overlay Panel]
    end
    
    HTML <-->|REST/WS| APP
    HTML <-->|WebSocket| WS_H
    
    APP --> PROC_S
    APP --> STREAM_S
    WS_H --> PROC_S
    
    PROC_S --> PV
    STREAM_S --> PV
    ANAL_S --> PV
    
    PV --> MCM
    PV --> COT
    
    COT --> DET
    COT --> TRK
    PV --> PANEL
    
    style HTML fill:#e1f5ff
    style APP fill:#b3e5fc
    style PV fill:#81d4fa
    style COT fill:#4fc3f7
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** package manager
- **CUDA** (optional, for GPU acceleration)
- **Docker** (optional, for containerized deployment)

### Local Development (Without Docker)

```bash
# Clone the repository
git clone <repository-url>
cd chair_occupancy_project

# Create virtual environment
python -m venv AI_Chair_Occupancy_Analytics_env
source AI_Chair_Occupancy_Analytics_env/bin/activate  # On Windows: AI_Chair_Occupancy_Analytics_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open your browser
# Navigate to http://localhost:8000
```

The YOLO model (`yolo11m.pt`) will be automatically downloaded on first run if not present.

### Docker Deployment (Recommended for Production)

```bash
# With GPU support (NVIDIA)
docker-compose up --build

# Without GPU (CPU only)
docker-compose -f docker-compose.cpu.yml up --build

# Access the application
# Navigate to http://localhost:8000
```

---

## Installation

### Method 1: Standard Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd chair_occupancy_project
```

2. **Set up Python virtual environment**

```bash
python -m venv AI_Chair_Occupancy_Analytics_env

# Activate on Linux/Mac
source AI_Chair_Occupancy_Analytics_env/bin/activate

# Activate on Windows
AI_Chair_Occupancy_Analytics_env\Scripts\activate
```

3. **Install dependencies**

For systems with CUDA GPU:
```bash
pip install -r requirements.txt
```

For CPU-only systems:
```bash
# Edit requirements.txt to remove the --extra-index-url line
pip install -r requirements.txt
```

4. **Download YOLO model** (optional, auto-downloads on first run)

```bash
# The yolo11m.pt model is included in the repo
# Or download manually from Ultralytics
```

### Method 2: Docker Installation

1. **Prerequisites**: Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. **For GPU support**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

3. **Build and run**

```bash
# GPU-enabled deployment
docker-compose up --build -d

# CPU-only deployment
docker-compose -f docker-compose.cpu.yml up --build -d
```

---

## Usage

### Web Interface

1. **Access the dashboard** at `http://localhost:8000`
2. **Upload a video** using the file input
3. **Configure parameters** (optional):
   - Proximity Threshold (10-500 px)
   - Occupancy Frames Threshold (1-60 frames)
   - Motion Blur Threshold (10-500)
4. **Click "Process Video"** and monitor progress
5. **View results** including:
   - Annotated output video
   - Occupancy rate over time chart
   - Detailed analytics and interaction ledger

### Live Streaming

Access real-time occupancy detection from webcam or RTSP stream:

```bash
# View live stream endpoint
GET /api/stream/{camera_id}?source=0

# Example: Webcam stream
http://localhost:8000/api/stream/1?source=0

# Example: RTSP stream
http://localhost:8000/api/stream/1?source=rtsp://user:pass@192.168.1.100:554/stream
```

### Multi-Camera Setup

1. **Configure cameras** via API:

```bash
POST /api/multi-camera/setup
Content-Type: application/json

{
  "name": "Office Floor 1",
  "cameras": [
    {
      "camera_id": 1,
      "priority": 10,
      "zones": [
        {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}
      ],
      "video_path": "/path/to/camera1.mp4"
    },
    {
      "camera_id": 2,
      "priority": 5,
      "zones": [
        {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}
      ],
      "video_path": "/path/to/camera2.mp4"
    }
  ],
  "overlap_regions": [
    {
      "region": {"x1": 800, "y1": 400, "x2": 1200, "y2": 800},
      "camera_ids": [1, 2]
    }
  ]
}
```

2. **Process multi-camera setup**:

```bash
POST /api/multi-camera/process/{setup_id}
```

### Command Line Processing

For batch processing without the web interface:

```python
from process_video import process_video_for_api

settings = {
    'proximity_threshold': 80,
    'occupancy_frames_threshold': 5,
    'motion_blur_threshold': 100
}

results = process_video_for_api(
    input_path="input.mp4",
    output_path="output.mp4",
    settings=settings
)

print(f"Average occupancy: {results['average_occupancy_rate']:.2f}%")
```

---

## API Reference

### Core Endpoints

#### Upload and Process Video

```http
POST /process-video
Content-Type: multipart/form-data

file: <video_file>
proximity_threshold: 80 (optional)
occupancy_frames_threshold: 5 (optional)
motion_blur_threshold: 100 (optional)
```

**Response:**
```json
{
  "success": true,
  "message": "Video processed successfully",
  "file_id": "uuid-here",
  "output_video_url": "/outputs/uuid_output.mp4",
  "results_api_url": "/api/results/uuid",
  "processing_results": { /* full analytics */ }
}
```

#### Get Analysis Results

```http
GET /api/results/{file_id}
```

**Response:**
```json
{
  "average_occupancy_rate": 65.4,
  "max_occupied_chairs": 12,
  "total_chairs": 15,
  "interaction_ledger": [
    {
      "person_id": 1,
      "chair_id": 3,
      "start_frame": 120,
      "end_frame": 450,
      "duration_seconds": 11.0
    }
  ],
  "per_person_stats": { /* ... */ },
  "per_chair_stats": { /* ... */ },
  "peak_activity_window": { /* ... */ },
  "low_activity_window": { /* ... */ }
}
```

#### Get Analysis History

```http
GET /api/history?skip=0&limit=100
```

#### Delete Analysis

```http
DELETE /api/history/{file_id}
```

#### Download Processed Video

```http
GET /download/{file_id}
```

### Multi-Camera Endpoints

#### Configure Multi-Camera Setup

```http
POST /api/multi-camera/setup
Content-Type: application/json

{
  "name": "Setup Name",
  "cameras": [ /* camera configs */ ],
  "overlap_regions": [ /* overlap definitions */ ]
}
```

#### List All Setups

```http
GET /api/multi-camera/setups
```

#### Get Specific Setup

```http
GET /api/multi-camera/setup/{setup_id}
```

#### Delete Setup

```http
DELETE /api/multi-camera/setup/{setup_id}
```

#### Process Multi-Camera

```http
POST /api/multi-camera/process/{setup_id}
```

### Streaming Endpoints

#### Live Video Stream

```http
GET /api/stream/{camera_id}?source=<source>

source: Camera index (0, 1, ...) or RTSP URL
```

Returns MJPEG stream suitable for `<img>` tag.

#### Stream Status

```http
GET /api/stream-status
```

#### List Recordings

```http
GET /api/recordings
```

#### Download Recording

```http
GET /api/recordings/{filename}
```

### WebSocket Endpoints

#### Progress Updates

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/progress/${fileId}`);
ws.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    console.log(`Progress: ${progress.percent}%`);
};
```

#### Live Statistics

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/live-stats/${cameraId}`);
ws.onmessage = (event) => {
    const stats = JSON.parse(event.data);
    console.log(`Occupancy: ${stats.occupancy_rate}%`);
};
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1.0"
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=sqlite:///./analysis.db

# Directories
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs

# YOLO Model
MODEL_PATH=yolo11m.pt

# Processing defaults
DEFAULT_PROXIMITY_THRESHOLD=80
DEFAULT_OCCUPANCY_FRAMES=5
DEFAULT_MOTION_BLUR_THRESHOLD=100

# Limits
MAX_FILE_SIZE=104857600  # 100MB in bytes
MAX_WORKERS=2

# API Security
API_KEY=your-secret-api-key-here
```

### Configuration Parameters

#### Processing Parameters

| Parameter | Range | Default | Description |
|:----------|:------|:--------|:------------|
| `proximity_threshold` | 10-500 | 80 | Maximum distance (pixels) between person and chair to consider occupied |
| `occupancy_frames_threshold` | 1-60 | 5 | Number of consecutive frames required to confirm occupancy |
| `motion_blur_threshold` | 10-500 | 100 | Laplacian variance threshold for blur detection (lower = more sensitive) |

#### System Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DATABASE_URL` | `sqlite:///./analysis.db` | SQLAlchemy database connection string |
| `UPLOAD_DIR` | `uploads` | Directory for temporary uploaded files |
| `OUTPUT_DIR` | `outputs` | Directory for processed videos and results |
| `MODEL_PATH` | `yolo11m.pt` | Path to YOLO model weights |
| `MAX_FILE_SIZE` | `104857600` | Maximum upload size in bytes (100MB) |
| `MAX_WORKERS` | `2` | ThreadPoolExecutor workers for async processing |
| `API_KEY` | `dev-key-change-me` | API key for protected endpoints (change in production!) |

### Docker Configuration

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - API_KEY=${API_KEY:-your-production-key}
  - MAX_WORKERS=4
  - DEFAULT_PROXIMITY_THRESHOLD=100

volumes:
  - ./uploads:/app/uploads
  - ./outputs:/app/outputs
  - ./analysis.db:/app/analysis.db

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Project Structure

```
chair_occupancy_project/
│
├── app.py                      # Main FastAPI application and API endpoints
├── config.py                   # Configuration management and environment variables
├── database.py                 # SQLAlchemy database setup and session management
├── schemas.py                  # Pydantic schemas for API validation
├── sql_models.py               # SQLAlchemy ORM models for database tables
├── process_video.py            # Core ML pipeline (YOLO + DeepSort + Analytics)
├── websocket_handler.py        # WebSocket endpoints for real-time updates
├── cleanup.py                  # Utility script for managing disk space
├── index.html                  # Web dashboard with glassmorphism UI
│
├── api/                        # API route modules
│   ├── __init__.py
│   └── multi_camera.py         # Multi-camera configuration and processing endpoints
│
├── services/                   # Business logic and processing services
│   ├── __init__.py
│   ├── processor.py            # Frame processing generator (reusable)
│   ├── stream.py               # Live streaming MJPEG generator
│   └── analytics.py            # Analytics calculation utilities
│
├── myutils/                    # Utility modules and helpers
│   ├── detector.py             # YOLO detector wrapper
│   ├── tracker.py              # DeepSort format conversion utilities
│   └── panel.py                # Video overlay panel rendering
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures and configuration
│   ├── test_api.py             # API endpoint tests
│   └── test_occupancy.py       # Unit tests for core logic
│
├── data/                       # Data directory
│   └── raw_videos/             # Sample videos for testing
│
├── uploads/                    # Temporary upload directory (gitignored)
├── outputs/                    # Processed videos and results (gitignored)
│   └── recordings/             # Live stream recordings
│
├── requirements.txt            # Python dependencies (with GPU support)
├── requirements-docker.txt     # Minimal dependencies for Docker
│
├── Dockerfile                  # Multi-stage Docker image definition
├── docker-compose.yml          # Docker Compose with GPU support
├── docker-compose.cpu.yml      # Docker Compose for CPU-only deployment
│
├── .gitignore                  # Git ignore patterns
├── .dockerignore               # Docker build context exclusions
│
├── analysis.db                 # SQLite database (auto-created)
├── yolo11m.pt                  # YOLO model weights (auto-downloaded)
│
└── README.md                   # This file
```

### Key Components Explained

#### Core Files

- **`app.py`**: FastAPI application with all REST endpoints, middleware setup, and dependency injection
- **`process_video.py`**: Main video processing pipeline containing:
  - `MultiCameraManager`: Handles multi-camera deduplication and conflict resolution
  - `ChairOccupancyTracker`: Core tracking logic with smart occupancy detection
  - `process_video_for_api()`: Main entry point for video analysis
  
- **`websocket_handler.py`**: Real-time communication layer with:
  - `ProgressManager`: Tracks video processing progress
  - `LiveStatsManager`: Broadcasts live stream statistics
  - `LiveVideoStreamer`: Manages WebSocket video streaming with recording

#### Service Layer

- **`services/processor.py`**: Reusable frame processing generator used by both batch and streaming
- **`services/stream.py`**: MJPEG stream generator for live camera feeds
- **`services/analytics.py`**: Analytics calculation and aggregation logic

#### Utilities

- **`myutils/detector.py`**: Wrapper around YOLO for consistent detection interface
- **`myutils/tracker.py`**: DeepSort format conversion and tracking utilities
- **`myutils/panel.py`**: Renders overlay panels on video frames

---

## Advanced Features

### Smart Chair Tracking

The system includes sophisticated tracking capabilities:

#### 1. **Position Interpolation**
When a chair is temporarily occluded, its position is interpolated based on recent history:
```python
# Automatically interpolates missing chair positions
interpolated_bbox = tracker.interpolate_chair_position(chair_id)
```

#### 2. **Chair Signature Matching**
Uses color histograms and spatial features to re-identify chairs after occlusion:
```python
signature = tracker.extract_chair_signature(frame, bbox)
similarity = tracker.compare_chair_signatures(sig1, sig2)
```

#### 3. **Reflection Filtering**
Detects and filters false positives from reflective surfaces:
```python
reflection_zones = tracker.detect_reflections(frame)
if not tracker.is_in_reflection_zone(detection_bbox):
    # Process detection
```

#### 4. **Motion Blur Handling**
Adapts processing when rapid camera or subject movement is detected:
```python
blur_score = tracker.detect_motion_blur(frame)
if blur_score < threshold:
    # Apply interpolation or skip frame
```

### Zone-Based Occupancy Detection

Instead of simple center-point proximity, the system divides chairs into zones:

```python
chair_zones = tracker.get_chair_zones(chair_bbox)  # {'seat': bbox, 'back': bbox}
occupied_zones = tracker.check_zone_occupancy(person_center, chair_zones)
```

This enables detection of:
- Person sitting on chair (seat zone occupied)
- Person leaning on chair (back zone occupied)
- Person standing near chair (proximity but no zone occupation)

### Multi-Camera Deduplication

When multiple cameras view the same space:

```python
multi_cam_manager = MultiCameraManager()

# Set camera priorities and zones
multi_cam_manager.set_camera_zones(camera_id=1, zones=[bbox], priority=10)
multi_cam_manager.set_camera_zones(camera_id=2, zones=[bbox], priority=5)

# Define overlapping regions
multi_cam_manager.add_overlap_region(region=bbox, camera_ids=[1, 2])

# Resolve conflicts and get unified count
unified_stats = multi_cam_manager.get_unified_occupancy_count(all_camera_data)
```

**Conflict Resolution Strategy:**
1. Check if chair is in camera's authoritative zone
2. If in overlap region, higher priority camera wins
3. Spatial proximity used to detect duplicate chairs across cameras
4. Global chair registry maintains consistent IDs

---

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test modules
pytest tests/test_api.py -v
pytest tests/test_occupancy.py -v

# Run specific test class
pytest tests/test_occupancy.py::TestConfigValidation -v

# Run specific test
pytest tests/test_api.py::test_health_check -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_api.py              # API endpoint tests
│   ├── test_health_check()
│   ├── test_process_video_endpoint()
│   ├── test_get_analysis_history()
│   ├── test_multi_camera_setup()
│   └── ...
└── test_occupancy.py        # Core logic tests
    ├── TestConfigValidation
    ├── TestChairOccupancyTracker
    ├── TestMultiCameraManager
    └── ...
```

### Writing Tests

Example test for API endpoint:

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_process_video():
    with open("test_video.mp4", "rb") as video:
        response = client.post(
            "/process-video",
            files={"file": video},
            data={"proximity_threshold": 80}
        )
    assert response.status_code == 200
    assert response.json()["success"] is True
```

---

## Cleanup & Maintenance

### Automatic Cleanup

The project includes a Docker service for automatic cleanup:

```yaml
# In docker-compose.yml
cleanup-cron:
  command: |
    while true; do
      python cleanup.py --retention-days 7 --max-size-mb 5000
      sleep 86400
    done
```

### Manual Cleanup

```bash
# Preview what would be deleted (dry run)
python cleanup.py --dry-run

# Delete files older than 7 days
python cleanup.py --retention-days 7

# Keep directory under 5GB total
python cleanup.py --max-size-mb 5000

# Combine both strategies
python cleanup.py --retention-days 7 --max-size-mb 5000
```

### Cleanup Strategies

The `OutputCleaner` class implements two strategies:

1. **Age-Based Cleanup**: Removes files older than `retention_days`
2. **Size-Based Cleanup**: Removes oldest files when total size exceeds `max_size_mb`

Both strategies operate on file pairs (video + JSON results) to maintain consistency.

### Database Maintenance

```bash
# Backup database
cp analysis.db analysis.db.backup

# Vacuum database (reclaim space)
sqlite3 analysis.db "VACUUM;"

# View database statistics
sqlite3 analysis.db "SELECT COUNT(*) FROM analysis_results;"
```

---

## Deployment

### Production Deployment Checklist

- [ ] Change `API_KEY` from default value
- [ ] Set appropriate `MAX_FILE_SIZE` and `MAX_WORKERS`
- [ ] Configure persistent volumes for database and outputs
- [ ] Set up automated backups for `analysis.db`
- [ ] Configure cleanup cron job with appropriate retention
- [ ] Set up monitoring and logging
- [ ] Enable HTTPS with reverse proxy (nginx/traefik)
- [ ] Configure firewall rules (only expose port 8000 or reverse proxy port)
- [ ] Set up container restart policies
- [ ] Review and adjust resource limits

### Docker Production Deployment

1. **Create production environment file** (`.env.prod`):

```bash
API_KEY=your-very-secure-random-key-here
DATABASE_URL=sqlite:///./analysis.db
MAX_WORKERS=4
MAX_FILE_SIZE=524288000  # 500MB
DEFAULT_PROXIMITY_THRESHOLD=80
```

2. **Update docker-compose.yml**:

```yaml
version: '3.8'

services:
  chair-analytics:
    build: .
    container_name: chair_analytics_prod
    ports:
      - "8000:8000"
    volumes:
      - /var/chair_analytics/uploads:/app/uploads
      - /var/chair_analytics/outputs:/app/outputs
      - /var/chair_analytics/analysis.db:/app/analysis.db
    env_file:
      - .env.prod
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cleanup-cron:
    image: chair-occupancy-app
    container_name: chair_cleanup_prod
    command: |
      sh -c "while true; do
        python cleanup.py --retention-days 30 --max-size-mb 100000
        sleep 86400
      done"
    volumes:
      - /var/chair_analytics/outputs:/app/outputs
    restart: unless-stopped
```

3. **Deploy with Docker Compose**:

```bash
docker-compose up -d --build
```

4. **Set up nginx reverse proxy** (optional but recommended):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    client_max_body_size 500M;
}
```

### Cloud Platform Deployment

#### AWS EC2

```bash
# Launch EC2 instance with GPU (p3.2xlarge or similar)
# Install NVIDIA drivers and Docker
# Follow Docker deployment steps above

# Set up CloudWatch for monitoring
# Configure S3 for long-term storage of processed videos
```

#### Google Cloud Platform

```bash
# Create Compute Engine instance with GPU
# Install NVIDIA drivers
# Deploy using Docker Compose

# Use Cloud Storage for video archives
# Set up Cloud Monitoring
```

#### Azure

```bash
# Create Azure VM with GPU
# Install NVIDIA drivers
# Deploy with Docker

# Use Azure Blob Storage for archives
# Configure Azure Monitor
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests (if available), or create:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chair-analytics
spec:
  replicas: 2
  selector:
    matchLabels:
      app: chair-analytics
  template:
    metadata:
      labels:
        app: chair-analytics
    spec:
      containers:
      - name: app
        image: chair-occupancy-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: chair-analytics-secrets
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. YOLO Model Not Found

**Error**: `FileNotFoundError: yolo11m.pt not found`

**Solution**:
```bash
# Model auto-downloads on first run, but if it fails:
# Download manually from Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt
```

#### 2. CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
- Reduce `MAX_WORKERS` in config
- Process smaller videos
- Use CPU-only mode
- Reduce batch size in YOLO detection

```python
# In myutils/detector.py, adjust:
results = self.model(frame, verbose=False, imgsz=640)  # Reduce from 1280
```

#### 3. WebSocket Connection Refused

**Error**: WebSocket connection fails

**Solutions**:
- Check CORS settings in `app.py`
- Ensure FastAPI is running
- Verify WebSocket URL format
- Check firewall rules

#### 4. Video Processing Stalls

**Possible causes**:
- Corrupted video file
- Out of memory
- Disk space full

**Solutions**:
```bash
# Check disk space
df -h

# Check memory
free -h

# Test with smaller video
# Enable verbose logging in process_video.py
```

#### 5. Docker Container Exits Immediately

**Solutions**:
```bash
# Check logs
docker logs chair_project

# Common fix: NVIDIA runtime not configured
# Edit /etc/docker/daemon.json:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

#### 6. Database Locked Error

**Error**: `sqlite3.OperationalError: database is locked`

**Solutions**:
- Ensure only one process accesses database
- Increase timeout in `database.py`
- Consider PostgreSQL for production

```python
# In database.py
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}  # Increase timeout
)
```

### Performance Optimization

#### For Better FPS

```python
# In process_video.py
# Reduce detection frequency
if frame_number % 2 == 0:  # Detect every 2nd frame
    detections = detector.detect(frame)
```

#### For Better Accuracy

```python
# Increase occupancy confirmation threshold
tracker = ChairOccupancyTracker(
    proximity_threshold=60,  # Stricter proximity
    occupancy_frames_threshold=10  # More frames to confirm
)
```

#### For Lower Memory Usage

```python
# Process video in chunks
# Release frames immediately
# Use generators instead of loading entire video
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Commit changes**: `git commit -m 'Add new feature'`
4. **Push to branch**: `git push origin feature/your-feature-name`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions and classes
- Write tests for new features
- Update README if adding new functionality
- Ensure all tests pass before submitting PR

### Code Style

```python
# Use type hints
def calculate_distance(point1: tuple, point2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    pass

# Document complex logic
# Use descriptive variable names
# Keep functions focused and small
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Support

For issues, questions, or contributions:

- **Documentation**: This README and inline code documentation
- **Email**: <support-email>

---

## Acknowledgments

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: State-of-the-art object detection
- **[DeepSort](https://github.com/nwojke/deep_sort)**: Multiple object tracking algorithm
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern, fast web framework
- **[OpenCV](https://opencv.org/)**: Computer vision library

---

**Built with  for workspace optimization and smart facility management**
