# Dental Analysis Platform (Modern Stack)

This is the modernized version of the Dental STL Analyzer, re-architected as a scalable Client-Server application.

## Architecture

- **Frontend**: React (Vite + TypeScript) - `http://localhost:5173`
- **Backend**: FastAPI (Python) - `http://localhost:8000`
- **Worker**: Celery + Redis for background 3D processing
- **Infrastructure**: Docker Compose

## Quick Start

The easiest way to run the application is using Docker.

### Prerequisites
- Docker Desktop installed and running.

### Run with Docker

```bash
# Build and start all services
docker-compose up --build
```

Once running, access the application:
- **Web UI**: [http://localhost:5173](http://localhost:5173)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Development

#### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Features (Migrated)

- **3D Analysis**: Upload Reference (`.3dm`) and Test (`.stl`) files.
- **Background Processing**: Heavy ICP registration and deviation calculations are handled by Celery workers to keep the UI responsive.
- **Modern UI**: React-based interface for better interactivity (Work in Progress).

## Project Structure

```
Dental/
├── backend/             # FastAPI application
│   ├── api/             # API endpoints
│   ├── core/            # Core logic (refactored from original scripts)
│   └── worker/          # Celery tasks
├── frontend/            # React application
├── docker-compose.yml   # Orchestration
└── README.md
```
