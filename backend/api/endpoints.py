from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Body
from typing import List, Dict, Any, Optional
import uuid
import shutil
import os
from ..core.processing import RhinoAnalyzer
import logging
import numpy as np
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Global store for simplicity (in production use Redis/DB)
jobs: Dict[str, Dict] = {}
analyzer = RhinoAnalyzer()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
current_test_files: List[str] = []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(metrics, dict):
        return {k: serialize_metrics(v) for k, v in metrics.items()}
    elif isinstance(metrics, list):
        return [serialize_metrics(v) for v in metrics]
    elif isinstance(metrics, np.integer):
        return int(metrics)
    elif isinstance(metrics, np.floating):
        return float(metrics)
    elif isinstance(metrics, np.ndarray):
        return metrics.tolist()
    else:
        return metrics

class AnalysisParams(BaseModel):
    processing_mode: str = "Balanced"
    num_points: int = 15000
    use_global_reg: bool = True
    voxel_size: float = 1.5
    icp_threshold: float = 0.3
    icp_max_iter: int = 200
    icp_mode: str = "auto"
    stl_scale: float = 1.0
    ignore_outside_bbox: bool = False
    use_full_ref_global: bool = False
    include_notimportant_metrics: bool = False
    volume_voxel_size: float = 0.5

@router.post("/upload/reference")
async def upload_reference(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"ref_{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load reference immediately with default weights
        analyzer.load_reference_3dm(file_path, layer_weights={})
    except Exception as e:
        logger.error(f"Failed to load reference: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load reference: {str(e)}")
        
    return {"filename": file.filename, "message": "Reference uploaded and loaded successfully"}

@router.post("/upload/test")
async def upload_test(files: List[UploadFile] = File(...)):
    global current_test_files
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, f"test_{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
        
    current_test_files = saved_files
    
    return {"filenames": [f.filename for f in files], "message": "Test files uploaded successfully"}

@router.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, params: AnalysisParams = Body(...)):
    if not current_test_files:
        raise HTTPException(status_code=400, detail="No test files uploaded")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    
    # Run analysis in background
    background_tasks.add_task(run_analysis, job_id, current_test_files, params)
    
    return {"job_id": job_id, "message": "Analysis started"}

def run_analysis(job_id: str, test_files: List[str], params: AnalysisParams):
    results = []
    try:
        for file_path in test_files:
            res = analyzer.process_test_file(
                file_path=file_path,
                stl_scale_to_mm=params.stl_scale,
                use_global_reg=params.use_global_reg,
                voxel_size=params.voxel_size,
                icp_threshold=params.icp_threshold,
                max_iter=params.icp_max_iter,
                ignore_outside_bbox=params.ignore_outside_bbox,
                include_notimportant_metrics=params.include_notimportant_metrics,
                use_full_ref_global=params.use_full_ref_global,
                icp_mode=params.icp_mode,
                volume_voxel_size=params.volume_voxel_size
            )
            
            # Extract and serialize metrics
            metrics = serialize_metrics(res['metrics'])
            
            results.append({
                "file": os.path.basename(file_path),
                "metrics": metrics
            })
            
        jobs[job_id] = {"status": "completed", "results": results}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        jobs[job_id] = {"status": "failed", "error": str(e)}

@router.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
