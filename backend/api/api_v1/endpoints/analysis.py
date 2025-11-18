from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()

@router.post("/upload/reference")
async def upload_reference(file: UploadFile = File(...)):
    """
    Upload a reference .3dm file.
    """
    if not file.filename.endswith('.3dm'):
        raise HTTPException(status_code=400, detail="File must be a .3dm file")
    
    return {"filename": file.filename, "status": "uploaded"}

@router.post("/upload/test")
async def upload_test(files: List[UploadFile] = File(...)):
    """
    Upload one or more test .stl files.
    """
    uploaded_files = []
    for file in files:
        if not file.filename.lower().endswith('.stl'):
            continue
        uploaded_files.append(file.filename)
        
    return {"uploaded": uploaded_files, "count": len(uploaded_files)}
