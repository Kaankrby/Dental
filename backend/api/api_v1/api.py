from fastapi import APIRouter
from backend.api.api_v1.endpoints import analysis

api_router = APIRouter()
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
