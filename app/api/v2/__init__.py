from fastapi import APIRouter
from app.api.v1.disease import router as disease_router

router = APIRouter(prefix="/v2")
router.include_router(disease_router)

__all__ = ["router"]
