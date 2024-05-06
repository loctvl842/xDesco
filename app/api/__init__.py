from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.api.ping import router as ping_router
from app.api.v1 import router as router_v1
from app.api.v2 import router as router_v2

router = APIRouter()
router.include_router(ping_router)
router.include_router(router_v1)
router.include_router(router_v2)
templates = Jinja2Templates(directory="templates")


@router.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


__all__ = ["router"]
