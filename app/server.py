from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from ai_kit import load_models
from app.api import router
from core.cache import Cache, DefaultKeyMaker, RedisBackend
from core.settings import settings


def init_routers(app_: FastAPI) -> None:
    app_.include_router(router)


def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    return middleware


def init_cache() -> None:
    Cache.configure(backend=RedisBackend(), key_maker=DefaultKeyMaker())


@asynccontextmanager
async def lifespan(app: FastAPI):
    seg_model, classif_model, rf, anomaly_extractor = load_models()
    app.state.models = seg_model, classif_model, rf, anomaly_extractor
    yield


def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Trading Logic",
        description="Trading Logic API",
        version="0.0.1",
        docs_url=None if settings.ENV == "production" else "/docs",
        redoc_url=None if settings.ENV == "production" else "/redoc",
        middleware=make_middleware(),
        lifespan=lifespan,
    )
    app_.settings = settings
    init_routers(app_)
    init_cache()
    return app_


app = create_app()


def get_app() -> FastAPI:
    return app
