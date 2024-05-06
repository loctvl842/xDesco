from app.api import router
from core.cache import Cache, DefaultKeyMaker, RedisBackend
from core.settings import settings
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware


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


def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Trading Logic",
        description="Trading Logic API",
        version="0.0.1",
        docs_url=None if settings.ENV == "production" else "/docs",
        redoc_url=None if settings.ENV == "production" else "/redoc",
        middleware=make_middleware(),
    )
    app_.settings = settings
    init_routers(app_)
    init_cache()
    return app_


app = create_app()
