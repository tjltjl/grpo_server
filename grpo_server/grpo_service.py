"""
Note: RUN ONLY IN A SINGLE PROCESS!

Restart for a different training run.

Everything that is not "get completions" and "give reward"
happens outside this server.
"""

import asyncio
from contextlib import asynccontextmanager
import docker.errors
import fastapi
import functools
import logging
from pydantic import BaseModel
import pydantic_settings
import typing as t
import uuid

from grpo_server import grpo_queuer
from grpo_server.testing import testing_utils
from grpo_server.data import *

logger = logging.getLogger(__name__)


class Settings(pydantic_settings.BaseSettings):
    api_key: str
    output_dir: str


global_settings: None | Settings = None


# TODO: dep. injections doesn't go right for lifespan?
@functools.cache
def get_settings():
    assert global_settings
    return global_settings


def verify_api_key(
    settings: Settings = fastapi.Depends(get_settings),
    api_key_header: str = fastapi.Header(None, alias="api-key"),
):

    if api_key_header != settings.api_key:
        raise fastapi.HTTPException(status_code=401, detail="Invalid API key")
    return True


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    settings = get_settings()
    logger.critical("Settings:\n%s", settings)
    app.state.queuer = None
    yield


def get_queuer() -> grpo_queuer.BaseQueuer:
    if app.state.queuer is None:
        raise fastapi.HTTPException(status_code=409, detail="Not started")
    return app.state.queuer


app = fastapi.FastAPI(lifespan=lifespan, dependencies=[fastapi.Depends(verify_api_key)])


@app.exception_handler(grpo_queuer.StopTrainingException)
def stop_exception_handler(request: fastapi.Request, exc: fastapi.HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=fastapi.status.HTTP_409_CONFLICT,
        content={"message": "training stopped"},
    )  # use the exc object's message attribute


# The heaviest-used training calls


@app.post("/start")
async def start(training_settings: TrainingSettings):
    settings = get_settings()
    app.state.queuer = grpo_queuer.create_queuer(training_settings, settings.output_dir)
    app.state.queuer_context = app.state.queuer.context()
    await app.state.queuer_context.__aenter__()

    return {"status": "started"}


@app.post("/stop")
async def stop():
    await app.state.queuer_context.__aexit__(None, None, None)
    app.state.queuer = None
    app.state.queuer_context = None

    return {"status": "stopped"}


@app.post("/completions", response_model=CompletionsResponse)
async def completions(request: CompletionsRequest) -> CompletionsResponse:
    return await get_queuer().get_completions(request)


@app.post("/rewards")
async def rewards(request: RewardsRequest) -> RewardsResponse:

    return await get_queuer().rewards(request)


# Bookkeeping


@app.get("/training_settings")
def training_settings() -> TrainingSettings:
    return get_settings().training


# TODO snapshots etc etc
@app.post("/model")
async def model(
    request: ModelRequest,
    settings=fastapi.Depends(get_settings),
) -> fastapi.responses.FileResponse:
    # TODO: Puts things in a weird location, doesn't delete
    # TODO: Do through queuer queue
    settings = get_settings()
    d = settings.output_dir
    zp = f"{d}/{uuid.uuid4()}.zip"
    zp = "/tmp/foo.zip"
    await asyncio.to_thread(lambda: get_queuer().create_snapshot(zp))
    return fastapi.responses.FileResponse(zp)
