"""
Note: RUN ONLY IN A SINGLE PROCESS!

Restart for a different training run.

Everything that is not "get completions" and "give reward"
happens outside this server.
"""

from contextlib import asynccontextmanager
import docker.errors
import fastapi
import functools
from pydantic import BaseModel
import pydantic_settings
import typing as t

from grpo_server import grpo_queuer
from grpo_server.data import *


@functools.cache
def get_settings():
    # pyright gets confused by pydantic_settings?
    return Settings()  # type: ignore


class Settings(pydantic_settings.BaseSettings):
    api_key: str
    output_dir: str
    training: grpo_queuer.TrainingSettings = grpo_queuer.TrainingSettings()


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

    app.state.queuer = grpo_queuer.create_queuer(settings.training, settings.output_dir)
    async with app.state.queuer.context():
        print("APP STATE QUEUER", app.state.queuer)

        yield
        print("OUT OF LIFESPAN")


def get_queuer() -> grpo_queuer.BaseQueuer:
    return app.state.queuer


app = fastapi.FastAPI(lifespan=lifespan)


@app.post("/completions", response_model=CompletionsResponse)
async def completions(
    request: CompletionsRequest, api_key_check: bool = fastapi.Depends(verify_api_key)
) -> CompletionsResponse:
    return await get_queuer().get_completions(request)


@app.post("/rewards")
async def rewards(
    request: RewardsRequest, api_key_check: bool = fastapi.Depends(verify_api_key)
) -> RewardsResponse:

    return await get_queuer().rewards(request)
