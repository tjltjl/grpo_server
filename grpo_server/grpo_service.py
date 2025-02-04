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


@functools.cache
def get_settings():
    # pyright gets confused by pydantic_settings?
    return Settings()  # type: ignore


class Settings(pydantic_settings.BaseSettings):
    api_key: str
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
    # Load the ML model
    # TODO
    # the_trainer = grpo_trainer.grpo_trainer()
    yield
    # the_trainer.stop()


app = fastapi.FastAPI(lifespan=lifespan)


class CompletionsRequest(BaseModel):
    prompt: str


class CompletionsResponse(BaseModel):
    prompt: str
    completions: list[str]
    completion_tokens: list[list[int]]  # Needed to ensure we respond correctly.
    model_version: tuple[str, int]  # uuid of run + number of changes


@app.post("/completions", response_model=CompletionsResponse)
def completions(
    request: CompletionsRequest, api_key_check: bool = fastapi.Depends(verify_api_key)
) -> CompletionsResponse:
    # Mock implementation - replace with real completion logic
    completions = [request.prompt + "_completion"] * 2
    return CompletionsResponse(
        prompt=request.prompt, completions=completions, completion_tokens=[[]]
    )


class RewardsRequest(BaseModel):
    prompt: str
    completions: list[str]
    completion_tokens: list[str]
    rewards: list[float]

    # Could have things like the following:
    #   rewards: list[dict[str, float]]
    #   reward_formula: str
    #   total_rewards: list[float]
    # but those are outside the scope of this container.


class RewardsResponse(BaseModel):
    model_version: tuple[str, int]  # uuid of run + version has seen this example


@app.post("/rewards")
def rewards(
    request: RewardsRequest, api_key_check: bool = fastapi.Depends(verify_api_key)
) -> dict[str, t.Any]:
    # Mock implementation - replace with real rewards logic
    return {"stats": {}}
