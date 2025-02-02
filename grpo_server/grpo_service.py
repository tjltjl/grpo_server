"""
Note: RUN ONLY IN A SINGLE PROCESS!

Restart for a different training run.

Everything that is not "get completions" and "give reward"
happens outside this server.
"""

from contextlib import asynccontextmanager
import docker.errors
import fastapi
from pydantic import BaseModel
import typing as t

import grpo_trainer


def verify_api_key():
    # Read API key once at startup
    with open("sandboxer_api_key") as f:
        api_key = f.read().strip()

    def verify_key(api_key_header: str = fastapi.Header(None, alias="api-key")):
        if api_key_header != api_key:
            raise fastapi.HTTPException(status_code=401, detail="Invalid API key")
        return api_key_header

    return verify_key


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Load the ML model
    # TODO
    # the_trainer = grpo_trainer.grpo_trainer()
    yield
    # the_trainer.stop()


app = fastapi.FastAPI(lifespan=lifespan)


class CompletionsRequest(BaseModel):
    prompts: list[str]


class CompletionsResponse(BaseModel):
    prompts: list[str]
    completions: list[list[str]]


@app.post("/completions", response_model=CompletionsResponse)
def completions(
    request: CompletionsRequest, api_key: str = fastapi.Depends(verify_api_key)
) -> CompletionsResponse:
    # Mock implementation - replace with real completion logic
    completions = [[p + "_completion"] for p in request.prompts]
    return CompletionsResponse(prompts=request.prompts, completions=completions)


class RewardsRequest(BaseModel):
    prompts: list[str]
    completions: list[list[str]]
    rewards: list[list[float]]


@app.post("/rewards")
def rewards(
    request: RewardsRequest, api_key: str = fastapi.Depends(verify_api_key)
) -> dict[str, t.Any]:
    # Mock implementation - replace with real rewards logic
    return {"stats": {}}
