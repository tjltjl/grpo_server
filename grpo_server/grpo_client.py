#  Copyright 2025 Tuomas J. Lukka. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import aiohttp
from typing import Optional
from grpo_server.data import (
    CompletionsRequest,
    CompletionsResponse,
    RewardsRequest,
    RewardsResponse,
    TrainingSettings,
    ModelRequest,
    StatusResponse,
)


class GrpoClient:
    def __init__(self, base_url: str, api_key: str):
        """Initialize the GRPO client.

        Args:
            base_url: Base URL of the GRPO service (e.g. "http://localhost:8000")
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers={"api-key": self.api_key})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("Client must be used as an async context manager")
        return self._session

    async def start(self, settings: TrainingSettings):
        """Start training with given settings."""
        async with self.session.post(
            f"{self.base_url}/start", json=settings.model_dump()
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def stop(self):
        """Stop training."""
        async with self.session.post(f"{self.base_url}/stop") as response:
            response.raise_for_status()
            return await response.json()

    async def get_completions(self, request: CompletionsRequest) -> CompletionsResponse:
        """Get completions for a prompt."""
        async with self.session.post(
            f"{self.base_url}/completions", json=request.model_dump()
        ) as response:
            response.raise_for_status()
            return CompletionsResponse(**await response.json())

    async def submit_rewards(self, request: RewardsRequest) -> RewardsResponse:
        """Submit rewards for completions."""
        async with self.session.post(
            f"{self.base_url}/rewards", json=request.model_dump()
        ) as response:
            response.raise_for_status()
            return RewardsResponse(**await response.json())

    async def get_training_settings(self) -> TrainingSettings:
        """Get current training settings."""
        async with self.session.get(f"{self.base_url}/training_settings") as response:
            response.raise_for_status()
            return TrainingSettings(**await response.json())

    async def get_status(self) -> StatusResponse:
        """Get current server status."""
        async with self.session.get(f"{self.base_url}/status") as response:
            response.raise_for_status()
            return StatusResponse(**await response.json())

    async def get_model(self, request: ModelRequest) -> bytes:
        """Get the current model as a zip file."""
        async with self.session.post(
            f"{self.base_url}/model", json=request.model_dump()
        ) as response:
            response.raise_for_status()
            return await response.read()
