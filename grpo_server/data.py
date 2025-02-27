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

from pydantic import BaseModel
import typing as t


class CompletionsRequest(BaseModel):
    prompt: str


class CompletionsResponse(BaseModel):
    prompt: str
    completions: list[str]
    completion_tokens: list[list[int]]  # Needed to ensure we respond correctly.
    model_version: tuple[str, int]  # uuid of run + number of changes


class RewardsRequest(BaseModel):
    prompt: str
    completions: list[str]
    completion_tokens: list[list[int]]
    rewards: t.Sequence[float]

    @classmethod
    def from_completions(cls, completions, rewards):
        return cls(
            prompt=completions.prompt,
            completions=completions.completions,
            completion_tokens=completions.completion_tokens,
            rewards=rewards,
        )

    # Could have things like the following:
    #   rewards: list[dict[str, float]]
    #   reward_formula: str
    #   total_rewards: list[float]
    # but those are outside the scope of this container.


class RewardsResponse(BaseModel):
    model_version: tuple[str, int]  # uuid of run + version has seen this example


class TrainingSettings(BaseModel):
    # Model to start from
    model_id: str = "./test_data/simple_linear_40_5"
    max_completion_length: int = 6
    num_completions_per_prompt: int = 3

    training_batch_size: int = 1  # Larger needs debugging (padding)
    learning_rate: float = 5e-2
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    max_steps: int = 201


class ModelRequest(BaseModel):
    pass
    # model_version: tuple[str, int] TODO


class StatusResponse(BaseModel):
    training_settings: TrainingSettings | None = None
    completion_requests_served: int = 0
    reward_requests_served: int = 0
