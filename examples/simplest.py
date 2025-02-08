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

from grpo_server import grpo_client, data
from grpo_server.data import *


async def main():
    async with grpo_client.GrpoClient("http://localhost:3247", "default_key") as client:

        # Start training
        settings = TrainingSettings(
            model_id="test_data/simple_linear_49152_5",
            max_completion_length=12,
            num_completions_per_prompt=16,
            training_batch_size=1,
            learning_rate=5e-2,
            max_steps=2000,
        )
        await client.start(settings)

        while True:
            completions = await client.get_completions(
                CompletionsRequest(prompt="Hello world")
            )

            rewards = [len([c for c in s if c == "*"]) for s in completions.completions]

            print(f"Got completions: {completions.completions}: {rewards}")

            # Submit rewards
            rewards_response = await client.submit_rewards(
                RewardsRequest(
                    prompt="Hello world",
                    completions=completions.completions,
                    completion_tokens=completions.completion_tokens,
                    rewards=rewards,
                )
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
