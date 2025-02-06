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
