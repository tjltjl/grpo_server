from pydantic import BaseModel


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
    rewards: list[float]

    # Could have things like the following:
    #   rewards: list[dict[str, float]]
    #   reward_formula: str
    #   total_rewards: list[float]
    # but those are outside the scope of this container.


class RewardsResponse(BaseModel):
    model_version: tuple[str, int]  # uuid of run + version has seen this example
