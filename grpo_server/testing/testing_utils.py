"""Utilities for testing.

Including a really small language model (linear),
parameterizations to use it and a reward function.
"""

import asyncio
import dataclasses
import datasets
import logging
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
import transformers.utils.generic
import trl
import torch
import torch.nn as nn
from typeguard import typechecked
import typing as t

import grpo_server.grpo_trainer_reversed
from grpo_server import grpo_queuer
from grpo_server.testing import simple_linear_lm

logger = logging.getLogger(__name__)

# This is the config that is saved to test_data/simple_linear_40_5/config.json
# to allow getting that model with from_pretrained.
SIMPLE_MODEL_CONFIG = simple_linear_lm.SimpleLinearLMConfig(
    vocab_size=40,
    context_size=2,
    hidden_dim=5,
)


class SimpleProblem:
    def __init__(self):
        # model_name = "HuggingFaceTB/SmolLM-135M"
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #    model_name, padding_side="left"
        # )
        # Would use 2 but the special tokens get in the way
        self.n_vocab = 40

        rng = np.random.default_rng(43)
        # The tokenizer to use outside training
        self.outside_tokenizer = self.create_tokenizer()

        self.dataset = datasets.Dataset.from_list(
            [
                dict(prompt=self.generate_prompt(rng, self.outside_tokenizer))
                for i in range(100)
            ]
        )
        self.model_name = "./test_data/simple_linear_40_5"

    def create_tokenizer(self):
        """Tokenizers are not re-entrant, create one."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "./test_data/smolm135_tokenizer"  # "/tokenizer_config.json",
        )

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate_prompt(self, rng, tokenizer):
        while True:
            # Generate until ok (may get too large tokens
            # otherwise)
            prompt = tokenizer.decode(
                torch.as_tensor(rng.integers(self.n_vocab, size=(rng.integers(2, 10),)))
            )
            if np.all(np.array(tokenizer(prompt).input_ids) < self.n_vocab):
                return prompt

    def create_normal_model_and_trainer(self, output_dir):
        """Create a model and a trainer on conventional dataset."""
        model = transformers.AutoModel.from_pretrained(self.model_name)

        self.grpo_config = trl.trainer.grpo_trainer.GRPOConfig(
            # beta=0.1,
            per_device_train_batch_size=4,
            output_dir=output_dir,
            do_train=True,
            do_eval=False,
            learning_rate=5e-2,
            logging_steps=1,
            gradient_accumulation_steps=1,
            max_completion_length=6,
            eval_on_start=False,
            label_names=[],
            save_steps=50,
            weight_decay=0.001,
            num_train_epochs=20,
            use_cpu=True,  # cpu
        )
        # trainer.train()

        trainer_params = dict(
            model=model,
            reward_funcs=[self.reward_adapter],
            args=self.grpo_config,
            train_dataset=self.dataset,
            # peft_config=peft.LoraConfig(task_type="CAUSAL_LM"),
            processing_class=self.create_tokenizer(),
        )

        trainer = trl.trainer.grpo_trainer.GRPOTrainer(**trainer_params)  # type: ignore

        return model, trainer

    def create_split_model_and_trainer_and_queuer(self, output_dir):
        """Create model and trainer + queuer to reverse the control flow"""

        # The defaults are the test values...
        training_settings = grpo_queuer.TrainingSettings()

        queuer: grpo_queuer.GRPOQueuer = grpo_queuer.create_queuer(
            training_settings, output_dir=output_dir
        )
        trainer = queuer.trainer
        model = queuer.model

        return model, trainer, queuer

    @typechecked
    def reward_adapter(self, prompts: list[str], completions: list[str]) -> list[float]:
        return [
            self.calculate_rewards(
                grpo_queuer.CompletionDict(
                    prompt=prompt,
                    completions=[completion],
                    extra={},
                )
            )["rewards"][0]
            for prompt, completion in zip(prompts, completions, strict=True)
        ]

    @typechecked
    def calculate_rewards(
        self, completion_dict: grpo_queuer.CompletionDict
    ) -> grpo_queuer.RewardDict:
        """Calculate rewards for a completion.
        Reward: just keep repeating the last token.
        """
        prompt = completion_dict["prompt"]
        completions = completion_dict["completions"]

        rewards = []
        for completion in completions:
            ids = (
                self.outside_tokenizer(prompt).input_ids
                + self.outside_tokenizer(completion).input_ids
            )
            d = (np.diff(ids) == 0) + 0.0
            rewards.append(float(np.mean(d)))

        print("REWARDS", rewards)

        return grpo_queuer.RewardDict(
            prompt=prompt,
            completions=completions,
            rewards=rewards,
            extra=completion_dict.get("extra", {}),
        )

    def has_learned(self, model):
        ok = True
        for i in range(10):
            prompt = self.dataset[i]["prompt"]
            print(i, prompt)
            input_tokens = self.outside_tokenizer(prompt)
            print("IT", input_tokens)
            output_tokens = model.generate(
                torch.as_tensor(input_tokens.input_ids)[None, :]
            )
            completion = self.outside_tokenizer.decode(
                output_tokens[0, len(input_tokens.input_ids) :]
            )
            print("TP", prompt, completion)

            reward_dict = self.calculate_rewards(
                grpo_queuer.CompletionDict(
                    prompt=prompt,
                    completions=[completion],
                    extra={},
                )
            )
            ok = ok and reward_dict["rewards"][0] > 0.5
        return ok

    async def run_loop_async(
        self,
        get_completions: t.Callable[
            [grpo_queuer.PromptDict],
            t.Coroutine[t.Any, t.Any, grpo_queuer.CompletionDict],
        ],
        set_rewards: t.Callable[
            [grpo_queuer.RewardDict], t.Coroutine[t.Any, t.Any, t.Any]
        ],
    ):
        """Run an async loop training given the ops to use.

        Used to test both grpo_service and grpo_queuer
        """

        async with asyncio.TaskGroup() as tg:
            while True:
                for row in self.dataset:
                    prompt: str = row["prompt"]  # type: ignore
                    logger.debug("call get completions %s", prompt)
                    completions = await get_completions(
                        grpo_queuer.PromptDict(prompt=prompt)
                    )
                    logger.debug("got completions %s", completions)
                    rewards_dict = self.calculate_rewards(completions)

                    logger.debug("setting rewards: %s", rewards_dict)
                    tg.create_task(set_rewards(rewards_dict))
