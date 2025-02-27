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
from grpo_server import data, grpo_queuer
from grpo_server.data import TrainingSettings
from grpo_server.testing import simple_linear_lm

logger = logging.getLogger(__name__)
logdebug = logger.critical


# This is the config that is saved to test_data/simple_linear_40_5/config.json
# to allow getting that model with from_pretrained.
SIMPLE_MODEL_CONFIG = simple_linear_lm.SimpleLinearLMConfig(
    vocab_size=40,
    context_size=2,
    hidden_dim=5,
)

SIMPLE_MODEL_CONFIG_LARGE = simple_linear_lm.SimpleLinearLMConfig(
    vocab_size=49152,
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
        training_settings = TrainingSettings()

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
                data.CompletionsResponse(
                    prompt=prompt,
                    completions=[completion],
                    completion_tokens=[[0]],
                    model_version=("", 0),
                )
            ).rewards[0]
            for prompt, completion in zip(prompts, completions, strict=True)
        ]

    @typechecked
    def calculate_rewards(
        self, completions_response: data.CompletionsResponse
    ) -> data.RewardsRequest:
        """Calculate rewards for a completion.
        Reward: just keep repeating the last token.
        """
        prompt = completions_response.prompt
        completions = completions_response.completions

        rewards = []
        for completion in completions:
            ids = (
                self.outside_tokenizer(prompt).input_ids
                + self.outside_tokenizer(completion).input_ids
            )
            d = (np.diff(ids) == 0) + 0.0
            rewards.append(float(np.mean(d)))

        return data.RewardsRequest(
            prompt=prompt,
            completions=completions,
            rewards=rewards,
            completion_tokens=completions_response.completion_tokens,
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
                data.CompletionsResponse(
                    prompt=prompt,
                    completions=[completion],
                    completion_tokens=[],
                    model_version=("", 0),
                )
            )
            ok = ok and reward_dict.rewards[0] > 0.5
        return ok

    async def run_loop_async(
        self,
        get_completions: t.Callable[
            [data.CompletionsRequest],
            t.Coroutine[t.Any, t.Any, data.CompletionsResponse],
        ],
        set_rewards: t.Callable[
            [data.RewardsRequest], t.Coroutine[t.Any, t.Any, t.Any]
        ],
    ):
        """Run an async loop training given the ops to use.

        Used to test both grpo_service and grpo_queuer
        """

        tasks = []
        try:
            async with asyncio.TaskGroup() as tg:
                while True:
                    for row in self.dataset:
                        prompt: str = row["prompt"]  # type: ignore
                        logdebug("call get completions %s", prompt)
                        completions = await get_completions(
                            data.CompletionsRequest(prompt=prompt)
                        )
                        logdebug("got completions %s", completions)
                        rewards_dict = self.calculate_rewards(completions)

                        logdebug("setting rewards: %s", rewards_dict)
                        tasks.append(tg.create_task(set_rewards(rewards_dict)))
            for task in tasks:
                task.result()  # Raise exceptions here
        except grpo_queuer.StopTrainingException as e:
            logdebug("stop training exception caught, exiting")
