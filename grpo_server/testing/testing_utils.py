"""Utilities for testing.

Including a really small language model (linear),
parameterizations to use it and a reward function.
"""

import dataclasses
import datasets
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
import transformers.utils.generic
import trl
import torch
import torch.nn as nn
from typeguard import typechecked

import grpo_server.grpo_trainer
import grpo_server.grpo_dataset
from grpo_server.testing import simple_linear_lm


class GeneratorWrapper:
    """Wrap a generator so that trying to pickle/dill it doesn't crash."""

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def __call__(self):
        return self

    def __getstate__(self):
        # Return a state that excludes the generator itself
        state = self.__dict__.copy()
        del state["generator"]
        return state

    def __setstate__(self, state):
        # Restore the state and recreate the generator
        self.__dict__.update(state)
        raise Exception("Can't unpickle")


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

    def create_model(self):
        return transformers.AutoModel.from_pretrained(self.model_name)

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
        model = self.create_model()

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
            reward_funcs=[self.calculate_rewards],
            args=self.grpo_config,
            train_dataset=self.dataset,
            # peft_config=peft.LoraConfig(task_type="CAUSAL_LM"),
            processing_class=self.create_tokenizer(),
        )

        trainer = trl.trainer.grpo_trainer.GRPOTrainer(**trainer_params)  # type: ignore

        return model, trainer

    def create_split_model_and_trainer_and_queuer(self, output_dir):
        model = self.create_model()

        self.grpo_config = trl.trainer.grpo_trainer.GRPOConfig(
            # beta=0.1,
            per_device_train_batch_size=1,
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
            num_train_epochs=1,
            max_steps=200,
            use_cpu=True,  # cpu
            save_strategy="no",  # Don't save (pickling queuer is no good)'
            # Otherwise, we get an error from accelerate dataset batching strs
            accelerator_config=dict(dispatch_batches=False),
        )
        # trainer.train()

        queuer = grpo_server.grpo_dataset.GRPOQueuer(model)

        dataset = datasets.Dataset.from_generator(
            GeneratorWrapper(queuer.data_getter()), streaming=True
        )

        trainer_params = dict(
            model=model,
            reward_funcs=[self.calculate_rewards],
            args=self.grpo_config,
            train_dataset=dataset,
            # peft_config=peft.LoraConfig(task_type="CAUSAL_LM"),
            processing_class=self.create_tokenizer(),
        )

        trainer = grpo_server.grpo_trainer.GRPOTrainerSplit(**trainer_params)  # type: ignore

        queuer.set_trainer(trainer)

        return model, trainer, queuer

    @typechecked
    def calculate_rewards(
        self, prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        """Reward: just keep repeating the last token."""
        # print(prompts, completions, kwargs)
        res = []
        for i, (p, c) in enumerate(zip(prompts, completions)):
            ids = (
                self.outside_tokenizer(p).input_ids
                + self.outside_tokenizer(c).input_ids
            )
            # print(ids)
            d = (np.diff(ids) == 0) + 0.0
            # print(d)
            res.append(float(np.mean(d)))
            # if i == 0:
            #    print(p, c, ids, res[-1])  # d,
        print("REWARDS", res)
        return res

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
            ok = ok and sum(self.calculate_rewards([prompt], [completion])) > 0.5
        return ok
