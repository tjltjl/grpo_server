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


class GeneratorWrapper:
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


@dataclasses.dataclass
class SimpleLinearLMOutput(transformers.utils.generic.ModelOutput):
    logits: torch.Tensor


class SimpleLinearLM(PreTrainedModel, GenerationMixin):
    def __init__(self, vocab_size, context_size, hidden_dim):
        super().__init__(PretrainedConfig())
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(context_size * hidden_dim, vocab_size)

    def forward(self, input_ids, return_dict=True, num_logits_to_keep=None):
        #  print(input_ids)

        batch_size, seq_len = input_ids.shape
        windows = []
        for i in range(seq_len):
            if i < self.context_size - 1:
                padded = torch.zeros(
                    (batch_size, self.context_size),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                if i > 0:
                    padded[:, -i - 1 :] = input_ids[:, : i + 1]
                windows.append(padded)
            else:
                windows.append(input_ids[:, i - self.context_size + 1 : i + 1])
        input_ids = torch.stack(windows, dim=1)

        # input_ids shape: (batch_size, seq_len, context_size)
        batch_size, seq_len, _ = input_ids.shape
        embeds = self.embeddings(
            input_ids
        )  # (batch_size, seq_len, context_size, hidden_dim)
        flatten = embeds.view(
            batch_size * seq_len, -1
        )  # (batch_size * seq_len, context_size * hidden_dim)
        logits = self.linear(flatten)  # (batch_size * seq_len, vocab_size)
        logits = logits.view(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, vocab_size)

        if num_logits_to_keep is not None:
            logits = logits[:, -num_logits_to_keep:, :]

        # print("LOGITS", torch.min(logits))
        # print(logits.shape)
        if return_dict:
            return SimpleLinearLMOutput(
                logits=logits,
            )
        return logits

    def prepare_inputs_for_generation(  # type: ignore
        self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs  # type: ignore
    ):  # type: ignore
        return dict(input_ids=input_ids)  # type: ignore


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
        model = SimpleLinearLM(self.n_vocab, 2, 5)

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
        model = SimpleLinearLM(self.n_vocab, 2, 5)

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
