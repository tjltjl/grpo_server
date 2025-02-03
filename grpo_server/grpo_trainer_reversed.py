import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Below: modified version of GRPO trainer compute_loss

import logging
import torch
from trl.data_utils import maybe_apply_chat_template
import trl.trainer.grpo_trainer

logger = logging.getLogger(__name__)
# pytest doesn't flush logger.debug()..
logdebug = logger.info

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from trl.trainer.grpo_trainer import (
    maybe_apply_chat_template,
    unwrap_model_for_generation,
    gather_object,
    broadcast_object_list,
    pad,
    is_conversational,
    PreTrainedModel,
    apply_chat_template,
)

# pyright: basic
# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false


class GRPOTrainerSplit(trl.trainer.GRPOTrainer):

    def prepare_prompts(self, inputs):
        logdebug("Prepare prompts: %s", inputs)

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]  # type: ignore
            for example in inputs
        ]
        prompt_inputs = self.processing_class(  # type: ignore
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",  # type: ignore
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)  # type: ignore

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -self.max_prompt_length :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -self.max_prompt_length :
            ]

        return prompts, prompts_text, prompt_inputs

    def generate_completions(self, model, inputs):
        """Map inputs to a new one with ids and completion"""
        device = self.accelerator.device

        prompts, prompts_text, prompt_inputs = self.prepare_prompts(inputs)

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    model, self.accelerator
                ) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(
                    all_prompts_text,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [
                    out.token_ids
                    for completions in outputs
                    for out in completions.outputs
                ]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1)
                * len(prompts)
                * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)  # type: ignore
            prompt_inputs_repeated = torch.repeat_interleave(
                prompt_inputs["input_ids"], self.num_generations, dim=0
            )
            prompt_completion_ids = torch.cat(
                [prompt_inputs_repeated, completion_ids], dim=1
            )
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        flat_completions = self.processing_class.batch_decode(  # type: ignore
            completion_ids, skip_special_tokens=True
        )
        completions = [
            flat_completions[i : i + self.num_generations]
            for i in range(0, len(flat_completions), self.num_generations)
        ]

        prompt_completion_ids = torch.reshape(
            prompt_completion_ids, (len(inputs), self.num_generations, -1)
        )
        logdebug("Completions: %s %s", inputs, prompt_completion_ids)
        return [
            dict(input, completions=compl, extra=dict(prompt_completion_ids=ids))
            for input, compl, ids in zip(
                inputs, completions, prompt_completion_ids, strict=True
            )
        ]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # Original structure:
        # prepare_prompts
        # infer completions
        # -> proceed
        #
        # We use prepare_prompts in both generate_completions and compute_loss,
        # and pass the completions and rewards to compute_loss directly

        prompts, prompts_text, prompt_inputs = self.prepare_prompts(inputs)

        print(
            "COMPLIDS",
            [example["extra"]["prompt_completion_ids"] for example in inputs],
        )
        prompt_completion_ids = torch.cat(
            [example["extra"]["prompt_completion_ids"] for example in inputs], axis=0
        )
        rewards = torch.cat(
            [torch.as_tensor(example["rewards"]) for example in inputs], axis=0
        )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids, num_logits_to_keep=num_logits_to_keep + 1
            ).logits  # (B, L, V)
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(
                logits, input_ids[:, -num_logits_to_keep:]
            ):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(
            model, prompt_completion_ids, num_logits_to_keep
        )

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, prompt_completion_ids, num_logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(
                        model, prompt_completion_ids, num_logits_to_keep
                    )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id  # type: ignore
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # # Decode the generated completions
        # completions = self.processing_class.batch_decode(  # type: ignore
        #     completion_ids, skip_special_tokens=True
        # )
        # if is_conversational(inputs[0]):
        #     completions = [
        #         [{"role": "assistant", "content": completion}]
        #         for completion in completions
        #     ]

        # # Compute the rewards
        # prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        # rewards_per_func = torch.zeros(
        #     len(prompts), len(self.reward_funcs), device=device
        # )
        # for i, (reward_func, reward_processing_class) in enumerate(  # type: ignore
        #     zip(self.reward_funcs, self.reward_processing_classes)
        # ):
        #     if isinstance(reward_func, PreTrainedModel):
        #         if is_conversational(inputs[0]):
        #             messages = [
        #                 {"messages": p + c} for p, c in zip(prompts, completions)
        #             ]
        #             texts = [
        #                 apply_chat_template(x, reward_processing_class)["text"]
        #                 for x in messages
        #             ]
        #         else:
        #             texts = [p + c for p, c in zip(prompts, completions)]
        #         reward_inputs = reward_processing_class(
        #             texts,
        #             return_tensors="pt",
        #             padding=True,
        #             padding_side="right",
        #             add_special_tokens=False,
        #         )
        #         reward_inputs = super()._prepare_inputs(reward_inputs)
        #         with torch.inference_mode():
        #             rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
        #                 :, 0
        #             ]  # Shape (B*G,)
        #     else:
        #         # Repeat all input columns (but "prompt" and "completion") to match the number of generations
        #         reward_kwargs = {
        #             key: []
        #             for key in inputs[0].keys()
        #             if key not in ["prompt", "completion"]
        #         }
        #         for key in reward_kwargs:
        #             for example in inputs:
        #                 # Repeat each value in the column for `num_generations` times
        #                 reward_kwargs[key].extend([example[key]] * self.num_generations)
        #         output_reward_func = reward_func(
        #             prompts=prompts, completions=completions, **reward_kwargs
        #         )
        #         rewards_per_func[:, i] = torch.tensor(
        #             output_reward_func, dtype=torch.float32, device=device
        #         )

        # # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        logdebug("REW %s %s %s", rewards, per_token_logps, advantages)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        # for i, reward_func in enumerate(self.reward_funcs):
        #     if isinstance(reward_func, PreTrainedModel):
        #         reward_func_name = reward_func.config._name_or_path.split("/")[-1]
        #     else:
        #         reward_func_name = reward_func.__name__
        #     self._metrics[f"rewards/{reward_func_name}"].append(
        #         reward_per_func[i].item()
        #     )

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss
