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

import dataclasses
import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
import transformers.utils.generic


@dataclasses.dataclass
class SimpleLinearLMOutput(transformers.utils.generic.ModelOutput):
    logits: torch.Tensor


class SimpleLinearLMConfig(PretrainedConfig):
    model_type: str = "simple_linear_lm"
    vocab_size: int
    context_size: int
    hidden_dim: int


class SimpleLinearLM(PreTrainedModel, GenerationMixin):
    config_class = SimpleLinearLMConfig

    def __init__(self, config: SimpleLinearLMConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        self.linear = nn.Linear(
            self.config.context_size * self.config.hidden_dim, self.config.vocab_size
        )

    def forward(self, input_ids, return_dict=True, num_logits_to_keep=None):
        #  print(input_ids)

        batch_size, seq_len = input_ids.shape
        windows = []
        for i in range(seq_len):
            if i < self.config.context_size - 1:
                padded = torch.zeros(
                    (batch_size, self.config.context_size),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                if i > 0:
                    padded[:, -i - 1 :] = input_ids[:, : i + 1]
                windows.append(padded)
            else:
                windows.append(input_ids[:, i - self.config.context_size + 1 : i + 1])
        input_ids = torch.stack(windows, dim=1)

        # input_ids shape: (batch_size, seq_len, context_size)
        batch_size, seq_len, _ = input_ids.shape
        embeds = self.embeddings(
            input_ids
        )  # (batch_size, seq_len, self.context.context_size, self.context.hidden_dim)
        flatten = embeds.view(
            batch_size * seq_len, -1
        )  # (batch_size * seq_len, self.context.context_size * self.context.hidden_dim)
        logits = self.linear(flatten)  # (batch_size * seq_len, self.config.vocab_size)
        logits = logits.view(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, self.config.vocab_size)

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


# Register in order to be able to load the simple
# model from a string
transformers.AutoConfig.register(SimpleLinearLMConfig.model_type, SimpleLinearLMConfig)
transformers.AutoModel.register(SimpleLinearLMConfig, SimpleLinearLM)
transformers.AutoModelForCausalLM.register(SimpleLinearLMConfig, SimpleLinearLM)
