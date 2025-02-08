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

import english_words
import numpy as np
import transformers

from grpo_server import grpo_client, data
from grpo_server.data import *

words = list(english_words.get_english_words_set(["web2"], lower=True))


def word_included(word, s):
    return word in s


def zmean(lst):
    if len(lst) == 0:
        return 0
    return np.mean(lst)


async def main():
    async with grpo_client.GrpoClient("http://localhost:8000", "default_key") as client:

        model_name = "./models/SmolLM-135M-Instruct"
        # model_name = "./models/SmolLM-1.7B-Instruct"
        # Start training
        training_settings = TrainingSettings(
            model_id=model_name,
            max_completion_length=48,
            num_completions_per_prompt=16,
            training_batch_size=1,
            learning_rate=5e-6,
            max_steps=10000,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        status = await client.get_status()
        if status.training_settings != training_settings:
            if status.training_settings is not None:
                await client.stop()
            await client.start(training_settings)

        while True:

            n_words = np.random.randint(2, 6)
            chosen_words = set(np.random.choice(words, n_words))

            chat = [
                dict(
                    role="user",
                    content="Sort the following words alphabetically: "
                    + ",".join(chosen_words)
                    + ". Provide your answer by listing one word on each row. ONLY LIST THE WORDS. BE VERY CONCISE.",
                ),
            ]

            chat_tok = tokenizer.apply_chat_template(chat)
            chat_str = tokenizer.decode(chat_tok)
            # chat_retok = tokenizer(chat_str)
            # print(chat_tok, chat_str, chat_retok)

            sorted_words = sorted(chosen_words)

            completions = await client.get_completions(
                CompletionsRequest(prompt=chat_str)
            )

            print(completions.prompt)
            rewards = []
            for completion in completions.completions:
                print("==")
                print(completion)

                completion_words = [word.strip(",. \n") for word in completion.split()]
                correct_completion_words = [
                    word for word in completion_words if word in chosen_words
                ]

                reward = (
                    0.01
                    * zmean(
                        [
                            word_included(word.lower(), completion.lower())
                            for word in chosen_words
                        ]
                    )
                    + 0.1
                    * zmean([word_included(word, completion) for word in chosen_words])
                    + 1.0
                    * zmean(
                        [
                            any(row.startswith(word) for row in completion.split("\n"))
                            for word in chosen_words
                        ]
                    )
                    + 1.0
                    * zmean(
                        [
                            completion_word in chosen_words
                            for completion_word in completion_words
                        ]
                    )
                    + 10.0
                    * zmean(
                        [
                            correct_completion_words[i] < correct_completion_words[j]
                            for i in range(len(correct_completion_words))
                            for j in range(i)
                        ]
                    )
                )
                print("==", reward)
                rewards.append(reward)

            await client.submit_rewards(
                RewardsRequest.from_completions(completions, rewards)
            )

            # break

        # await client.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
