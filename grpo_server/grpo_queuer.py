"""
The dataset source for grpo training
that implements the control loop reversal.
"""

from abc import ABC, abstractmethod
import asyncio
import datasets
import dill
import janus
import logging
import pydantic_settings
import queue
import traceback
import transformers
import trl.trainer.grpo_trainer
from typeguard import typechecked
import typing as t
from grpo_server import data

logger = logging.getLogger(__name__)
# pytest doesn't flush logger.debug()..
logdebug = logger.info

from grpo_server import grpo_trainer_reversed


class TrainingSettings(pydantic_settings.BaseSettings):
    # Model to start from
    model_id: str = "./test_data/simple_linear_40_5"
    max_completion_length: int = 6
    num_completions_per_prompt: int = 3

    training_batch_size: int = 1  # Larger needs debugging (padding)
    learning_rate: float = 5e-2
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1


class BaseQueuer(ABC):
    """The main training interface that the outside world connects to.

    The core interface of grpo_server.

    Two operations:

        prompt -> completions (answers)
        rewards -> ... (fed to training)
    """

    @abstractmethod
    async def get_completions(
        self, completions_request: data.CompletionsRequest
    ) -> data.CompletionsResponse:
        pass

    @abstractmethod
    async def rewards(
        self, rewards_request: data.RewardsRequest
    ) -> data.RewardsResponse:
        pass


def create_queuer(training_settings: TrainingSettings, output_dir: str):
    model = transformers.AutoModel.from_pretrained(training_settings.model_id)

    grpo_config = trl.trainer.grpo_trainer.GRPOConfig(
        # beta=0.1,
        per_device_train_batch_size=training_settings.training_batch_size,
        output_dir=output_dir,
        do_train=True,
        do_eval=False,
        learning_rate=training_settings.learning_rate,
        logging_steps=training_settings.logging_steps,
        gradient_accumulation_steps=training_settings.gradient_accumulation_steps,
        max_completion_length=training_settings.max_completion_length,
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

    queuer = GRPOQueuer(model)

    dataset = datasets.Dataset.from_generator(
        GeneratorWrapper(queuer.data_getter()), streaming=True
    )

    # tokenizer = transformers.AutoTokenizer.from_pretrained(training_settings.model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./test_data/smolm135_tokenizer"
    )

    trainer_params = dict(
        model=model,
        reward_funcs=[],
        args=grpo_config,
        train_dataset=dataset,
        # peft_config=peft.LoraConfig(task_type="CAUSAL_LM"),
        processing_class=tokenizer,
    )

    trainer = grpo_trainer_reversed.GRPOTrainerSplit(**trainer_params)  # type: ignore

    queuer.set_trainer(trainer)

    return queuer


class ActionItem(asyncio.Event):
    def __init__(self, callable):
        self.callable = callable
        self.event_loop = asyncio.get_running_loop()
        self.event = asyncio.Event()

    def run(self):
        self.result, return_value = self.callable()

        async def set_event():
            self.event.set()

        asyncio.run_coroutine_threadsafe(set_event(), self.event_loop)
        return return_value


class GRPOQueuer(BaseQueuer):
    """The main querier class.

    As a context manager, starts a training thread and stops it.
    """

    queue: janus.Queue[ActionItem]
    async_queue: t.Any
    sync_queue: t.Any

    model: t.Any
    trainer: t.Any

    def __init__(self, model):
        self.queue = janus.Queue()
        self.model = model
        self.async_queue = self.queue.async_q
        self.sync_queue = self.queue.sync_q

    # Avoid loops
    def set_trainer(self, trainer):
        self.trainer = trainer

    async def __aenter__(self):

        def loop():
            try:
                logdebug("START TRAINING LOOP")
                self.trainer.train()
            except:
                logdebug("TRAINING LOOP EXCEPTION")
                traceback.print_exc()
                raise
            logdebug("TRAINING LOOP EXIT")

        self.loop_task = asyncio.create_task(asyncio.to_thread(loop))

        return self

    async def __aexit__(self, exc_type, exc, tb):

        logdebug("AExit")

        def action() -> None:
            logdebug("AExit action")
            raise Exception("END")

        await self.async_queue.put(ActionItem(action))
        await asyncio.wait([self.loop_task])

    def is_alive(self):
        return not self.loop_task.done()

    def data_getter(self):
        queue = self.sync_queue
        while True:
            # Do not wait for rewards; wait for
            logdebug("data_getter: to get")
            item = queue.get()
            logdebug("data_getter: got")
            retval = item.run()
            logdebug("data_getter: run complete: %s", retval)
            if retval:
                yield retval
            # if queue.qsize() > 10:
            #    print("DROP ON THE FLOOR", item.data)
            # else:
            #    yield item

    @typechecked
    async def get_completions(
        self, completions_request: data.CompletionsRequest
    ) -> data.CompletionsResponse:
        assert self.trainer

        logdebug("get_completions: start %s", completions_request)

        # This is run inside data_getter
        @typechecked
        def action() -> tuple[data.CompletionsResponse, None]:
            completions = self.trainer.generate_completions(
                self.trainer.model,
                [dict(prompt=completions_request.prompt)],
            )
            assert len(completions) == 1
            logdebug("get_completions: gen returned %s", completions[0])
            return (
                data.CompletionsResponse(
                    prompt=completions_request.prompt,
                    completions=completions[0]["completions"],
                    completion_tokens=completions[0]["extra"]["prompt_completion_ids"],
                    model_version=("", 0),  # TODO
                ),
                None,
            )

        item = ActionItem(action)
        await self.async_queue.put(item)
        logdebug("get_completions: to wait")
        await item.event.wait()
        logdebug("get_completions: done %s", item.result)

        # Ensure response matches CompletionsResponse structure
        return item.result

    async def rewards(
        self, rewards_request: data.RewardsRequest
    ) -> data.RewardsResponse:
        logdebug("rewards: start %s", rewards_request)
        queue = self.async_queue

        # Pipe rewards straight to learning
        #
        def action():
            # It's a reward; return this data with the labels
            dataset_row = dict(
                prompt=rewards_request.prompt,
                rewards=rewards_request.rewards,
                extra=dict(prompt_completion_ids=rewards_request.completion_tokens),
            )
            logdebug("data_getter returning %s", dataset_row)
            return data.RewardsResponse(model_version=("", 0)), dataset_row

        item = ActionItem(action)
        await self.async_queue.put(item)
        logdebug("rewards: to wait")
        await item.event.wait()
        logdebug("rewards: done %s", item.result)
        return item.result

    def __getstate__(self):
        # Ensure no state is pickled
        output = self.__dict__.copy()
        # Ensure grpoqueuer doesn't try to serialize anything important
        output.pop("model", None)
        output.pop("trainer", None)
        output.pop("queue", None)
        return output

    def __setstate__(self, input_dict):
        self.__dict__.update(input_dict)
        self.queue = janus.Queue()
        self.model = None
        self.trainer = None


# Register with dill to allow pickling
dill.register(GRPOQueuer)


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
