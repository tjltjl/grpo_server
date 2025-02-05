"""
The dataset source for grpo training
that implements the control loop reversal.
"""

from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import datasets
import dill
import logging
import pydantic_settings
from queue import Queue
import shutil
import tempfile
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


from grpo_server.data import TrainingSettings


class BaseQueuer(ABC):
    """The main training interface that the outside world connects to.

    The core interface of grpo_server.

    Two operations:

        prompt -> completions (answers)
        rewards -> ... (fed to training)
    """

    # TODO async
    @abstractmethod
    def create_snapshot(self, zip_path):
        pass

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
        max_steps=training_settings.max_steps,
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
        self.error_class = None
        self.error: None | type = None
        self.result = None

    async def _set_event(self):
        self.event.set()

    def return_error(self, error_class, error):
        self.error_class = error_class
        self.error = error
        asyncio.run_coroutine_threadsafe(self._set_event(), self.event_loop)

    def run(self):
        "Called in the training thread"
        self.result, return_value = self.callable()

        asyncio.run_coroutine_threadsafe(self._set_event(), self.event_loop)
        return return_value

    def raise_from_error(self):
        "Called in the server thread to raise exception"
        if self.error:
            logdebug("RAISING FROM ERROR TO SOURCE %s %s", self.error_class, self.error)
            assert self.error_class
            raise self.error_class(self.error)


class StopTrainingException(Exception):
    pass


class GRPOQueuer(BaseQueuer):
    """The main querier class.

    As a context manager, starts a training thread and stops it.
    ```
       async with queuer.context():
           ...
    ```
    """

    queue: Queue[ActionItem]
    model: t.Any
    trainer: t.Any

    def __init__(self, model):
        self.queue = Queue()
        self.model = model
        self.exited = False

    # Avoid loops
    def set_trainer(self, trainer):
        self.trainer = trainer

    @asynccontextmanager
    async def context(self):

        def loop():
            try:
                logdebug("START TRAINING LOOP")
                self.trainer.train()
            except StopTrainingException:
                logdebug("TRAINING LOOP AEXIT")
            except:
                logdebug("TRAINING LOOP EXCEPTION")
                traceback.print_exc()
                raise
            logdebug("TRAINING LOOP EXIT")
            self.exited = True
            # Flush queue
            while True:
                try:
                    logdebug("flushing queue...")
                    entry = self.queue.get(timeout=0.1)
                except:
                    logdebug("queue empty")
                    return
                logdebug("Setting return_error %s", entry)
                entry.return_error(StopTrainingException, "Training loop exited")

        async with asyncio.TaskGroup() as tg:
            self.loop_task = tg.create_task(asyncio.to_thread(loop))

            yield

            logdebug("AExit")

            def action() -> None:
                logdebug("AExit action")
                raise StopTrainingException("END")

            self.queue.put_nowait(ActionItem(action))

    def is_alive(self):
        return not self.loop_task.done()

    def create_snapshot(self, zip_path):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            assert zip_path.endswith(".zip")
            shutil.make_archive(zip_path[:-4], "zip", path)

    def data_getter(self):
        """Main control flow reversal loop.

        Yields dataset rows for training.

        While doing the next row, accepts completion
        requests and runs them on the model.
        """
        queue = self.queue
        while True:
            # Do not wait for rewards; wait for
            logdebug("data_getter: to get")
            item = queue.get()
            if self.exited:
                item.return_error(StopTrainingException, "Training loop exited")
                continue
            logdebug("data_getter: got")
            retval = item.run()
            logdebug("data_getter: run complete: %s", retval)
            if retval:
                yield retval

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
        self.queue.put_nowait(item)
        logdebug("get_completions: to wait")
        await item.event.wait()
        logdebug("get_completions: done %s %s", item.error, item.result)
        item.raise_from_error()

        # Ensure response matches CompletionsResponse structure
        assert item.result
        return item.result

    async def rewards(
        self, rewards_request: data.RewardsRequest
    ) -> data.RewardsResponse:
        # logdebug("rewards: start %s", rewards_request)
        logdebug("rewards: start")

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
        self.queue.put_nowait(item)
        logdebug("rewards: to wait")
        await item.event.wait()
        logdebug("rewards: done %s %s", item.error, item.result)
        item.raise_from_error()
        assert item.result
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
        self.queue = Queue()
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
