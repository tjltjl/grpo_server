
"""
The dataset source for grpo training
that implements the control loop reversal.
"""
import asyncio
import dill
import janus
import logging
import queue
from typeguard import typechecked
import typing as t

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# pytest doesn't flush logger.debug()..
logdebug = logger.info

class ActionItem(asyncio.Event):
    def __init__(self, callable):
        self.callable = callable
        self.event_loop = asyncio.get_running_loop()
        self.event = asyncio.Event()

    def run(self):
        self.result, return_value = self.callable()

        async def set_event():
            self.event.set()

        asyncio.run_coroutine_threadsafe(set_event(),
            self.event_loop
        )
        return return_value

class PromptDict(t.TypedDict):
    prompt: str

class CompletionDict(t.TypedDict):
    prompt: str
    completions: list[str]
    extra: t.Any

class RewardDict(t.TypedDict):
    prompt: str
    completions: list[str]
    rewards: list[str]
    extra: t.Any

class GRPOQueuer:
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
    async def get_completions(self, prompt: PromptDict) -> CompletionDict:
        assert self.trainer

        logdebug("get_completions: start %s", prompt)

        # This is run inside data_getter
        @typechecked
        def action() -> tuple[CompletionDict, None]:
            completions = self.trainer.generate_completions(
                self.trainer.model,
                [prompt],
            )
            assert len(completions) == 1
            logdebug("get_completions: gen returned %s", completions[0])
            return completions[0], None

        item = ActionItem(action)
        await self.async_queue.put(item)
        logdebug("get_completions: to wait")
        await item.event.wait()
        logdebug("get_completions: done %s", item.result)

        # Ensure response matches CompletionDict structure
        return item.result

    async def rewards(self, rewards: list[RewardDict]) -> str:
        logdebug("rewards: start %s", rewards)
        queue = self.async_queue
        # Pipe rewards straight to learning
        #
        def action():
            # It's a reward; return this data with the labels
            dataset_row = rewards
            logdebug("data_getter returning %s", dataset_row)
            return "DONE", dataset_row

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
