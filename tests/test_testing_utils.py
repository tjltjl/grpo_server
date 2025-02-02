"""E2e learning test for grpo queuer.
"""
from grpo_server import testing_utils
import logging
import pytest
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture(scope="session")
def simple_problem():
    return testing_utils.SimpleProblem()

def test_rewards(simple_problem):
    assert simple_problem.calculate_rewards(["3434"], ["343434"]) == [0]
    assert simple_problem.calculate_rewards(["3434"], ["444444"]) == [2./3.]


def test_learns(simple_problem, tmp_path):
    """Test that the problem is ok, i.e., learning happens normally"""

    model, trainer = simple_problem.create_normal_model_and_trainer(tmp_path)

    assert not simple_problem.has_learned(model)
    trainer.train()
    assert simple_problem.has_learned(model)

# This test may deadlock; need a timeout.
@pytest.mark.timeout(12)
def test_server_learns_immediate(simple_problem, tmp_path):
    """Test that learning happens when run through queuer."""

    model, trainer, queuer = simple_problem.create_split_model_and_trainer_and_queuer(tmp_path)

    complete = False

    lock = threading.Lock()
    tasks = []

    def loop_thread():
        logger.debug("START LOOP THREAD")
        async def run_loop():
            nonlocal complete
            while not complete:
                with lock:
                    for task in tasks:
                        if task.done():
                            print("TASK RESULT", task.result())

                for row in simple_problem.dataset:
                    prompt = row["prompt"]
                    logger.debug("call get completions %s", prompt)
                    completions = await queuer.get_completions(dict(prompt=prompt))
                    logger.debug("got completions %s", completions)
                    rewards = simple_problem.calculate_rewards(
                        [completions["prompt"]] * len(completions["completions"]),
                        completions["completions"],
                    )
                    async def reward_setter(prompt, completions, rewards):
                        logger.debug("setting rewards: %s %s %s", prompt, completions, rewards)
                        await queuer.rewards(
                            dict(
                                prompt=completions["prompt"],
                                completions=completions["completions"],
                                rewards=rewards,
                                extra=completions["extra"],
                            ),
                        )

                    task = asyncio.create_task(reward_setter(prompt, completions, rewards))
                    with lock: tasks.append(task)
                    # await asyncio.sleep(0.01)
            logger.debug("loop_thread exiting")

        import asyncio
        asyncio.run(run_loop())


    thread = threading.Thread(target=loop_thread)
    thread.start()

    # assert not simple_problem.has_learned(model)
    trainer.train()
    complete = True
    # assert simple_problem.has_learned(model)

    thread.join(timeout=1)
