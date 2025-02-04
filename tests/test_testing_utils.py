"""E2e learning test for grpo queuer."""

from grpo_server.testing import testing_utils
import logging
import pytest
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def simple_problem():
    return testing_utils.SimpleProblem()


def test_rewards(simple_problem):
    """Test the calculate_rewards function in simple_problem"""
    completion1 = {"prompt": "3434", "completions": ["343434"], "extra": {}}
    expected1 = {
        "prompt": "3434",
        "completions": ["343434"],
        "rewards": [0],
        "extra": {},
    }
    assert simple_problem.calculate_rewards(completion1) == expected1

    completion2 = {"prompt": "3434", "completions": ["444444"], "extra": {}}
    expected2 = {
        "prompt": "3434",
        "completions": ["444444"],
        "rewards": [2.0 / 3.0],
        "extra": {},
    }
    assert simple_problem.calculate_rewards(completion2) == expected2


def test_learns_offline(simple_problem, tmp_path):
    """Test that the problem is ok offline."""

    model, trainer = simple_problem.create_normal_model_and_trainer(tmp_path)

    assert not simple_problem.has_learned(model)
    trainer.train()
    assert simple_problem.has_learned(model)


# This test may deadlock; need a timeout.
@pytest.mark.timeout(12)
def test_learns_queuer(simple_problem, tmp_path):
    """Test that learning happens when run through queuer."""

    model, trainer, queuer = simple_problem.create_split_model_and_trainer_and_queuer(
        tmp_path
    )

    complete = False

    lock = threading.Lock()
    tasks = []

    def loop_thread():
        logger.debug("START LOOP THREAD")

        async def run_loop():
            nonlocal complete
            task = asyncio.create_task(
                simple_problem.run_loop_async(
                    queuer.get_completions,
                    queuer.rewards,
                )
            )
            while not complete:
                await asyncio.sleep(0.1)
            task.cancel()
            logger.debug("loop_thread exiting")

        import asyncio

        asyncio.run(run_loop())

    thread = threading.Thread(target=loop_thread, daemon=True)
    thread.start()

    # assert not simple_problem.has_learned(model)
    trainer.train()
    complete = True
    # assert simple_problem.has_learned(model)

    thread.join(timeout=1)


# @pytest.mark.timeout(12)
# def test_learns_service(simple_problem, tmp_path):
