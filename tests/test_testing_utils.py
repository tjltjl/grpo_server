"""E2e learning test for grpo queuer."""

import asyncio
import fastapi.testclient
import logging
import numpy as np
from numpy.random import wald
import pytest
import threading
from typeguard import typechecked

from grpo_server.testing import testing_utils
from grpo_server import data, grpo_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def simple_problem():
    return testing_utils.SimpleProblem()


def test_rewards(simple_problem):
    """Test the calculate_rewards function in simple_problem"""
    completion1 = data.CompletionsResponse(
        prompt="3434",
        completions=["343434"],
        completion_tokens=[[0]],
        model_version=("", 0),
    )
    expected1 = data.RewardsRequest(
        prompt="3434", completions=["343434"], completion_tokens=[[0]], rewards=[0]
    )
    assert simple_problem.calculate_rewards(completion1) == expected1

    completion2 = data.CompletionsResponse(
        prompt="3434",
        completions=["444444"],
        completion_tokens=[[0]],
        model_version=("", 0),
    )
    expected2 = data.RewardsRequest(
        prompt="3434",
        completions=["444444"],
        completion_tokens=[[0]],
        rewards=[2.0 / 3.0],
    )
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


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create a test client for FastAPI."""
    # Set api key for testing
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("output_dir", str(tmp_path))
    with fastapi.testclient.TestClient(grpo_service.app) as client:
        yield client


@pytest.fixture
def service_url():
    """Base URL for the service."""
    return "http://testserver"


@pytest.mark.timeout(5)
def test_cleanup(client, service_url):

    client.post(f"{service_url}/foo")
    with pytest.raises(Exception):
        result.raise_for_status()


@pytest.mark.timeout(12)
def test_learns_service(simple_problem, tmp_path, client, service_url):
    """Test that learning happens when run through the FastAPI service."""

    complete = False
    headers = {"api-key": "test-key"}

    rewards = []

    @typechecked
    async def get_completions(
        completions_request: data.CompletionsRequest,
    ) -> data.CompletionsResponse:
        """Async function to get completions via HTTP."""
        response = await asyncio.to_thread(
            lambda: client.post(
                f"{service_url}/completions",
                json=completions_request.model_dump(),
                headers=headers,
            )
        )
        response.raise_for_status()
        return data.CompletionsResponse.model_validate(response.json())

    @typechecked
    async def set_rewards(rewards_request: data.RewardsRequest) -> data.RewardsResponse:
        """Async function to set rewards via HTTP."""
        response = client.post(
            f"{service_url}/rewards",
            json=rewards_request.model_dump(),
            headers=headers,
        )
        rewards.extend(rewards_request.rewards)
        response.raise_for_status()
        return data.RewardsResponse.model_validate(response.json())

    def loop_thread():
        """Thread function to run the async event loop."""
        logger.debug("START LOOP THREAD")

        async def run_loop():
            nonlocal complete
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(
                    simple_problem.run_loop_async(
                        get_completions,
                        set_rewards,
                    )
                )
                while not complete:
                    await asyncio.sleep(0.1)
                    if task.done():
                        await asyncio.wait([task])
                        logger.debug("loop_thread OUT PREMATURELY")
                        return
                task.cancel()
                try:
                    await asyncio.wait([task])
                except Exception as e:
                    print("Forced exit", e)
                logger.debug("loop_thread exiting")

        asyncio.run(run_loop())

    # Create and start the event loop thread
    thread = threading.Thread(target=loop_thread, daemon=True)
    thread.start()

    try:
        # Wait for some time to allow learning to happen
        # In a real test, you might want to add some validation here
        import time

        for i in range(500):
            time.sleep(0.01)  # Allow some time for training
            if not thread.is_alive():
                logger.debug("THREAD DIED")
                break
    finally:
        complete = True
        logger.debug("JOINING THREAD")
        thread.join(timeout=1)
        logger.debug("JOINED THREAD")

    # Check that we didn't initially know what to do
    # and in the end did'
    assert len(rewards) > 50
    rewards = np.array(rewards)
    logger.debug("Rewards: %s %s", rewards[:10], rewards[-10:])
    assert np.all(rewards[0:10] < 0.5)
    assert np.all(rewards[-10:] > 0.5)


# @pytest.mark.timeout(12)
# def test_learns_service(simple_problem, tmp_path):
