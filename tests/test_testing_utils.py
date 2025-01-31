from grpo_server import testing_utils
import pytest

@pytest.fixture(scope="session")
def simple_problem():
    return testing_utils.SimpleProblem()

def test_rewards(simple_problem):
    assert simple_problem.calculate_rewards(["3434"], ["343434"]) == [0]
    assert simple_problem.calculate_rewards(["3434"], ["444444"]) == [2./3.]


def test_learns(simple_problem, tmp_path):

    model, trainer = simple_problem.create_normal_model_and_trainer(tmp_path)

    assert not simple_problem.has_learned(model)
    trainer.train()
    assert simple_problem.has_learned(model)
