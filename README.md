# grpo_server Control-flow-reversed remote grpo training

Basic architecture:

* start_service starts the application

* `grpo_server.grpo_service` provides a fastapi interface to get
  completions from and return rewards to.

* `grpo_server.grpo_queuer` provides a control-flow-reversed hacky
  dataset, goes between service and trainer (TODO: a proper implementation)

* `grpo_server.grpo_trainer` provides a control-flow-reversed hacky
  grpo trainer created by splitting the hf trainer funcs. (TODO: a proper implementation)

* `grpo_server.testing_utils` contains a really simple and fast
  language model to run tests using. Included in main body
  to allow access e.g. from jupyterlab

* `test_data` contains data used in the test, e.g., smolm135 tokenizer (incorporated
  in the simple linear model too)

Details:

* Sequencing: we give out tuple[str, int] (uuid of run id + gradient cycle)
  when completing and when having used results.

    * TODO: more control over when to include given input, now it's async (but queued)

Biggest TODOs:

* batches
* example task containers
* how to start main container with settings?
* checkpoint model, load checkpoints
* (continue training): either load clean model and peft it,
  or load an already pefted model and keep training that peft.
