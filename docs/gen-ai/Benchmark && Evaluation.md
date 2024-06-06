It's critical to evaluate the performance of the GenAI model once it's available. The evaluation && benchmark will be on two-fold:
- evaluation on various eval datasets: this is to make sure our implementation is correct and the model is working as expected comparing to python-implemented model.
- benchmark on inference speed: this is to make sure the model can be used in real-time applications.

This document will cover the topic of how to evaluate the model on various eval datasets.

## How we evaluate the model
To get the most comparable result with other llms, we evaluate the model in the same way as [Open LLM leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), which uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as the evaluation framework.

For the details of which evaluation datasets are used, please refer to the [Open LLM leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

Because `lm-evaluation-harness` is written in python, there is no way to directly use it in .NET. Therefore we use the following steps as a workaround:
- in C#, start a openai chat completion service server with the model we want to evaluate.
- in python, use `lm-evaluation-harness` to evaluate the model using openai mode.