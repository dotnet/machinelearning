The GenAI project will be a collection of popular open source AI models. It will be organized in the following structure:

- Microsoft.ML.GenAI.Core: the core library for GenAI project, it contains the fundamental contracts or classes like `CausalLanguageModel` and `CausalLMPipeline`
- Microsoft.ML.GenAI.{ModelName}: the implementation of a specific model, which includes the model configuration, causal lm model implementation (like `Phi3ForCausalLM`) and tokenizer implementation if any. In the first stage, we plan to provide the following models:
  - Microsoft.ML.GenAI.Phi: the implementation of Phi-series model
  - Microsoft.ML.GenAI.LLaMA: the implementation of LLaMA-series model
  - Microsoft.ML.GenAI.StableDiffusion: the implementation of Stable Diffusion model