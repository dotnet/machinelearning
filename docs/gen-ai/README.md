This folder contains the design doc for GenAI Model package

### Basic
- [Package Structure](./Package%20Structure.md): the structure of GenAI Model package
- [Usage](./Usage.md): how to use the model from GenAI Model package
- [Benchmark && Evaluation](./Benchmark%20&&%20Evaluation.md): how to evaluate the model from GenAI Model package

### Contracts && API
- [CausalLMPipeline](./CausalLMPipeline.md)
- [CausalLMModelInput and CausalLMModelOutput](./CausalLanguageModel.md)
- [Tokenizer](./Tokenizer.md)

### Need further investigation
- [Dynamic loading](./DynamicLoading.md): load only part of model to GPU when gpu memory is limited. We explore the result w/o dynamic loading in [this report](./DynamicLoadingReport.md)
- Improve loading speed: I notice that the model loading speed from disk to memory is slower in torchsharp than what it is in huggingface. Need to investigate the reason and improve the loading speed
- Quantization: quantize the model to reduce the model size and improve the inference speed
