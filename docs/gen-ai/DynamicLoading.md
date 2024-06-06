Dynamic loading is a technique to inference very large model on a machine with limited GPU memory. The idea is to load only part of the model to GPU memory and run inference on the loaded part. Once the inference is done, the loaded part is released from GPU memory and the next part is loaded to GPU memory. This process is repeated until the whole model is processed.

The technique is available in both llama.cpp and [huggingface accelerate](https://huggingface.co/blog/accelerate-large-models). The GenAI model package should also support this technique.

## Update on 2024/05/30
Experiment over partial loading is done in PR #10. The main take-away are
- partial loading can gain acceleration from 1.03X to over 30X even not fully loading model to GPU.
- the main bottleneck is still memory traffic between CPU and GPU.
- larger blocks should have higher priority when deciding which block to be 'pin' to GPU memory.

The result can be found in [this report](DynamicLoadingReport.md)
