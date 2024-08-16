# Microsoft.ML.GenAI.Phi
Torchsharp implementation of Microsoft phi-series models for GenAI

## Supported list
The following phi-models are supported and tested:
- [x] [Phi-2](https://huggingface.co/microsoft/phi-2)
- [x] [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [x] [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [x] [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)
- [x] [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
- [ ] [Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct)
- [ ] [Phi-3-small-128k-instruct](https://huggingface.co/microsoft/Phi-3-small-128k-instruct)
- [ ] [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-large-4k-instruct)

## Getting Started with Semantic Kernel

### Download model weight (e.g. phi-3-mini-4k-instruct) from Huggingface
```bash
## make sure you have lfs installed
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
```

### Load model
```csharp
var weightFolder = "/path/to/Phi-3-mini-4k-instruct";
var configName = "config.json";
var config = JsonSerializier.Deserialize<Phi3Config>(File.ReadAllText(Path.Combine(weightFolder, configName)));
var model = new Phi3ForCasualLM(config);

// load tokenizer
var tokenizerModelName = "tokenizer.model";
var tokenizer = Phi3TokenizerHelper.FromPretrained(Path.Combine(weightFolder, tokenizerModelName));

// load weight
model.LoadSafeTensors(weightFolder);

// initialize device
var device = "cuda";
if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
}


// create causal language model pipeline
var pipeline = new CausalLMPipeline<Tokenizer, Phi3ForCausalLM>(tokenizer, model, device);
```

### Add pipeline as `IChatCompletionService` to sematic kernel
```csharp
var kernel = Kernel.CreateBuilder()
    .AddGenAIChatCompletion(pipeline)
    .Build();
```

### Chat with the model
```csharp
var chatService = kernel.GetRequiredService<IChatCompletionService>();
var chatHistory = new ChatHistory();
chatHistory.AddSystemMessage("you are a helpful assistant");
chatHistory.AddUserMessage("write a C# program to calculate the factorial of a number");
await foreach (var response in chatService.GetStreamingChatMessageContentsAsync(chatHistory))
{
    Console.Write(response);
}
```

## Getting started with AutoGen.Net
### Follow the same steps download model weight and load model
### Create `Phi3Agent` from pipeline
```csharp
var agent = new Phi3Agent(pipeline, name: "assistant")
                .RegisterPrintMessage();
```

### Chat with the model
```csharp
var task = """
write a C# program to calculate the factorial of a number
""";

await agent.SendAsync(task);
```

### More examples
Please refer to [Microsoft.ML.GenAI.Samples](./../../docs/samples/Microsoft.ML.GenAI.Samples/) for more examples.

## Dynamic loading
It's recommended to run model inference on GPU, which requires at least 8GB of GPU memory for phi-3-mini-4k-instruct model if fully loaded.

If your GPU memory is not enough, you can choose to dynamically load the model weight to GPU memory. Here is how it works behind the scene:
- when initializing the model, the size of each layer is calculated and stored in a dictionary
- when loading the model weight, each layer is assigned to a device (CPU or GPU) based on the size of the layer and the remaining memory of the device. If there is no enough memory on the device, the layer is loaded to CPU memory.
- when inference, the layer which is loaded to CPU memory is moved to GPU memory before the inference and moved back to CPU memory after the inference.

Here is how to enable dynamic loading of model:
### Step 1: Infer the size of each layer
You can infer the size of each layer using `InferDeviceMapForEachLayer` API. The `deviceMap` will be a key-value dictionary, where the key is the layer name and the value is the device name (e.g. "cuda" or "cpu").

```csharp
// manually set up the available memory on each device
var deviceSizeMap = new Dictionary<string, long>
    {
        ["cuda"] = modelSizeOnCudaInGB * 1L * 1024 * 1024 * 1024,
        ["cpu"] = modelSizeOnMemoryInGB * 1L * 1024 * 1024 * 1024,
        ["disk"] = modelSizeOnDiskInGB * 1L * 1024 * 1024 * 1024,
    };

var deviceMap = model.InferDeviceMapForEachLayer(
        devices: ["cuda", "cpu", "disk"],
        deviceSizeMapInByte: deviceSizeMap);
```

### Step 2: Load model weights using `ToDynamicLoadingModel` API
Once the `deviceMap` is calculated, you can pass it to `ToDynamicLoadingModel` api to load the model weight.

```csharp
model = model.ToDynamicLoadingModel(deviceMap, "cuda");
```
