using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.ML.GenAI.Phi;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;

namespace Microsoft.ML.GenAI.Samples.Phi3Mini;

public class AutoGenSample
{
    public static async Task RunAsync()
    {
        var device = "cuda";
        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var weightFolder = @"C:\Users\xiaoyuz\source\repos\Phi-3-medium-4k-instruct";
        var pipeline = Utils.LoadPhi3Mini4KFromFolder(weightFolder, device: device, quantizeToInt8: true);

        // agent
        var agent = new Phi3Agent(pipeline, "assistant")
            .RegisterPrintMessage();
        var question = @"write a C# program to calculate the factorial of a number";

        // chat with the assistant
        await agent.SendAsync(question);
    }
}
