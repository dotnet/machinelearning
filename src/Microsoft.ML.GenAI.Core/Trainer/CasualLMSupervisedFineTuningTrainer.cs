// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.Extensions.Logging;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core.Trainer;

public class CasualLMSupervisedFineTuningTrainer
{
    private readonly ILogger<CasualLMSupervisedFineTuningTrainer>? _logger;
    private readonly ICausalLMPipeline _pipeline;

    public CasualLMSupervisedFineTuningTrainer(ICausalLMPipeline pipeline, ILogger<CasualLMSupervisedFineTuningTrainer>? logger = null)
    {
        _logger = logger;
        _pipeline = pipeline;
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public async IAsyncEnumerable<ICausalLMPipeline> TrainAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        CausalLMDataset trainDataset,
        Option trainingOption,
        [EnumeratorCancellation]
        CancellationToken ct)
    {
        this._logger?.LogInformation("Start training...");
        var batches = trainDataset.Chunk(trainingOption.BatchSize);
        var optimizer = new Adam(_pipeline.Model.parameters(), lr: trainingOption.LearningRate);
        var device = torch.device(trainingOption.Device);

        for (int i = 0; i < trainingOption.Epoch; i++)
        {
            this._logger?.LogInformation($"Epoch {i + 1}/{trainingOption.Epoch}");
            var losses = new List<float>();
            foreach (var batch in batches)
            {
                if (ct.IsCancellationRequested)
                {
                    yield break;
                }
                var scope = NewDisposeScope();
                // merge items in batch
                var inputIds = torch.cat(batch.Select(x => x.InputIds).ToArray(), 1).to(device);
                var attentionMask = torch.cat(batch.Select(x => x.AttentionMask!).ToArray(), 1).to(device);
                var labels = torch.cat(batch.Select(x => x.Labels!).ToArray(), 1).to(device);
                // Forward the model
                var output = _pipeline.Model.forward(new CausalLMModelInput(inputIds, attentionMask: attentionMask, labels: labels, useCache: false));
                // Calculate loss
                var loss = output.Loss;
                // Backward the model
                optimizer.zero_grad();
                loss!.backward();
                optimizer.step();

                losses.Add(loss.data<float>().ToArray()[0]);

                // dispose loss
                loss.Dispose();

                // dispose output
                output.LastHiddenState.Dispose();
                output.Logits!.Dispose();
                inputIds.Dispose();
                attentionMask.Dispose();

                scope.Dispose();
            }

            _logger?.LogInformation($"Epoch {i + 1} loss: {losses.Average()}");

            yield return _pipeline;
        }
    }


    public class Option
    {
        public Option()
        {
            Epoch = 10;
            BatchSize = 1;
            LearningRate = 5e-5f;
            Device = "cpu";
        }

        public int Epoch { get; set; }

        public int BatchSize { get; set; }

        public float LearningRate { get; set; }

        public string Device { get; set; }
    }
}
