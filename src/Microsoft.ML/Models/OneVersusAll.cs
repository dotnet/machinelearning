// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using static Microsoft.ML.Runtime.EntryPoints.CommonInputs;

namespace Microsoft.ML.Models
{
    public sealed partial class OneVersusAll
    {
        /// <summary>
        /// One versus all learner (also known as One vs the rest) is a multiclass learner with
        /// strategy to fit one classifier per class. For each class it trains the provided 
        /// binary classifier against all the other classes. 
        /// See <a href="https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest">wikipedia</a> page.
        /// </summary>
        /// <param name="trainer">Underlying binary trainer</param>
        /// <param name="useProbabilities">"Use probabilities (vs. raw outputs) to identify top-score category</param>
        /// See  TrainOva <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/Scenarios/IrisPlantClassificationTests.cs">unit test</a>
        /// for an example on how to train a one versus all model.
        public static ILearningPipelineItem With(ITrainerInputWithLabel trainer, bool useProbabilities = true)
        {
            return new OvaPipelineItem(trainer, useProbabilities);
        }

        private class OvaPipelineItem : ILearningPipelineItem
        {
            private Var<IDataView> _data;
            private ITrainerInputWithLabel _trainer;
            private bool _useProbabilities;

            public OvaPipelineItem(ITrainerInputWithLabel trainer, bool useProbabilities)
            {
                _trainer = trainer;
                _useProbabilities = useProbabilities;
            }

            public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
            {
                using (var env = new TlcEnvironment())
                {
                    var subgraph = env.CreateExperiment();
                    subgraph.Add(_trainer);
                    var ova = new OneVersusAll();
                    if (previousStep != null)
                    {
                        if (!(previousStep is ILearningPipelineDataStep dataStep))
                        {
                            throw new InvalidOperationException($"{ nameof(OneVersusAll)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                        }

                        _data = dataStep.Data;
                        ova.TrainingData = dataStep.Data;
                        ova.UseProbabilities = _useProbabilities;
                        ova.Nodes = subgraph;
                    }
                    Output output = experiment.Add(ova);
                    return new OvaPipelineStep(output);
                }
            }

            public Var<IDataView> GetInputData() => _data;
        }

        private class OvaPipelineStep : ILearningPipelinePredictorStep
        {
            public OvaPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<IPredictorModel> Model { get; }
        }
    }
}
