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
        /// One-versus-all, OvA, learner (also known as One-vs.-rest, "OvR") is a multi-class learner 
        /// with the strategy to fit one binary classifier per class in the dataset.
        /// It trains the provided binary classifier for each class against the other classes, where the current 
        /// class is treated as the positive labels and examples in other classes are treated as the negative classes.
        /// See <a href="https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest">wikipedia</a> page.
        /// </summary>
        ///<example>
        /// In order to use it all you need to do is add it to pipeline as regular learner:
        /// 
        /// pipeline.Add(OneVersusAll.With(new StochasticDualCoordinateAscentBinaryClassifier()));
        /// </example>
        /// <remarks>
        /// The base trainer must be a binary classifier. To check the available binary classifiers, type BinaryClassifiers,
        /// and look at the available binary learners as suggested by IntelliSense.
        /// </remarks>
        /// <param name="trainer">Underlying binary trainer</param>
        /// <param name="useProbabilities">"Use probabilities (vs. raw outputs) to identify top-score category.
        /// By specifying it to false, you can tell One-versus-all to not use the probabilities but instead
        /// the raw uncalibrated scores from each predictor.This is generally not recommended, since these quantities
        /// are not meant to be comparable from one predictor to another, unlike calibrated probabilities.</param>
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
