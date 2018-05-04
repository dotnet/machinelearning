// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Microsoft.ML
{
    class ScorerPipelineStep : ILearningPipelineDataStep
    {
        public ScorerPipelineStep(Var<IDataView> data, Var<ITransformModel> model)
        {
            Data = data;
            Model = model;
        }

        public Var<IDataView> Data { get; }
        public Var<ITransformModel> Model { get; }
    }

    [DebuggerTypeProxy(typeof(LearningPipelineDebugProxy))]
    public class LearningPipeline : ICollection<ILearningPipelineItem>
    {
        private List<ILearningPipelineItem> Items { get; } = new List<ILearningPipelineItem>();

        public LearningPipeline()
        {
        }

        public int Count => Items.Count;
        public bool IsReadOnly => false;
        public void Add(ILearningPipelineItem item) => Items.Add(item);
        public void Clear() => Items.Clear();
        public bool Contains(ILearningPipelineItem item) => Items.Contains(item);
        public void CopyTo(ILearningPipelineItem[] array, int arrayIndex) => Items.CopyTo(array, arrayIndex);
        public IEnumerator<ILearningPipelineItem> GetEnumerator() => Items.GetEnumerator();
        public bool Remove(ILearningPipelineItem item) => Items.Remove(item);
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public PredictionModel<TInput, TOutput> Train<TInput, TOutput>()
            where TInput : class
            where TOutput : class, new()
        {

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineStep step = null;
                List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
                List<Var<ITransformModel>> transformModels = new List<Var<ITransformModel>>();
                Var<ITransformModel> lastTransformModel = null;

                foreach (ILearningPipelineItem currentItem in this)
                {
                    if (currentItem is ILearningPipelineLoader loader)
                        loaders.Add(loader);

                    step = currentItem.ApplyStep(step, experiment);
                    if (step is ILearningPipelineDataStep dataStep && dataStep.Model != null)
                        transformModels.Add(dataStep.Model);
                     
                    else if (step is ILearningPipelinePredictorStep predictorDataStep)
                    {
                        if (lastTransformModel != null)
                            transformModels.Insert(0, lastTransformModel);

                        var localModelInput = new Transforms.ManyHeterogeneousModelCombiner
                        {
                            PredictorModel = predictorDataStep.Model,
                            TransformModels = new ArrayVar<ITransformModel>(transformModels.ToArray())
                        };

                        var localModelOutput = experiment.Add(localModelInput);

                        var scorer = new Transforms.Scorer
                        {
                            PredictorModel = localModelOutput.PredictorModel
                        };

                        var scorerOutput = experiment.Add(scorer);
                        lastTransformModel = scorerOutput.ScoringTransform;
                        step = new ScorerPipelineStep(scorerOutput.ScoredData, scorerOutput.ScoringTransform);
                        transformModels.Clear();
                    }
                }

                if (transformModels.Count > 0)
                {
                    transformModels.Insert(0,lastTransformModel);
                    var modelInput = new Transforms.ModelCombiner
                    {
                        Models = new ArrayVar<ITransformModel>(transformModels.ToArray())
                    };

                    var modelOutput = experiment.Add(modelInput);
                    lastTransformModel = modelOutput.OutputModel;
                }

                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                {
                    loader.SetInput(environment, experiment);
                }
                experiment.Run();

                ITransformModel model = experiment.GetOutput(lastTransformModel);
                BatchPredictionEngine<TInput, TOutput> predictor;
                using (var memoryStream = new MemoryStream())
                {
                    model.Save(environment, memoryStream);

                    memoryStream.Position = 0;

                    predictor = environment.CreateBatchPredictionEngine<TInput, TOutput>(memoryStream);

                    return new PredictionModel<TInput, TOutput>(predictor, memoryStream);
                }
            }
        }

        /// <summary>
        /// Executes a pipeline and returns the resulting data.
        /// </summary>
        /// <returns>
        /// The IDataView that was returned by the pipeline.
        /// </returns>
        internal IDataView Execute(IHostEnvironment environment)
        {
            Experiment experiment = environment.CreateExperiment();
            ILearningPipelineStep step = null;
            List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
            foreach (ILearningPipelineItem currentItem in this)
            {
                if (currentItem is ILearningPipelineLoader loader)
                    loaders.Add(loader);

                step = currentItem.ApplyStep(step, experiment);
            }

            if (!(step is ILearningPipelineDataStep endDataStep))
            {
                throw new InvalidOperationException($"{nameof(LearningPipeline)}.{nameof(Execute)} must have a Data step as the last step.");
            }

            experiment.Compile();
            foreach (ILearningPipelineLoader loader in loaders)
            {
                loader.SetInput(environment, experiment);
            }
            experiment.Run();

            return experiment.GetOutput(endDataStep.Data);
        }
    }
}
