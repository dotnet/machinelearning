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


    /// <summary>
    /// The <see cref="LearningPipeline"/> class is used to define the steps needed to perform a desired machine learning task.<para/>
    /// The steps are defined by adding a data loader (e.g. <see cref="TextLoader"/>) followed by zero or more transforms (e.g. <see cref="Microsoft.ML.Transforms.TextFeaturizer"/>) 
    /// and at most one trainer/learner (e.g. <see cref="Microsoft.ML.Trainers.FastTreeBinaryClassifier"/>) in the pipeline.
    /// 
    /// </summary>
    /// <example>
    /// <para/>
    /// For example,<para/>
    /// <code>
    /// var pipeline = new LearningPipeline();
    /// pipeline.Add(new TextLoader &lt;SentimentData&gt; (dataPath, separator: ","));
    /// pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
    /// pipeline.Add(new FastTreeBinaryClassifier());
    /// 
    /// var model = pipeline.Train&lt;SentimentData, SentimentPrediction&gt;();
    /// </code>
    /// </example>
    [DebuggerTypeProxy(typeof(LearningPipelineDebugProxy))]
    public class LearningPipeline : ICollection<ILearningPipelineItem>
    {
        private List<ILearningPipelineItem> Items { get; } = new List<ILearningPipelineItem>();
        private readonly int? _seed;
        private readonly int _conc;

        /// <summary>
        ///  Construct an empty <see cref="LearningPipeline"/> object.
        /// </summary>
        /// <param name="seed">Specify seed for random generator</param>
        /// <param name="conc">Specify concurrency factor (default value - autoselection)</param>
        public LearningPipeline(int? seed=null, int conc=0)
        {
            _seed = seed;
            _conc = conc;
        }

        /// <summary>
        /// Get the count of ML components in the <see cref="LearningPipeline"/> object
        /// </summary>
        public int Count => Items.Count;
        public bool IsReadOnly => false;

        /// <summary>
        /// Add a data loader, transform or trainer into the pipeline. 
        /// Possible data loader(s), transforms and trainers options are
        /// <para>
        /// Data Loader:
        ///     <see cref="Microsoft.ML.Data.TextLoader" />
        ///     etc.
        /// </para>
        /// <para>
        /// Transforms:
        ///     <see cref="Microsoft.ML.Transforms.Dictionarizer"/>,
        ///     <see cref="Microsoft.ML.Transforms.CategoricalOneHotVectorizer"/>
        ///     <see cref="Microsoft.ML.Transforms.MinMaxNormalizer"/>,
        ///     <see cref="Microsoft.ML.Transforms.ColumnCopier"/>,
        ///     <see cref="Microsoft.ML.Transforms.ColumnConcatenator"/>,
        ///     <see cref="Microsoft.ML.Transforms.TextFeaturizer"/>,
        ///     etc.
        /// </para>
        /// <para>
        /// Trainers:
        ///     <see cref="Microsoft.ML.Trainers.AveragedPerceptronBinaryClassifier"/>,
        ///     <see cref="Microsoft.ML.Trainers.LogisticRegressionClassifier"/>,
        ///     <see cref="Microsoft.ML.Trainers.StochasticDualCoordinateAscentClassifier"/>,
        ///     <see cref="Microsoft.ML.Trainers.FastTreeRegressor"/>,
        ///     etc.
        /// </para>
        /// For a complete list of transforms and trainers, please see "Microsoft.ML.Transforms" and "Microsoft.ML.Trainers" namespaces.
        /// </summary>
        /// <param name="item">Any ML component (data loader, transform or trainer) defined as <see cref="ILearningPipelineItem"/>.</param>
        public void Add(ILearningPipelineItem item) => Items.Add(item);

        /// <summary>
        /// Remove all the loaders/transforms/trainers from the pipeline.
        /// </summary>
        public void Clear() => Items.Clear();

        /// <summary>
        /// Check if a specific loader/transform/trainer is in the pipeline?
        /// </summary>
        /// <param name="item">Any ML component (data loader, transform or trainer) defined as <see cref="ILearningPipelineItem"/>.</param>
        /// <returns>true if item is found in the pipeline; otherwise, false.</returns>
        public bool Contains(ILearningPipelineItem item) => Items.Contains(item);

        /// <summary>
        /// Copy the pipeline items into an array.
        /// </summary>
        /// <param name="array">The one-dimensional Array that is the destination of the elements copied from.</param>
        /// <param name="arrayIndex">The zero-based index in <paramref name="array" /> at which copying begins.</param>
        public void CopyTo(ILearningPipelineItem[] array, int arrayIndex) => Items.CopyTo(array, arrayIndex);
        public IEnumerator<ILearningPipelineItem> GetEnumerator() => Items.GetEnumerator();

        /// <summary>
        /// Remove an item from the pipeline.
        /// </summary>
        /// <param name="item"><see cref="ILearningPipelineItem"/> to remove.</param>
        /// <returns>true if item was removed from the pipeline; otherwise, false.</returns>
        public bool Remove(ILearningPipelineItem item) => Items.Remove(item);
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Train the model using the ML components in the pipeline.
        /// </summary>
        /// <typeparam name="TInput">Type of data instances the model will be trained on. It's a custom type defined by the user according to the structure of data.
        /// <para/>
        /// Please see https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet/get-started/windows for more details on input type.
        /// </typeparam>
        /// <typeparam name="TOutput">Ouput type. The prediction will be return based on this type.
        /// Please see https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet/get-started/windows for more details on output type.
        /// </typeparam>
        /// <returns>PredictionModel object. This is the model object used for prediction on new instances. </returns>
        public PredictionModel<TInput, TOutput> Train<TInput, TOutput>()
            where TInput : class
            where TOutput : class, new()
        {
            using (var environment = new TlcEnvironment(seed:_seed, conc:_conc))
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

                        Var<IPredictorModel> predictorModel;
                        if (transformModels.Count != 0)
                        {
                            var localModelInput = new Transforms.ManyHeterogeneousModelCombiner
                            {
                                PredictorModel = predictorDataStep.Model,
                                TransformModels = new ArrayVar<ITransformModel>(transformModels.ToArray())
                            };
                            var localModelOutput = experiment.Add(localModelInput);
                            predictorModel = localModelOutput.PredictorModel;
                        }
                        else
                            predictorModel = predictorDataStep.Model;

                        var scorer = new Transforms.Scorer
                        {
                            PredictorModel = predictorModel
                        };

                        var scorerOutput = experiment.Add(scorer);
                        lastTransformModel = scorerOutput.ScoringTransform;
                        step = new ScorerPipelineStep(scorerOutput.ScoredData, scorerOutput.ScoringTransform);
                        transformModels.Clear();
                    }
                }

                if (transformModels.Count > 0)
                {
                    if (lastTransformModel != null)
                        transformModels.Insert(0, lastTransformModel);

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
