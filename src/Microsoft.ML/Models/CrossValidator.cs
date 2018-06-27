// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// Performs cross-validation on a pipeline.
    /// </summary>
    public sealed partial class CrossValidator
    {
        /// <summary>
        /// Performs cross validation on a pipeline.
        /// </summary>
        /// <typeparam name="TInput">Class type that represents input schema.</typeparam>
        /// <typeparam name="TOutput">Class type that represents prediction schema.</typeparam>
        /// <param name="pipeline">Machine learning pipeline may contain loader, transforms and at least one trainer.</param>
        /// <returns>List containing metrics and predictor model for each fold</returns>
        public CrossValidationOutput<TInput, TOutput> CrossValidate<TInput, TOutput>(LearningPipeline pipeline)
            where TInput : class
            where TOutput : class, new()
        {
            using (var environment = new TlcEnvironment())
            {
                Experiment subGraph = environment.CreateExperiment();
                ILearningPipelineStep step = null;
                List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
                List<Var<ITransformModel>> transformModels = new List<Var<ITransformModel>>();
                Var<ITransformModel> lastTransformModel = null;
                Var<IDataView> firstPipelineDataStep = null;
                Var<IPredictorModel> firstModel = null;
                ILearningPipelineItem firstTransform = null;
                foreach (ILearningPipelineItem currentItem in pipeline)
                {
                    if (currentItem is ILearningPipelineLoader loader)
                    {
                        loaders.Add(loader);
                        continue;
                    }

                    step = currentItem.ApplyStep(step, subGraph);

                    if (step is ILearningPipelineDataStep dataStep && dataStep.Model != null)
                    {
                        transformModels.Add(dataStep.Model);
                        if (firstPipelineDataStep == null)
                        {
                            firstPipelineDataStep = dataStep.Data;
                            firstTransform = currentItem;
                        }
                    }
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
                            var localModelOutput = subGraph.Add(localModelInput);
                            predictorModel = localModelOutput.PredictorModel;
                        }
                        else
                            predictorModel = predictorDataStep.Model;
                        firstModel = predictorModel;

                        var scorer = new Transforms.Scorer
                        {
                            PredictorModel = predictorModel
                        };

                        var scorerOutput = subGraph.Add(scorer);
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

                    var modelOutput = subGraph.Add(modelInput);
                    lastTransformModel = modelOutput.OutputModel;
                }

                var experiment = environment.CreateExperiment();
                var importTextOutput = loaders[0].ApplyStep(null, experiment);

                Data = (importTextOutput as ILearningPipelineDataStep).Data;
                Nodes = subGraph;
                TransformModel = null;
                Inputs.Data = firstTransform.GetInputData();
                Outputs.PredictorModel = null;
                Outputs.TransformModel = lastTransformModel;
                var crossValidateOutput = experiment.Add(this);
                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                {
                    loader.SetInput(environment, experiment);
                }

                experiment.Run();

                var cvOutput = new CrossValidationOutput<TInput, TOutput>();
                cvOutput.PredictorModels = new PredictionModel<TInput, TOutput>[NumFolds];

                for (int Index = 0; Index < NumFolds; Index++)
                {

                    if (Kind == MacroUtilsTrainerKinds.SignatureBinaryClassifierTrainer)
                    {
                        cvOutput.BinaryClassificationMetrics = BinaryClassificationMetrics.FromMetrics(
                            environment,
                            experiment.GetOutput(crossValidateOutput.OverallMetrics),
                            experiment.GetOutput(crossValidateOutput.ConfusionMatrix), 2);
                    }
                    else if (Kind == MacroUtilsTrainerKinds.SignatureMultiClassClassifierTrainer)
                    {
                        cvOutput.ClassificationMetrics = ClassificationMetrics.FromMetrics(
                            environment,
                            experiment.GetOutput(crossValidateOutput.OverallMetrics),
                            experiment.GetOutput(crossValidateOutput.ConfusionMatrix), 2);
                    }
                    else if (Kind == MacroUtilsTrainerKinds.SignatureRegressorTrainer)
                    {
                        cvOutput.RegressionMetrics = RegressionMetrics.FromOverallMetrics(
                            environment,
                            experiment.GetOutput(crossValidateOutput.OverallMetrics));
                    }
                    else if (Kind == MacroUtilsTrainerKinds.SignatureClusteringTrainer)
                    {
                        cvOutput.ClusterMetrics = ClusterMetrics.FromOverallMetrics(
                            environment,
                            experiment.GetOutput(crossValidateOutput.OverallMetrics));
                    }
                    else
                    {
                        //Implement metrics for ranking, clustering and anomaly detection.
                        throw Contracts.Except($"{Kind.ToString()} is not supported at the moment.");
                    }

                    ITransformModel model = experiment.GetOutput(crossValidateOutput.TransformModel[Index]);
                    BatchPredictionEngine<TInput, TOutput> predictor;
                    using (var memoryStream = new MemoryStream())
                    {
                        model.Save(environment, memoryStream);

                        memoryStream.Position = 0;

                        predictor = environment.CreateBatchPredictionEngine<TInput, TOutput>(memoryStream);

                        cvOutput.PredictorModels[Index] = new PredictionModel<TInput, TOutput>(predictor, memoryStream);
                    }
                }

                return cvOutput;
            }
        }
    }

    public class CrossValidationOutput<TInput, TOutput>
            where TInput : class
            where TOutput : class, new()
    {
        public List<BinaryClassificationMetrics> BinaryClassificationMetrics;
        public List<ClassificationMetrics> ClassificationMetrics;
        public List<RegressionMetrics> RegressionMetrics;
        public List<ClusterMetrics> ClusterMetrics;
        public PredictionModel<TInput, TOutput>[] PredictorModels;

        //REVIEW: Add warnings and per instance results and implement 
        //metrics for ranking, clustering and anomaly detection.
    }
}
