using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// Performs Train-Test on a pipeline.
    /// </summary>
    public sealed partial class TrainTestEvaluator
    {
        /// <summary>
        /// Performs train-test on a pipeline.
        /// </summary>
        /// <typeparam name="TInput">Class type that represents input schema.</typeparam>
        /// <typeparam name="TOutput">Class type that represents prediction schema.</typeparam>
        /// <param name="pipeline">Machine learning pipeline that contains <see cref="ILearningPipelineLoader"/>, 
        /// transforms and at least one trainer.</param>
        /// <param name="testData"><see cref="ILearningPipelineLoader"/> that represents the test dataset.</param>
        /// <returns>Metrics and predictor model.</returns>
        public TrainTestEvaluatorOutput<TInput, TOutput> TrainTestEvaluate<TInput, TOutput>(LearningPipeline pipeline, ILearningPipelineLoader testData)
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
                
                TrainingData = (loaders[0].ApplyStep(null, experiment) as ILearningPipelineDataStep).Data;
                TestingData = (testData.ApplyStep(null, experiment) as ILearningPipelineDataStep).Data;
                Nodes = subGraph;
                TransformModel = null;
                Inputs.Data = firstTransform.GetInputData();
                Outputs.TransformModel = lastTransformModel;
                Outputs.UseTransformModel = true;
                var crossValidateOutput = experiment.Add(this);
                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                    loader.SetInput(environment, experiment);

                testData.SetInput(environment, experiment);

                experiment.Run();

                var trainTestOutput = new TrainTestEvaluatorOutput<TInput, TOutput>();
                if (Kind == MacroUtilsTrainerKinds.SignatureBinaryClassifierTrainer)
                {
                    trainTestOutput.BinaryClassificationMetrics = BinaryClassificationMetrics.FromMetrics(
                        environment,
                        experiment.GetOutput(crossValidateOutput.OverallMetrics),
                        experiment.GetOutput(crossValidateOutput.ConfusionMatrix)).FirstOrDefault();
                }
                else if (Kind == MacroUtilsTrainerKinds.SignatureMultiClassClassifierTrainer)
                {
                    trainTestOutput.ClassificationMetrics = ClassificationMetrics.FromMetrics(
                        environment,
                        experiment.GetOutput(crossValidateOutput.OverallMetrics),
                        experiment.GetOutput(crossValidateOutput.ConfusionMatrix)).FirstOrDefault();
                }
                else if (Kind == MacroUtilsTrainerKinds.SignatureRegressorTrainer)
                {
                    trainTestOutput.RegressionMetrics = RegressionMetrics.FromOverallMetrics(
                        environment,
                        experiment.GetOutput(crossValidateOutput.OverallMetrics)).FirstOrDefault();
                }
                else
                {
                    //Implement metrics for ranking, clustering and anomaly detection.
                    throw Contracts.Except($"{Kind.ToString()} is not supported at the moment.");
                }

                ITransformModel model = experiment.GetOutput(crossValidateOutput.TransformModel);
                BatchPredictionEngine<TInput, TOutput> predictor;
                using (var memoryStream = new MemoryStream())
                {
                    model.Save(environment, memoryStream);

                    memoryStream.Position = 0;

                    predictor = environment.CreateBatchPredictionEngine<TInput, TOutput>(memoryStream);

                    trainTestOutput.PredictorModels = new PredictionModel<TInput, TOutput>(predictor, memoryStream);
                }
                
                return trainTestOutput;
            }
        }
    }

    public class TrainTestEvaluatorOutput<TInput, TOutput>
            where TInput : class
            where TOutput : class, new()
    {
        public BinaryClassificationMetrics BinaryClassificationMetrics;
        public ClassificationMetrics ClassificationMetrics;
        public RegressionMetrics RegressionMetrics;
        public PredictionModel<TInput, TOutput> PredictorModels;

        //REVIEW: Add warnings and per instance results and implement 
        //metrics for ranking, clustering and anomaly detection.
    }
}
