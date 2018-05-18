using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Models
{
    public sealed partial class BinaryCrossValidator
    {
        public void CrossValidate<TInput, TOutput>(LearningPipeline pipeline) 
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
                Var<IDataView> firstInput = null;
                Var<IPredictorModel> firstModel = null;
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
                        if (firstInput == null)
                            firstInput = dataStep.Data;
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

                        var scorer = new Transforms.Scorer
                        {
                            PredictorModel = predictorModel
                        };
                        firstModel = predictorModel;
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
                
                var crossValidateBinary = new ML.Models.BinaryCrossValidator
                {
                    Data = (importTextOutput as ILearningPipelineDataStep).Data,
                    Nodes = subGraph
                };
                crossValidateBinary.Inputs.Data = firstInput;
                crossValidateBinary.Outputs.Model = firstModel;
                var crossValidateOutput = experiment.Add(crossValidateBinary);

                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                {
                    loader.SetInput(environment, experiment);
                }
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics[0]);
                ITransformModel model = experiment.GetOutput(lastTransformModel);
                BatchPredictionEngine<TInput, TOutput> predictor;
                using (var memoryStream = new MemoryStream())
                {
                    model.Save(environment, memoryStream);

                    memoryStream.Position = 0;

                    predictor = environment.CreateBatchPredictionEngine<TInput, TOutput>(memoryStream);

                    //return new PredictionModel<TInput, TOutput>(predictor, memoryStream);
                }
            }
        }
    }
}
