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
    public sealed partial class CrossValidator
    {
        public PredictionModel<TInput, TOutput> CrossValidate<TInput, TOutput>(LearningPipeline pipeline) 
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
                Var<IDataView> lastData = null;
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
                        lastData = scorerOutput.ScoredData;
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
                    lastData = modelOutput.Data;
                }

                var experiment = environment.CreateExperiment();
                var importTextOutput = loaders[0].ApplyStep(null, experiment);

                Data = (importTextOutput as ILearningPipelineDataStep).Data;
                Nodes = subGraph;
                TransformModel = null;
                Inputs.Data = firstTransform.GetInputData();
                Outputs.Model = null;
                Outputs.TransformModel = lastTransformModel;
                Outputs.TransformData = lastData;
                Outputs.UseTransformModel = true;
                var crossValidateOutput = experiment.Add(this);
                experiment.Compile();
                foreach (ILearningPipelineLoader loader in loaders)
                {
                    loader.SetInput(environment, experiment);
                }

                experiment.Run();
                ITransformModel model = experiment.GetOutput(crossValidateOutput.TransformModel[0]);
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
    }
}
