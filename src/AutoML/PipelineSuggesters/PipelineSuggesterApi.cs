using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class PipelineSuggesterApi
    {
        // local
        public static Pipeline GetPipeline(TaskKind task, IDataView data, string label)
        {
            var mlContext = new MLContext();
            var availableTransforms = TransformInferenceApi.InferTransforms(mlContext, data, label);
            var availableTrainers = RecipeInference.AllowedTrainers(mlContext, task, 1);
            var pipeline = new InferredPipeline(availableTransforms, availableTrainers.First(), mlContext);
            return pipeline.ToPipeline();
        }
    }
}
