using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Models
{
    public sealed partial class OnnxConverter
    {
        /// <summary>
        /// Converts the model to ONNX format.
        /// </summary>
        /// <param name="model">Model that needs to be converted to ONNX format.</param>
        public void Convert(PredictionModel model)
        {
            using (var environment = new TlcEnvironment())
            {
                environment.CheckValue(model, nameof(model));

                Experiment experiment = environment.CreateExperiment();
                experiment.Add(this);
                experiment.Compile();
                experiment.SetInput(Model, new PredictorModel(environment, model.PredictorModel));
                experiment.Run();
            }
        }
    }
}
