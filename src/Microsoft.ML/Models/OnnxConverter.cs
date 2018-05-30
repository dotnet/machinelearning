// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

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
                experiment.SetInput(Model, model.PredictorModel);
                experiment.Run();
            }
        }
    }
}
