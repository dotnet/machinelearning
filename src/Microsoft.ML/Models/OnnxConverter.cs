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
        ///
        /// This API converts the model to ONNX format by inspecting the transform pipeline 
        /// from the end, checking for components that know how to save themselves as ONNX. 
        /// The first data view in the transform pipeline that does not know how to save itself 
        /// as ONNX, is considered the "input" to the ONNX pipeline. (Ideally this would be the 
        /// original loader itself, but this may not be possible if the user used unsavable 
        /// transforms in defining the pipe.) All the columns in the source that are a type the 
        /// ONNX knows what to deal with will be tracked. Intermediate transformations of the 
        /// data appearing as new columns will appear in the output block of the ONNX, with names 
        /// derived from the corresponding column names. The ONNX json will be serialized to a 
        /// path defined through the Json option.
        ///
        /// This API supports the following arguments:
        /// <see cref="Onnx"/> indicates the file to write the ONNX protocol buffer file to.
        /// <see cref="Json"/> indicates the file to write the JSON representation of the ONNX model.
        /// <see cref="Name"/> indicates the name property in the ONNX model. If left unspecified, it will 
        /// be the extension-less name of the file specified in the onnx indicates the protocol buffer file
        /// to write the ONNX representation to.
        /// <see cref="Domain"/> indicates the domain name of the model. We use reverse domain name space indicators.
        /// For example com.microsoft.cognitiveservices. This is a required field.
        /// <see cref="InputsToDrop"/> is a string array of input column names to omit from the input mapping.
        /// A common scenario might be to drop the label column, for instance, since it may not be practically
        /// useful for the pipeline. Note that any columns depending on these naturally cannot be saved.
        /// <see cref="OutputsToDrop"/> is similar, except for the output schema. Note that the pipeline handler 
        /// is currently not intelligent enough to drop intermediate calculations that produce this value: this will
        /// merely omit that value from the actual output.
        /// 
        /// Transforms that can be exported to ONNX
        /// 1. Concat
        /// 2. KeyToVector
        /// 3. NAReplace
        /// 4. Normalize
        /// 5. Term
        /// 6. Categorical
        /// 
        /// Learners that can be exported to ONNX
        /// 1. FastTree
        /// 2. LightGBM
        /// 3. LibSVM
        /// 4. Multi Class Logistic Regression
        /// 5. Logistic Regression
        /// 
        /// See <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxTests.cs"/>
        /// for an example on how to train a model and then convert that model to ONNX.
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
