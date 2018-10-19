// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Legacy.Models
{
    public sealed partial class OnnxConverter
    {
        /// <summary>
        /// <a href="https://onnx.ai/">ONNX</a> is an intermediate representation format
        /// for machine learning models.
        /// </summary>
        /// <remarks>
        /// <para>It is used to make models portable such that you can
        /// train a model using a toolkit and run it in another tookit's runtime, for example,
        /// you can create a model using ML.NET, export it to an ONNX-ML model file,
        /// then load and run that ONNX-ML model in Windows ML, on an UWP Windows 10 app.</para>
        ///
        /// <para>This API converts an ML.NET model to ONNX-ML format by inspecting the transform pipeline
        /// from the end, checking for components that know how to save themselves as ONNX.
        /// The first item in the transform pipeline that does not know how to save itself
        /// as ONNX, is considered the "input" to the ONNX pipeline. (Ideally this would be the
        /// original loader itself, but this may not be possible if the user used unsavable
        /// transforms in defining the pipe.) All the columns in the source that are a type the
        /// ONNX knows how to deal with will be tracked. Intermediate transformations of the
        /// data appearing as new columns will appear in the output block of the ONNX, with names
        /// derived from the corresponding column names. The ONNX JSON will be serialized to a
        /// path defined through the Json option.</para>
        ///
        /// <para>This API supports the following arguments:</para>
        /// <list type="bullet">
        /// <item><description><see cref="Onnx"/> indicates the file to write the ONNX protocol buffer file to. This is required.</description></item>
        /// <item><description><see cref="Json"/> indicates the file to write the JSON representation of the ONNX model. This is optional.</description></item>
        /// <item><description><see cref="Name"/> indicates the name property in the ONNX model. If left unspecified, it will
        /// be the extension-less name of the file specified in the onnx indicates the protocol buffer file
        /// to write the ONNX representation to.</description></item>
        /// <item><description><see cref="Domain"/> indicates the domain name of the model. ONNX uses reverse domain name space indicators.
        /// For example com.microsoft.cognitiveservices. This is a required field.</description></item>
        /// <item><description><see cref="InputsToDrop"/> is a string array of input column names to omit from the input mapping.
        /// A common scenario might be to drop the label column, for instance, since it may not be practically
        /// useful for the pipeline. Note that any columns depending on these naturally cannot be saved.</description></item>
        /// <item><description><see cref="OutputsToDrop"/> is similar, except for the output schema. Note that the pipeline handler
        /// is currently not intelligent enough to drop intermediate calculations that produce this value: this will
        /// merely omit that value from the actual output.</description></item>
        /// </list>
        ///
        /// <para>Transforms that can be exported to ONNX</para>
        /// <list type="number">
        /// <item><description>Concat</description></item>
        /// <item><description>KeyToVector</description></item>
        /// <item><description>NAReplace</description></item>
        /// <item><description>Normalize</description></item>
        /// <item><description>Term</description></item>
        /// <item><description>Categorical</description></item>
        /// </list>
        ///
        /// <para>Learners that can be exported to ONNX</para>
        /// <list type="number">
        /// <item><description>FastTree</description></item>
        /// <item><description>LightGBM</description></item>
        /// <item><description>Logistic Regression</description></item>
        /// </list>
        ///
        /// <para>See <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxTests.cs">OnnxTests.cs</a>
        /// for an example on how to train a model and then convert that model to ONNX.</para>
        /// </remarks>
        /// <param name="model">Model that needs to be converted to ONNX format.</param>
        public void Convert(PredictionModel model)
        {
            using (var environment = new ConsoleEnvironment())
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
