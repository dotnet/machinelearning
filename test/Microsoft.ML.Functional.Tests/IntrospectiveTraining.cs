// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class IntrospectiveTraining : BaseTestClass
    {
        public IntrospectiveTraining(ITestOutputHelper output): base(output)
        {
        }

        /// <summary>
        /// Introspective Training: Map hashed values back to the original value.
        /// </summary>
        [Fact]
        public void InspectSlotNamesForReversibleHash()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Load the Adult dataset.
            var data = mlContext.Data.LoadFromTextFile<Adult>(GetDataPath(TestDatasets.adult.trainFilename),
                hasHeader: TestDatasets.adult.fileHasHeader,
                separatorChar: TestDatasets.adult.fileSeparator);

            // Create the learning pipeline.
            var pipeline = mlContext.Transforms.Concatenate("NumericalFeatures", Adult.NumericalFeatures)
                .Append(mlContext.Transforms.Concatenate("CategoricalFeatures", Adult.CategoricalFeatures))
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("CategoricalFeatures", hashBits: 8, // get collisions!
                    invertHash: -1, outputKind: OneHotEncodingTransformer.OutputKind.Bag));

            // Train the model.
            var model = pipeline.Fit(data);

            // Transform the data.
            var transformedData = model.Transform(data);

            // Verify that the slotnames cane be used to backtrack by confirming that 
            //  all unique values in the input data are in the output data slot names.
            // First get a list of the unique values.
            VBuffer<ReadOnlyMemory<char>> categoricalSlotNames = new VBuffer<ReadOnlyMemory<char>>();
            transformedData.Schema["CategoricalFeatures"].GetSlotNames(ref categoricalSlotNames);
            var uniqueValues = new HashSet<string>();
            foreach (var slotName in categoricalSlotNames.GetValues())
            {
                var slotNameString = slotName.ToString();
                if (slotNameString.StartsWith('{'))
                {
                    // Values look like this: {3:Exec-managerial,2:Widowed}.
                    slotNameString = slotNameString.Substring(1, slotNameString.Length - 2);
                    foreach (var name in slotNameString.Split(','))
                        uniqueValues.Add(name);
                }
                else
                    uniqueValues.Add(slotNameString);
            }

            // Now validate that all values in the dataset are there
            var transformedRows = mlContext.Data.CreateEnumerable<Adult>(data, false);
            foreach (var row in transformedRows)
            {
                for (int i = 0; i < Adult.CategoricalFeatures.Length; i++)
                {
                    // Fetch the categorical value.
                    string value = (string) row.GetType().GetProperty(Adult.CategoricalFeatures[i]).GetValue(row, null);
                    Assert.Contains($"{i}:{value}", uniqueValues);
                }
            }

            float x = (float)double.MinValue;
            Output.WriteLine($"{x}");
        }

        //private void BooYa()
        //{
        //    // Create the learning pipeline
        //    var nestedPipeline = mlContext.Transforms.Concatenate("NumericalFeatures", Adult.NumericalFeatures)
        //        .Append(mlContext.Transforms.Concatenate("CategoricalFeatures", Adult.CategoricalFeatures))
        //        .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("CategoricalFeatures",
        //            invertHash: 2, outputKind: OneHotEncodingTransformer.OutputKind.Bag)
        //        .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoricalFeatures"))
        //        .Append(mlContext.BinaryClassification.Trainers.LogisticRegression()));

        //    // Train the model.
        //    var nestedModel = nestedPipeline.Fit(data);
        //    var nestedPredictor = nestedModel.LastTransformer.LastTransformer;
        //    var nestedTransformedData = nestedModel.Transform(data);

        //    Assert.Equal(predictor.Model.SubModel.Bias, nestedPredictor.Model.SubModel.Bias);
        //        int nFeatures = predictor.Model.SubModel.Weights.Count;
        //        for (int i = 0; i<nFeatures; i++ )
        //            Assert.Equal(predictor.Model.SubModel.Weights[i], nestedPredictor.Model.SubModel.Weights[i]);

        //        var transformedRows = mlContext.Data.CreateEnumerable<BinaryPrediction>(transformedData, false).ToArray();
        //    var nestedTransformedRows = mlContext.Data.CreateEnumerable<BinaryPrediction>(nestedTransformedData, false).ToArray();
        //        for (int i = 0; i<transformedRows.Length; i++)
        //            Assert.Equal(transformedRows[i].Score, nestedTransformedRows[i].Score);
        //}

        //private class BinaryPrediction
        //{
        //    public float Score { get; set; }
        //    public float Probability { get; set; }
        //}
    }
}