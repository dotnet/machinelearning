// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;

namespace Microsoft.ML.Benchmarks
{
    public class KMeansAndLogisticRegressionBench
    {
        private readonly string _dataPath = Program.GetInvariantCultureDataPath("adult.train");

        [Benchmark]
        public ParameterMixingCalibratedPredictor TrainKMeansAndLR()
        {
            var ml = new MLContext(seed: 1);
            // Pipeline

            var input = ml.Data.ReadFromTextFile(new[] {
                            new TextLoader.Column("Label", DataKind.R4, 14),
                            new TextLoader.Column("CatFeatures", DataKind.TX,
                                new [] {
                                    new TextLoader.Range() { Min = 1, Max = 1 },
                                    new TextLoader.Range() { Min = 3, Max = 3 },
                                    new TextLoader.Range() { Min = 5, Max = 9 },
                                    new TextLoader.Range() { Min = 13, Max = 13 }
                                }),
                            new TextLoader.Column("NumFeatures", DataKind.R4,
                                new [] {
                                    new TextLoader.Range() { Min = 0, Max = 0 },
                                    new TextLoader.Range() { Min = 2, Max = 2 },
                                    new TextLoader.Range() { Min = 4, Max = 4 },
                                    new TextLoader.Range() { Min = 10, Max = 12 }
                                }),
            }, _dataPath, s =>
            {
                s.HasHeader = true;
                s.Separator = ",";
            });

            var estimatorPipeline = ml.Transforms.Categorical.OneHotEncoding("CatFeatures")
                .Append(ml.Transforms.Normalize("NumFeatures"))
                .Append(ml.Transforms.Concatenate("Features", "NumFeatures", "CatFeatures"))
                .Append(ml.Clustering.Trainers.KMeans("Features"))
                .Append(ml.Transforms.Concatenate("Features", "Features", "Score"))
                .Append(ml.BinaryClassification.Trainers.LogisticRegression(advancedSettings: args => { args.EnforceNonNegativity = true; args.OptTol = 1e-3f; }));

            var model = estimatorPipeline.Fit(input);
            // Return the last model in the chain.
            return model.LastTransformer.Model;
        }
    }
}