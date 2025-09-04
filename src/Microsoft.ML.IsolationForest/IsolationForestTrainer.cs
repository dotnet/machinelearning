// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.ML.IsolationForest
{
    /// <summary>
    /// Minimal Isolation Forest 'trainer' implemented as an IEstimator that appends Score and PredictedLabel.
    /// NOTE: uses CustomMapping internally (not persisted by Save()) – production version should wrap a proper ITransformer with serialization.
    /// Targets netstandard2.0.
    /// </summary>
    public sealed class IsolationForestTrainer : IEstimator<ITransformer>
    {
        private readonly MLContext _ml;
        private readonly Options _opts;

        public sealed class Options
        {
            public string FeatureColumnName { get; set; }
            public string OutputScoreColumnName { get; set; }
            public string OutputPredictedLabelColumnName { get; set; }
            public int Trees { get; set; }
            public int SampleSize { get; set; }
            public int Seed { get; set; }
            public double? Contamination { get; set; }
            public bool ParallelBuild { get; set; }
            public double? ThresholdOverride { get; set; }

            public Options()
            {
                FeatureColumnName = "Features";
                OutputScoreColumnName = "Score";
                OutputPredictedLabelColumnName = "PredictedLabel";
                Trees = 100;
                SampleSize = 256;
                Seed = 42;
                Contamination = 0.01;
                ParallelBuild = true;
                ThresholdOverride = null;
            }
        }


        public IsolationForestTrainer(MLContext mlContext, Options options = null)
        {
            if (mlContext == null)
            {
                throw new ArgumentNullException(nameof(mlContext));
            }

            _ml = mlContext;
            _opts = options ?? new Options();
        }

        public ITransformer Fit(IDataView input)
        {
            var feats = input.GetColumn<VBuffer<float>>(_opts.FeatureColumnName).ToArray();
            if (feats.Length == 0)
            {
                throw new InvalidOperationException("Empty training data.");
            }

            int d = feats[0].Length;
            double[][] x = feats.Select(v =>
            {
                var vals = v.DenseValues().ToArray();
                var row = new double[d];
                for (int i = 0; i < d; i++)
                {
                    row[i] = vals[i];
                }
                return row;
            }).ToArray();

            var model = new IsolationForestModel(_opts.Trees, _opts.SampleSize, _opts.Seed);
            model.Fit(x, contamination: _opts.Contamination, parallel: _opts.ParallelBuild);

            var mapper = _ml.Transforms.CustomMapping(
                (IsolationForestInput inp, IsolationForestOutput outp) =>
                {
                    var vals = inp.Features.DenseValues().ToArray();
                    var row = new double[vals.Length];
                    for (int i = 0; i < vals.Length; i++)
                    {
                        row[i] = vals[i];
                    }
                    float score = (float)model.Score(row);
                    bool pred = model.Predict(row, _opts.ThresholdOverride);
                    outp.Score = score;
                    outp.PredictedLabel = pred;
                },
                contractName: "IsolationForestMapper"
            );

            return mapper.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // Validate presence and shape of the feature column without TryFindColumn / Columns
            bool found = false;
            SchemaShape.Column featureCol = default(SchemaShape.Column);

            foreach (var c in inputSchema)
            {
                if (c.Name == _opts.FeatureColumnName)
                {
                    featureCol = c;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                throw new InvalidOperationException("Missing column '" + _opts.FeatureColumnName + "'.");
            }

            if (featureCol.ItemType != NumberDataViewType.Single)
            {
                throw new InvalidOperationException("'" + _opts.FeatureColumnName + "' must have item type Single (float).");
            }

            if (featureCol.Kind != SchemaShape.Column.VectorKind.Vector)
            {
                throw new InvalidOperationException("'" + _opts.FeatureColumnName + "' must be a vector<float>.");
            }

            // Delegate schema construction to a no-op CustomMapping. This reliably appends
            // columns defined by IsolationForestOutput without touching internal APIs.
            var schemaProbe = _ml.Transforms.CustomMapping<IsolationForestInput, IsolationForestOutput>(
                (inp, outp) =>
                {
                    // no-op: schema is inferred from IsolationForestOutput's public properties
                },
                contractName: "IsolationForestSchemaProbe"
            );

            return schemaProbe.GetOutputSchema(inputSchema);
        }

        public sealed class IsolationForestInput
        {
            [VectorType]
            public VBuffer<float> Features { get; set; }
        }

        public sealed class IsolationForestOutput
        {
            public float Score { get; set; }
            public bool PredictedLabel { get; set; }
        }
    }
}
