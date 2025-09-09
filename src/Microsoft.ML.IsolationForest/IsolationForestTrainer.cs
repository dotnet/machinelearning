// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

#nullable enable

namespace Microsoft.ML.IsolationForest
{
    /// <summary>
    /// Trains a pure C# IsolationForestModel, then exposes a lightweight ML.NET pipeline
    /// that scores rows via CustomMapping:
    ///   Score = model.ScaledScore0To100WithDirectionality(...)
    ///   PredictedLabel = Score >= MatchThreshold
    ///
    /// Notes:
    /// - CustomMapping delegates aren’t persisted by ML.NET model save APIs.
    /// - Directionality (which features should be high/low) is supplied by the caller
    ///   via SetDirectionality(...). No domain logic lives here.
    /// </summary>
    public sealed class IsolationForestTrainer(IsolationForestTrainer.Options? options = null)
    {
        public sealed class Options
        {
            public Options()
            {
                // Defaults moved here to avoid "MSML_NonInstanceInitializers"
                FeatureColumnName = "Features";
                ScoreColumnName = "Score";
                PredictedLabelColumnName = "PredictedLabel";
                Trees = 100;
                SampleSize = 256;
                Seed = 2024;
                MatchThreshold = 0f;
                DebugScores = false;
            }

            public string FeatureColumnName { get; set; }
            public string ScoreColumnName { get; set; }
            public string PredictedLabelColumnName { get; set; }
            public int Trees { get; set; }
            public int SampleSize { get; set; }
            public int Seed { get; set; }
            public float MatchThreshold { get; set; }
            public bool DebugScores { get; set; }
        }

        // POCOs bound by CustomMapping (column names must match)
        public sealed class MapInput
        {
            [VectorType]
            public float[]? Features { get; set; }
        }

        public sealed class MapOutput
        {
            public float Score { get; set; }
            public bool PredictedLabel { get; set; }
        }

        private readonly Options _options = options ?? new Options();
        private IsolationForestModel? _model;

        // Learned during Fit
        private double[] _featureMedians = [];

        // Provided by caller (domain/app layer)
        private HashSet<int> _shouldBeHighIdx = [];
        private HashSet<int>? _shouldBeLowIdx; // optional

        /// <summary>
        /// Supply directionality after you know the feature index order (e.g., from slot names).
        /// Leave a set null/empty if you have no rule for that side.
        /// </summary>
        public void SetDirectionality(HashSet<int>? shouldBeHigh, HashSet<int>? shouldBeLow = null)
        {
            _shouldBeHighIdx = shouldBeHigh ?? [];
            _shouldBeLowIdx = shouldBeLow;
        }

        /// <summary>Fit the pure C# IsolationForestModel from an IDataView. Call before CreatePipeline.</summary>
        public void Fit(IDataView data, MLContext ml)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            if (ml is null) throw new ArgumentNullException(nameof(ml));

            // Resolve the feature column and validate type
            var featCol = data.Schema[_options.FeatureColumnName];
            if (featCol.Type is not VectorDataViewType vtype || !vtype.ItemType.Equals(NumberDataViewType.Single))
            {
                throw new ArgumentException(
                    $"Column '{_options.FeatureColumnName}' must be a vector of Single.",
                    nameof(data));
            }

            // Gather training rows into double[][]
            var x = new List<double[]>();
            using var cursor = data.GetRowCursor(data.Schema);
            var getter = cursor.GetGetter<VBuffer<float>>(featCol);
            VBuffer<float> buf = default;

            while (cursor.MoveNext())
            {
                getter.Invoke(ref buf);

                var fl = new float[buf.Length];
                buf.CopyTo(fl); // supports dense/sparse

                var dbl = new double[fl.Length];
                for (var i = 0; i < fl.Length; i++)
                {
                    dbl[i] = fl[i];
                }

                x.Add(dbl);
            }

            if (x.Count == 0)
            {
                throw new InvalidOperationException("Training data is empty.");
            }

            // Train the forest
            _model = new IsolationForestModel(
                nTrees: _options.Trees,
                sampleSize: _options.SampleSize,
                seed: _options.Seed);

            _model.Fit(x, parallel: true);

            // Compute per-feature medians for directionality scaler
            var d = x[0].Length;
            _featureMedians = new double[d];
            var col = new double[x.Count];
            for (var j = 0; j < d; j++)
            {
                for (var i = 0; i < x.Count; i++) col[i] = x[i][j];
                Array.Sort(col);
                var n = col.Length;
                _featureMedians[j] = (n % 2 == 1) ? col[n / 2] : 0.5 * (col[n / 2 - 1] + col[n / 2]);
            }
        }

        /// <summary>
        /// Build an ML.NET pipeline that:
        ///   - copies the user's feature column to "Features",
        ///   - runs CustomMapping to compute Score/PredictedLabel via the trained model with directionality,
        ///   - copies outputs to requested column names.
        /// </summary>
        public IEstimator<ITransformer> CreatePipeline(MLContext ml)
        {
            if (ml == null) throw new ArgumentNullException(nameof(ml));
            if (_model == null) throw new InvalidOperationException("Call Fit(...) before CreatePipeline(...).");

            // 1) Ensure a column literally named "Features" for CustomMapping binding
            IEstimator<ITransformer> pipe = ml.Transforms.CopyColumns(
                outputColumnName: nameof(MapInput.Features),
                inputColumnName: _options.FeatureColumnName);

            // Capture for mapping
            var medians = _featureMedians;
            var hiIdx = _shouldBeHighIdx;
            var lowIdx = _shouldBeLowIdx; // may be null

            // 2) CustomMapping: compute Score (0..100 match) & PredictedLabel using the trained model.
            pipe = pipe.Append(
                ml.Transforms.CustomMapping<MapInput, MapOutput>(
                    (input, output) =>
                    {
                        var feats = input.Features ?? [];
                        var row = new double[feats.Length];
                        for (var i = 0; i < feats.Length; i++) row[i] = feats[i];

                        var match = (float)_model.ScaledScore0To100WithDirectionality(
                            row,
                            featureMedians: medians,
                            shouldBeHighIdx: hiIdx,
                            shouldBeLowIdx: lowIdx,
                            clipTo100: true);

                        if (_options.DebugScores)
                        {
                            var dec = _model.DecisionFunction(row);
                            var norm = _model.NormalizedAveragePathLength(row);
                            System.Diagnostics.Debug.WriteLine(
                                $"IsolationForestTrainer DEBUG: linDir={match:F6}  dec={dec:F6}  normAPH={norm:F6}");
                        }

                        output.Score = match;
                        output.PredictedLabel = match >= _options.MatchThreshold;
                    },
                    contractName: "IsolationForestCustomMapping"
                )
            );

            // 3) Copy to user-requested output names if they differ
            if (_options.ScoreColumnName != nameof(MapOutput.Score))
            {
                pipe = pipe.Append(ml.Transforms.CopyColumns(
                    outputColumnName: _options.ScoreColumnName,
                    inputColumnName: nameof(MapOutput.Score)));
            }

            if (_options.PredictedLabelColumnName != nameof(MapOutput.PredictedLabel))
            {
                pipe = pipe.Append(ml.Transforms.CopyColumns(
                    outputColumnName: _options.PredictedLabelColumnName,
                    inputColumnName: nameof(MapOutput.PredictedLabel)));
            }

            return pipe;
        }

        /// <summary>Convenience: Fit + CreatePipeline + Fit the pipeline (returns ready-to-use transformer).</summary>
        public ITransformer FitAndCreateTransformer(MLContext ml, IDataView train)
        {
            Fit(train, ml);
            var pipe = CreatePipeline(ml);
            return pipe.Fit(train); // CustomMapping is stateless
        }
    }
}
