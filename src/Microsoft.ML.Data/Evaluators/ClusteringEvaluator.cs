// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(typeof(ClusteringEvaluator), typeof(ClusteringEvaluator), typeof(ClusteringEvaluator.Arguments), typeof(SignatureEvaluator),
    "Clustering Evaluator", ClusteringEvaluator.LoadName, "Clustering")]

[assembly: LoadableClass(typeof(ClusteringMamlEvaluator), typeof(ClusteringMamlEvaluator), typeof(ClusteringMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Clustering Evaluator", ClusteringEvaluator.LoadName, "Clustering")]

// This is for deserialization of the per-instance transform.
[assembly: LoadableClass(typeof(ClusteringPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", ClusteringPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed class ClusteringEvaluator : RowToRowEvaluatorBase<ClusteringEvaluator.Aggregator>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Calculate DBI? (time-consuming unsupervised metric)",
                ShortName = "dbi")]
            public bool CalculateDbi = false;
        }

        public const string LoadName = "ClusteringEvaluator";

        public const string Nmi = "NMI";
        public const string AvgMinScore = "AvgMinScore";
        public const string Dbi = "DBI";

        private readonly bool _calculateDbi;

        public ClusteringEvaluator(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            Host.AssertValue(args, "args");

            _calculateDbi = args.CalculateDbi;
        }

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            ColumnType type;
            if (schema.Label != null && (type = schema.Label.Type) != NumberType.Float && type.KeyCount == 0)
            {
                throw Host.Except("Clustering evaluator: label column '{0}' type must be {1} or Key of known cardinality." +
                    " Provide a correct label column, or none: it is optional.",
                    schema.Label.Name, NumberType.Float);
            }

            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            type = score.Type;
            if (!type.IsKnownSizeVector || type.ItemType != NumberType.Float)
                throw Host.Except("Scores column '{0}' type must be a float vector of known size", score.Name);
        }

        protected override void CheckCustomColumnTypesCore(RoleMappedSchema schema)
        {
            if (_calculateDbi)
            {
                Host.AssertValue(schema.Feature);
                var t = schema.Feature.Type;
                if (!t.IsKnownSizeVector || t.ItemType != NumberType.Float)
                {
                    throw Host.Except("Features column '{0}' type must be {1} vector of known-size",
                        schema.Feature.Name, NumberType.Float);
                }
            }
        }

        protected override Func<int, bool> GetActiveColsCore(RoleMappedSchema schema)
        {
            var pred = base.GetActiveColsCore(schema);
            // We also need the features column for dbi calculation.
            Host.Assert(!_calculateDbi || schema.Feature != null);
            return i => _calculateDbi && i == schema.Feature.Index || pred(i);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            Host.AssertValue(schema);
            Host.Assert(!_calculateDbi || (schema.Feature != null && schema.Feature.Type.IsKnownSizeVector));
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Host.Assert(score.Type.VectorSize > 0);
            int numClusters = score.Type.VectorSize;
            return new Aggregator(Host, schema.Feature, numClusters, _calculateDbi, schema.Weight != null, stratName);
        }

        protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            int numClusters = scoreInfo.Type.VectorSize;
            return new ClusteringPerInstanceEvaluator(Host, schema.Schema, scoreInfo.Name, numClusters);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("NMI", Nmi);
            yield return new MetricColumn("AvgMinScore", AvgMinScore, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("DBI", Dbi, MetricColumn.Objective.Minimize);
        }

        protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, DvText, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<DvText>();
            var isWeighted = new List<DvBool>();
            var nmi = new List<Double>();
            var avgMinScores = new List<Double>();
            var dbi = new List<Double>();

            bool hasStrats = Utils.Size(dictionaries) > 0;
            bool hasWeight = aggregator.Weighted;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    Host.Check(agg.Weighted == hasWeight, "All aggregators must either be weighted or unweighted");
                    Host.Check(agg.UnweightedCounters.CalculateDbi == aggregator.UnweightedCounters.CalculateDbi,
                        "All aggregators must either compute DBI or not compute DBI");

                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    isWeighted.Add(DvBool.False);
                    nmi.Add(agg.UnweightedCounters.Nmi);
                    avgMinScores.Add(agg.UnweightedCounters.AvgMinScores);
                    if (agg.UnweightedCounters.CalculateDbi)
                        dbi.Add(agg.UnweightedCounters.Dbi);
                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(DvBool.True);
                        nmi.Add(agg.WeightedCounters.Nmi);
                        avgMinScores.Add(agg.WeightedCounters.AvgMinScores);
                        if (agg.WeightedCounters.CalculateDbi)
                            dbi.Add(agg.WeightedCounters.Dbi);
                    }
                };

            consolidate =
                () =>
                {
                    var overallDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, stratVal.ToArray());
                    }
                    if (hasWeight)
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BoolType.Instance, isWeighted.ToArray());
                    overallDvBldr.AddColumn(Nmi, NumberType.R8, nmi.ToArray());
                    overallDvBldr.AddColumn(AvgMinScore, NumberType.R8, avgMinScores.ToArray());
                    if (aggregator.UnweightedCounters.CalculateDbi)
                        overallDvBldr.AddColumn(Dbi, NumberType.R8, dbi.ToArray());

                    var result = new Dictionary<string, IDataView>
                    {
                        { MetricKinds.OverallMetrics, overallDvBldr.GetDataView() }
                    };

                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public sealed class Counters
            {
                private Double _numInstances;
                private Double _sumMinScores;

                // Since we know in advance how many clusters we have, this can be an array.
                private readonly Double[] _numInstancesOfClstr;
                // We don't know how many classes there will be, so this is a list, that grows when we see new classes.
                private readonly List<Double> _numInstancesOfClass;

                private readonly List<Double[]> _confusionMatrix;

                // These are used for DBI calculation.
                private readonly VBuffer<Single>[] _clusterCentroids;
                private readonly Double[] _distancesToCentroids;

                private readonly int _numClusters;
                public readonly bool CalculateDbi;

                public Double Nmi
                {
                    get
                    {
                        Double nmi = Double.NaN;
                        if (_confusionMatrix.Count > 1)
                        {
                            nmi = 0;
                            Double entropy = 0;
                            for (int i = 0; i < _confusionMatrix.Count; i++)
                            {
                                var px = _numInstancesOfClass[i] / _numInstances;
                                if (px <= 0)
                                    continue;

                                for (int j = 0; j < _confusionMatrix[i].Length; j++)
                                {
                                    var pxy = _confusionMatrix[i][j] / _numInstances;
                                    var py = _numInstancesOfClstr[j] / _numInstances;
                                    if (pxy <= 0 || py <= 0)
                                        continue;

                                    nmi += pxy * Math.Log(pxy / (px * py));
                                }

                                entropy += -px * Math.Log(px);
                            }

                            nmi /= entropy; // entropy can't be zero, because there's at least 2 instances in at least 2 classes
                        }
                        return nmi;
                    }
                }

                public Double AvgMinScores { get { return _sumMinScores / _numInstances; } }

                public Double Dbi
                {
                    get
                    {
                        if (!CalculateDbi)
                            return Double.NaN;

                        Double dbi = 0;
                        var clusterCount = _distancesToCentroids.Length;
                        for (int i = 0; i < clusterCount; i++)
                            _distancesToCentroids[i] /= _numInstancesOfClstr[i];

                        for (int i = 0; i < clusterCount; i++)
                        {
                            Double maxi = 0;
                            if (_numInstancesOfClstr[i] == 0)
                                continue;
                            var centroidI = _clusterCentroids[i];

                            for (int j = 0; j < clusterCount; j++)
                            {
                                if (i == j)
                                    continue;
                                if (_numInstancesOfClstr[j] == 0)
                                    continue;
                                var centroidJ = _clusterCentroids[j];
                                Double num = _distancesToCentroids[i] + _distancesToCentroids[j];
                                Single denom = VectorUtils.Distance(ref centroidI, ref centroidJ);
                                maxi = Math.Max(maxi, num / denom);
                            }

                            dbi += maxi;
                        }

                        dbi /= clusterCount;
                        return dbi;
                    }
                }

                public Counters(int numClusters, bool calculateDbi, ColumnInfo features)
                {
                    _numClusters = numClusters;
                    CalculateDbi = calculateDbi;

                    _numInstancesOfClstr = new Double[_numClusters];
                    _numInstancesOfClass = new List<Double>();
                    _confusionMatrix = new List<Double[]>();
                    if (CalculateDbi)
                    {
                        Contracts.AssertValue(features);
                        _clusterCentroids = new VBuffer<Single>[_numClusters];
                        for (int i = 0; i < _numClusters; i++)
                            _clusterCentroids[i] = VBufferUtils.CreateEmpty<Single>(features.Type.VectorSize);
                        _distancesToCentroids = new Double[_numClusters];
                    }
                }

                public void UpdateFirstPass(int intLabel, Single[] scores, Single weight, int[] indices)
                {
                    Contracts.Assert(Utils.Size(scores) == _numClusters);
                    Contracts.Assert(Utils.Size(indices) == _numClusters);

                    int assigned = indices[0];

                    _numInstances += weight;

                    _sumMinScores += weight * scores[indices[0]];

                    while (_numInstancesOfClass.Count <= intLabel)
                        _numInstancesOfClass.Add(0);

                    _numInstancesOfClass[intLabel] += weight;
                    _numInstancesOfClstr[assigned] += weight;

                    while (_confusionMatrix.Count <= intLabel)
                        _confusionMatrix.Add(new Double[scores.Length]);
                    _confusionMatrix[intLabel][assigned] += weight;
                }

                public void InitializeSecondPass(VBuffer<Single>[] clusterCentroids)
                {
                    for (int i = 0; i < clusterCentroids.Length; i++)
                    {
                        clusterCentroids[i].CopyTo(ref _clusterCentroids[i]);
                        VectorUtils.ScaleBy(ref _clusterCentroids[i], (Single)(1.0 / _numInstancesOfClstr[i]));
                    }
                }

                public void UpdateSecondPass(ref VBuffer<Single> features, int[] indices)
                {
                    int assigned = indices[0];

                    var distance = VectorUtils.Distance(ref _clusterCentroids[assigned], ref features);
                    _distancesToCentroids[assigned] += distance;
                }
            }

            // The getters are initialized in InitializeNextPass(), when the new IRowCursor is available.
            private ValueGetter<Single> _labelGetter;
            private ValueGetter<VBuffer<Single>> _scoreGetter;
            private ValueGetter<Single> _weightGetter;
            private ValueGetter<VBuffer<Single>> _featGetter;

            // Buffers that hold the features and the scores of the current row.
            private VBuffer<Single> _scores;
            private readonly Single[] _scoresArr;
            private readonly int[] _indicesArr;
            private VBuffer<Single> _features;

            // This is used for DBI calculation.
            private readonly VBuffer<Single>[] _clusterCentroids;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;

            public readonly bool Weighted;

            private readonly bool _calculateDbi;

            public Aggregator(IHostEnvironment env, ColumnInfo features, int scoreVectorSize, bool calculateDbi, bool weighted, string stratName)
                : base(env, stratName)
            {
                _calculateDbi = calculateDbi;
                _scoresArr = new float[scoreVectorSize];
                _indicesArr = new int[scoreVectorSize];
                UnweightedCounters = new Counters(scoreVectorSize, _calculateDbi, features);
                Weighted = weighted;
                WeightedCounters = Weighted ? new Counters(scoreVectorSize, _calculateDbi, features) : null;
                if (_calculateDbi)
                {
                    Host.AssertValue(features);
                    _clusterCentroids = new VBuffer<Single>[scoreVectorSize];
                    for (int i = 0; i < scoreVectorSize; i++)
                        _clusterCentroids[i] = VBufferUtils.CreateEmpty<Single>(features.Type.VectorSize);
                }
            }

            private void ProcessRowFirstPass()
            {
                AssertValid(assertGetters: true);

                Single label = 0;
                _labelGetter(ref label);
                if (Single.IsNaN(label))
                {
                    NumUnlabeledInstances++;
                    label = 0;
                }
                var intLabel = (int)label;
                if (intLabel != label || intLabel < 0)
                    throw Host.Except("Invalid label: {0}", label);

                _scoreGetter(ref _scores);
                Host.Check(_scores.Length == _scoresArr.Length);

                if (VBufferUtils.HasNaNs(ref _scores) || VBufferUtils.HasNonFinite(ref _scores))
                {
                    NumBadScores++;
                    return;
                }
                _scores.CopyTo(_scoresArr);
                Single weight = 1;
                if (_weightGetter != null)
                {
                    _weightGetter(ref weight);
                    if (!FloatUtils.IsFinite(weight))
                    {
                        NumBadWeights++;
                        weight = 1;
                    }
                }

                int j = 0;
                foreach (var index in Enumerable.Range(0, _scoresArr.Length).OrderBy(i => _scoresArr[i]))
                    _indicesArr[j++] = index;

                UnweightedCounters.UpdateFirstPass(intLabel, _scoresArr, 1, _indicesArr);
                if (WeightedCounters != null)
                    WeightedCounters.UpdateFirstPass(intLabel, _scoresArr, weight, _indicesArr);

                if (_clusterCentroids != null)
                {
                    _featGetter(ref _features);
                    VectorUtils.Add(ref _features, ref _clusterCentroids[_indicesArr[0]]);
                }
            }

            private void ProcessRowSecondPass()
            {
                AssertValid(assertGetters: true);

                _featGetter(ref _features);
                _scoreGetter(ref _scores);
                Host.Check(_scores.Length == _scoresArr.Length);

                if (VBufferUtils.HasNaNs(ref _scores) || VBufferUtils.HasNonFinite(ref _scores))
                    return;
                _scores.CopyTo(_scoresArr);
                int j = 0;
                foreach (var index in Enumerable.Range(0, _scoresArr.Length).OrderBy(i => _scoresArr[i]))
                    _indicesArr[j++] = index;

                UnweightedCounters.UpdateSecondPass(ref _features, _indicesArr);
                if (WeightedCounters != null)
                    WeightedCounters.UpdateSecondPass(ref _features, _indicesArr);
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                AssertValid(assertGetters: false);

                Host.AssertValue(row);
                Host.AssertValue(schema);

                if (_calculateDbi)
                {
                    Host.AssertValue(schema.Feature);
                    _featGetter = row.GetGetter<VBuffer<Single>>(schema.Feature.Index);
                }
                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
                Host.Assert(score.Type.VectorSize == _scoresArr.Length);
                _scoreGetter = row.GetGetter<VBuffer<Single>>(score.Index);

                if (PassNum == 0)
                {
                    if (schema.Label != null)
                        _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
                    else
                        _labelGetter = (ref Single value) => value = Single.NaN;
                    if (schema.Weight != null)
                        _weightGetter = row.GetGetter<Single>(schema.Weight.Index);
                }
                else
                {
                    Host.Assert(PassNum == 1 && _calculateDbi);
                    UnweightedCounters.InitializeSecondPass(_clusterCentroids);
                    if (WeightedCounters != null)
                        WeightedCounters.InitializeSecondPass(_clusterCentroids);
                }
                AssertValid(assertGetters: true);
            }

            public override void ProcessRow()
            {
                if (PassNum == 0)
                    ProcessRowFirstPass();
                else
                    ProcessRowSecondPass();
            }

            public override bool IsActive()
            {
                return _calculateDbi && PassNum < 2 || PassNum < 1;
            }

            protected override void FinishPassCore()
            {
                AssertValid(assertGetters: false);
            }

            [Conditional("DEBUG")]
            private void AssertValid(bool assertGetters)
            {
                Host.Assert(IsActive());
                if (assertGetters)
                {
                    if (PassNum == 0)
                    {
                        Host.AssertValue(_labelGetter);
                        Host.AssertValue(_scoreGetter);
                        Host.AssertValueOrNull(_weightGetter);
                        Host.Assert(!_calculateDbi || _featGetter != null);
                    }
                    else
                    {
                        Host.Assert(PassNum == 1 && _calculateDbi);
                        Host.AssertValue(_featGetter);
                        Host.AssertValue(_scoreGetter);
                    }
                }
            }
        }
    }

    public sealed class ClusteringPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "ClusteringPerInstance";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CLSTRINS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const int ClusterIdCol = 0;
        private const int SortedClusterCol = 1;
        private const int SortedClusterScoreCol = 2;

        public const string ClusterId = "ClusterId";
        public const string SortedClusters = "SortedClusters";
        public const string SortedClusterScores = "SortedScores";

        private readonly int _numClusters;
        private readonly ColumnType[] _types;

        public ClusteringPerInstanceEvaluator(IHostEnvironment env, ISchema schema, string scoreCol, int numClusters)
            : base(env, schema, scoreCol, null)
        {
            CheckInputColumnTypes(schema);
            _numClusters = numClusters;

            _types = new ColumnType[3];
            var key = new KeyType(DataKind.U4, 0, _numClusters);
            _types[ClusterIdCol] = key;
            _types[SortedClusterCol] = new VectorType(key, _numClusters);
            _types[SortedClusterScoreCol] = new VectorType(NumberType.R4, _numClusters);
        }

        private ClusteringPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema);

            // *** Binary format **
            // base
            // int: number of clusters

            _numClusters = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numClusters > 0);

            _types = new ColumnType[3];
            var key = new KeyType(DataKind.U4, 0, _numClusters);
            _types[ClusterIdCol] = key;
            _types[SortedClusterCol] = new VectorType(key, _numClusters);
            _types[SortedClusterScoreCol] = new VectorType(NumberType.R4, _numClusters);
        }

        public static ClusteringPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ClusteringPerInstanceEvaluator(env, ctx, schema);
        }

        public override void Save(ModelSaveContext ctx)
        {
            // *** Binary format **
            // base
            // int: number of clusters

            base.Save(ctx);
            Host.Assert(_numClusters > 0);
            ctx.Writer.Write(_numClusters);
        }

        public override Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
        {
            return
                col =>
                    col == ScoreIndex &&
                    (activeOutput(ClusterIdCol) || activeOutput(SortedClusterCol) || activeOutput(SortedClusterScoreCol));
        }

        public override Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
        {
            disposer = null;

            var getters = new Delegate[3];

            if (!activeOutput(ClusterIdCol) && !activeOutput(SortedClusterCol) && !activeOutput(SortedClusterScoreCol))
                return getters;

            long cachedPosition = -1;
            VBuffer<Single> scores = default(VBuffer<Single>);
            var scoresArr = new Single[_numClusters];
            int[] sortedIndices = new int[_numClusters];

            var scoreGetter = input.GetGetter<VBuffer<Single>>(ScoreIndex);
            Action updateCacheIfNeeded =
                () =>
                {
                    if (cachedPosition != input.Position)
                    {
                        scoreGetter(ref scores);
                        scores.CopyTo(scoresArr);
                        int j = 0;
                        foreach (var index in Enumerable.Range(0, scoresArr.Length).OrderBy(i => scoresArr[i]))
                            sortedIndices[j++] = index;
                        cachedPosition = input.Position;
                    }
                };

            if (activeOutput(ClusterIdCol))
            {
                ValueGetter<uint> assignedFn =
                    (ref uint dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = (uint)sortedIndices[0] + 1;
                    };
                getters[ClusterIdCol] = assignedFn;
            }

            if (activeOutput(SortedClusterScoreCol))
            {
                ValueGetter<VBuffer<Single>> topKScoresFn =
                    (ref VBuffer<Single> dst) =>
                    {
                        updateCacheIfNeeded();
                        var values = dst.Values;
                        if (Utils.Size(values) < _numClusters)
                            values = new Single[_numClusters];
                        for (int i = 0; i < _numClusters; i++)
                            values[i] = scores.GetItemOrDefault(sortedIndices[i]);
                        dst = new VBuffer<Single>(_numClusters, values);
                    };
                getters[SortedClusterScoreCol] = topKScoresFn;
            }

            if (activeOutput(SortedClusterCol))
            {
                ValueGetter<VBuffer<uint>> topKClassesFn =
                    (ref VBuffer<uint> dst) =>
                    {
                        updateCacheIfNeeded();
                        var values = dst.Values;
                        if (Utils.Size(values) < _numClusters)
                            values = new uint[_numClusters];
                        for (int i = 0; i < _numClusters; i++)
                            values[i] = (uint)sortedIndices[i] + 1;
                        dst = new VBuffer<uint>(_numClusters, values);
                    };
                getters[SortedClusterCol] = topKClassesFn;
            }
            return getters;
        }

        public override RowMapperColumnInfo[] GetOutputColumns()
        {
            var infos = new RowMapperColumnInfo[3];
            infos[ClusterIdCol] = new RowMapperColumnInfo(ClusterId, _types[ClusterIdCol], null);

            var slotNamesType = new VectorType(TextType.Instance, _numClusters);

            var sortedClusters = new ColumnMetadataInfo(SortedClusters);
            sortedClusters.Add(MetadataUtils.Kinds.SlotNames, new MetadataInfo<VBuffer<DvText>>(slotNamesType,
                CreateSlotNamesGetter(_numClusters, "Cluster")));
            var sortedClusterScores = new ColumnMetadataInfo(SortedClusterScores);
            sortedClusterScores.Add(MetadataUtils.Kinds.SlotNames, new MetadataInfo<VBuffer<DvText>>(slotNamesType,
                CreateSlotNamesGetter(_numClusters, "Score")));

            infos[SortedClusterCol] = new RowMapperColumnInfo(SortedClusters, _types[SortedClusterCol], sortedClusters);
            infos[SortedClusterScoreCol] = new RowMapperColumnInfo(SortedClusterScores,
                _types[SortedClusterScoreCol], sortedClusterScores);
            return infos;
        }

        // REVIEW: Figure out how to avoid having the column name in each slot name.
        private MetadataUtils.MetadataGetter<VBuffer<DvText>> CreateSlotNamesGetter(int numTopClusters, string suffix)
        {
            return
                (int col, ref VBuffer<DvText> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < numTopClusters)
                        values = new DvText[numTopClusters];
                    for (int i = 1; i <= numTopClusters; i++)
                        values[i - 1] = new DvText(string.Format("#{0} {1}", i, suffix));
                    dst = new VBuffer<DvText>(numTopClusters, values);
                };
        }

        private void CheckInputColumnTypes(ISchema schema)
        {
            Host.AssertNonEmpty(ScoreCol);

            var type = schema.GetColumnType(ScoreIndex);
            if (!type.IsKnownSizeVector || type.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type {1}, but must be a float vector of known-size", ScoreCol, type);
        }
    }

    public sealed class ClusteringMamlEvaluator : MamlEvaluatorBase
    {
        public class Arguments : ArgumentsBase
        {
            // REVIEW: Remove BDI centroid measure which is sensible to apply in the k-means case only and remove features argument
            [Argument(ArgumentType.AtMostOnce, HelpText = "Features column name", ShortName = "feat")]
            public string FeatureColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Calculate DBI? (time-consuming unsupervised metric)", ShortName = "dbi")]
            public bool CalculateDbi = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output top K clusters", ShortName = "topk")]
            public int NumTopClustersToOutput = 3;
        }

        private readonly ClusteringEvaluator _evaluator;
        private readonly int _numTopClusters;
        private readonly string _featureCol;
        private readonly bool _calculateDbi;

        protected override IEvaluator Evaluator { get { return _evaluator; } }

        public ClusteringMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.Clustering, "ClusteringMamlEvaluator")
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(1 <= args.NumTopClustersToOutput, nameof(args.NumTopClustersToOutput));

            _numTopClusters = args.NumTopClustersToOutput;
            _featureCol = args.FeatureColumn;
            _calculateDbi = args.CalculateDbi;

            var evalArgs = new ClusteringEvaluator.Arguments
            {
                CalculateDbi = _calculateDbi
            };
            _evaluator = new ClusteringEvaluator(Host, evalArgs);
        }

        protected override IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            foreach (var col in base.GetInputColumnRolesCore(schema))
            {
                if (!col.Key.Equals(RoleMappedSchema.ColumnRole.Label))
                    yield return col;
                else if (schema.Schema.TryGetColumnIndex(col.Value, out int labelIndex))
                    yield return col;
            }

            if (_calculateDbi)
            {
                string feat = EvaluateUtils.GetColName(_featureCol, schema.Feature, DefaultColumnNames.Features);
                if (!schema.Schema.TryGetColumnIndex(feat, out int featCol))
                    throw Host.ExceptUserArg(nameof(Arguments.FeatureColumn), "Features column '{0}' not found", feat);
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(feat);
            }
        }

        // Clustering evaluator adds three per-instance columns: "ClusterId", "Top clusters" and "Top cluster scores".
        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            // Output the label column if it exists.
            if (schema.Label != null)
                yield return schema.Label.Name;

            // Return the output columns.
            yield return ClusteringPerInstanceEvaluator.ClusterId;
            yield return ClusteringPerInstanceEvaluator.SortedClusters;
            yield return ClusteringPerInstanceEvaluator.SortedClusterScores;
        }

        protected override IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            // Wrap with a DropSlots transform to pick only the first _numTopClusters slots.
            if (perInst.Schema.TryGetColumnIndex(ClusteringPerInstanceEvaluator.SortedClusters, out int index))
            {
                var type = perInst.Schema.GetColumnType(index);
                if (_numTopClusters < type.VectorSize)
                {
                    var args = new DropSlotsTransform.Arguments
                    {
                        Column = new DropSlotsTransform.Column[]
                        {
                            new DropSlotsTransform.Column()
                            {
                                Name = ClusteringPerInstanceEvaluator.SortedClusters,
                                Slots = new[] {
                                    new DropSlotsTransform.Range()
                                    {
                                        Min = _numTopClusters
                                    }
                                }
                            }
                        }
                    };
                    perInst = new DropSlotsTransform(Host, args, perInst);
                }
            }

            if (perInst.Schema.TryGetColumnIndex(ClusteringPerInstanceEvaluator.SortedClusterScores, out index))
            {
                var type = perInst.Schema.GetColumnType(index);
                if (_numTopClusters < type.VectorSize)
                {
                    var args = new DropSlotsTransform.Arguments
                    {
                        Column = new DropSlotsTransform.Column[]
                        {
                            new DropSlotsTransform.Column()
                            {
                                Name = ClusteringPerInstanceEvaluator.SortedClusterScores,
                                Slots = new[] {
                                    new DropSlotsTransform.Range()
                                    {
                                        Min = _numTopClusters
                                    }
                                }
                            }
                        }
                    };
                    perInst = new DropSlotsTransform(Host, args, perInst);
                }
            }
            return perInst;
        }
    }

    public static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.ClusterEvaluator", Desc = "Evaluates a clustering scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput Clustering(IHostEnvironment env, ClusteringMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateClustering");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            MatchColumns(host, input, out string label, out string weight, out string name);
            ISchema schema = input.Data.Schema;
            string features = TrainUtils.MatchNameOrDefaultOrNull(host, schema,
                nameof(ClusteringMamlEvaluator.Arguments.FeatureColumn),
                input.FeatureColumn, DefaultColumnNames.Features);
            var evaluator = new ClusteringMamlEvaluator(host, input);
            var data = new RoleMappedData(input.Data, label, features, null, weight, name);
            var metrics = evaluator.Evaluate(data);

            var warnings = ExtractWarnings(host, metrics);
            var overallMetrics = ExtractOverallMetrics(host, metrics, evaluator);
            var perInstanceMetrics = evaluator.GetPerInstanceMetrics(data);

            return new CommonOutputs.CommonEvaluateOutput()
            {
                Warnings = warnings,
                OverallMetrics = overallMetrics,
                PerInstanceMetrics = perInstanceMetrics
            };
        }
    }
}
