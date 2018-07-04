// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(RankerEvaluator), typeof(RankerEvaluator), typeof(RankerEvaluator.Arguments), typeof(SignatureEvaluator),
    "Ranking Evaluator", RankerEvaluator.LoadName, "Ranking", "rank")]

[assembly: LoadableClass(typeof(RankerMamlEvaluator), typeof(RankerMamlEvaluator), typeof(RankerMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Ranking Evaluator", RankerEvaluator.LoadName, "Ranking", "rank")]

[assembly: LoadableClass(typeof(RankerPerInstanceTransform), null, typeof(SignatureLoadDataTransform),
    "", RankerPerInstanceTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class RankerEvaluator : EvaluatorBase<RankerEvaluator.Aggregator>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum truncation level for computing (N)DCG", ShortName = "t")]
            public int DcgTruncationLevel = 3;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label relevance gains", ShortName = "gains")]
            public string LabelGains = "0,3,7,15,31";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Generate per-group (N)DCG", ShortName = "ogs")]
            public bool OutputGroupSummary;
        }

        public const string LoadName = "RankingEvaluator";

        public const string Ndcg = "NDCG";
        public const string Dcg = "DCG";
        public const string MaxDcg = "MaxDCG";

        /// <summary>
        /// The ranking evaluator outputs a data view by this name, which contains metrics aggregated per group. 
        /// It contains four columns: GroupId, NDCG, DCG and MaxDCG. Each row in the data view corresponds to one
        /// group in the scored data.
        /// </summary>
        public const string GroupSummary = "GroupSummary";

        private const string GroupId = "GroupId";

        private readonly int _truncationLevel;
        private readonly bool _groupSummary;
        private readonly Double[] _labelGains;

        public RankerEvaluator(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            // REVIEW: What kind of checking should be applied to labelGains?
            if (args.DcgTruncationLevel <= 0 || args.DcgTruncationLevel > Aggregator.Counters.MaxTruncationLevel)
                throw Host.ExceptUserArg(nameof(args.DcgTruncationLevel), "DCG Truncation Level must be between 1 and {0}", Aggregator.Counters.MaxTruncationLevel);
            Host.CheckUserArg(args.LabelGains != null, nameof(args.LabelGains), "Label gains cannot be null");

            _truncationLevel = args.DcgTruncationLevel;
            _groupSummary = args.OutputGroupSummary;

            var labelGains = new List<Double>();
            string[] gains = args.LabelGains.Split(',');
            for (int i = 0; i < gains.Length; i++)
            {
                Double gain;
                if (!Double.TryParse(gains[i], out gain))
                    throw Host.ExceptUserArg(nameof(args.LabelGains), "Label Gains must be of floating or integral type", Aggregator.Counters.MaxTruncationLevel);
                labelGains.Add(gain);
            }
            _labelGains = labelGains.ToArray();
        }

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var t = schema.Label.Type;
            if (t != NumberType.Float && !t.IsKey)
            {
                throw Host.ExceptUserArg(nameof(RankerMamlEvaluator.Arguments.LabelColumn), "Label column '{0}' has type '{1}' but must be R4 or a key",
                    schema.Label.Name, t);
            }
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            if (scoreInfo.Type != NumberType.Float)
            {
                throw Host.ExceptUserArg(nameof(RankerMamlEvaluator.Arguments.ScoreColumn), "Score column '{0}' has type '{1}' but must be R4",
                    scoreInfo.Name, t);
            }
        }

        protected override void CheckCustomColumnTypesCore(RoleMappedSchema schema)
        {
            var t = schema.Group.Type;
            if (!t.IsKey)
            {
                throw Host.ExceptUserArg(nameof(RankerMamlEvaluator.Arguments.GroupIdColumn),
                    "Group column '{0}' has type '{1}' but must be a key",
                    schema.Group.Name, t);
            }
        }

        // Add also the group column.
        protected override Func<int, bool> GetActiveColsCore(RoleMappedSchema schema)
        {
            var pred = base.GetActiveColsCore(schema);
            return i => i == schema.Group.Index || pred(i);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            return new Aggregator(Host, _labelGains, _truncationLevel, _groupSummary, schema.Weight != null, stratName);
        }

        public override IDataTransform GetPerInstanceMetrics(RoleMappedData data)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckParam(data.Schema.Label != null, nameof(data), "Schema must contain a label column");
            var scoreInfo = data.Schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Host.CheckParam(data.Schema.Group != null, nameof(data), "Schema must contain a group column");

            return new RankerPerInstanceTransform(Host, data.Data,
                data.Schema.Label.Name, scoreInfo.Name, data.Schema.Group.Name, _truncationLevel, _labelGains);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("NDCG@<number>", Ndcg, isVector: true,
                namePattern: new Regex(string.Format(@"^{0}@(?<at>\d+)", Ndcg), RegexOptions.IgnoreCase),
                groupName: "at", nameFormat: string.Format("{0} @{{0}}", Ndcg));
            yield return new MetricColumn("DCG@<number>", Dcg, isVector: true,
                namePattern: new Regex(string.Format(@"^{0}@(?<at>\d+)", Dcg), RegexOptions.IgnoreCase),
                groupName: "at", nameFormat: string.Format("{0} @{{0}}", Dcg));
            yield return new MetricColumn("MaxDcg@<number>", MaxDcg, isVector: true,
                namePattern: new Regex(string.Format(@"^{0}@(?<at>\d+)", MaxDcg), RegexOptions.IgnoreCase),
                groupName: "at", nameFormat: string.Format("{0} @{{0}}", MaxDcg));
        }

        protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, DvText, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<DvText>();
            var isWeighted = new List<DvBool>();
            var ndcg = new List<Double[]>();
            var dcg = new List<Double[]>();

            var groupName = new List<DvText>();
            var groupNdcg = new List<Double[]>();
            var groupDcg = new List<Double[]>();
            var groupMaxDcg = new List<Double[]>();
            var groupStratCol = new List<uint>();
            var groupStratVal = new List<DvText>();

            bool hasStrats = Utils.Size(dictionaries) > 0;
            bool hasWeight = aggregator.Weighted;
            bool groupSummary = aggregator.UnweightedCounters.GroupSummary;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    Host.Check(agg.Weighted == hasWeight, "All aggregators must either be weighted or unweighted");
                    Host.Check(agg.UnweightedCounters.GroupSummary == aggregator.UnweightedCounters.GroupSummary,
                        "All aggregators must either compute group summary or not compute group summary");

                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    isWeighted.Add(DvBool.False);
                    ndcg.Add(agg.UnweightedCounters.Ndcg);
                    dcg.Add(agg.UnweightedCounters.Dcg);
                    if (agg.UnweightedCounters.GroupSummary)
                    {
                        groupStratCol.AddRange(agg.UnweightedCounters.GroupDcg.Select(x => stratColKey));
                        groupStratVal.AddRange(agg.UnweightedCounters.GroupDcg.Select(x => stratColVal));
                        groupName.AddRange(agg.GroupId.Select(sb => new DvText(sb.ToString())));
                        groupNdcg.AddRange(agg.UnweightedCounters.GroupNdcg);
                        groupDcg.AddRange(agg.UnweightedCounters.GroupDcg);
                        groupMaxDcg.AddRange(agg.UnweightedCounters.GroupMaxDcg);
                    }

                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(DvBool.True);
                        ndcg.Add(agg.WeightedCounters.Ndcg);
                        dcg.Add(agg.WeightedCounters.Dcg);
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
                    overallDvBldr.AddColumn(Ndcg, aggregator.GetSlotNames, NumberType.R8, ndcg.ToArray());
                    overallDvBldr.AddColumn(Dcg, aggregator.GetSlotNames, NumberType.R8, dcg.ToArray());

                    var groupDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        groupDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, groupStratCol.ToArray());
                        groupDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, groupStratVal.ToArray());
                    }
                    if (groupSummary)
                    {
                        groupDvBldr.AddColumn(GroupId, TextType.Instance, groupName.ToArray());
                        groupDvBldr.AddColumn(Ndcg, aggregator.GetGroupSummarySlotNames("NDCG"), NumberType.R8, groupNdcg.ToArray());
                        groupDvBldr.AddColumn(Dcg, aggregator.GetGroupSummarySlotNames("DCG"), NumberType.R8, groupDcg.ToArray());
                        groupDvBldr.AddColumn(MaxDcg, aggregator.GetGroupSummarySlotNames("MaxDCG"), NumberType.R8, groupMaxDcg.ToArray());
                    }

                    var result = new Dictionary<string, IDataView>();
                    result.Add(MetricKinds.OverallMetrics, overallDvBldr.GetDataView());
                    if (groupSummary)
                        result.Add(GroupSummary, groupDvBldr.GetDataView());
                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public sealed class Counters
            {
                public const int MaxTruncationLevel = 10;

                public readonly int TruncationLevel;
                private readonly List<Double[]> _groupNdcg;
                private readonly List<Double[]> _groupDcg;
                private readonly List<Double[]> _groupMaxDcg;
                private readonly Double[] _groupDcgCur;
                private readonly Double[] _groupMaxDcgCur;

                private readonly Double[] _sumNdcgAtN;
                private readonly Double[] _sumDcgAtN;
                private Double _sumWeights;

                private readonly List<short> _queryLabels;
                private readonly List<Single> _queryOutputs;
                private readonly Double[] _labelGains;

                public bool GroupSummary { get { return _groupNdcg != null; } }

                public Double[] Ndcg
                {
                    get
                    {
                        var ndcg = new Double[TruncationLevel];
                        for (int i = 0; i < TruncationLevel; i++)
                            ndcg[i] = _sumNdcgAtN[i] / _sumWeights;
                        return ndcg;
                    }
                }

                public Double[] Dcg
                {
                    get
                    {
                        var dcg = new Double[TruncationLevel];
                        for (int i = 0; i < TruncationLevel; i++)
                            dcg[i] = _sumDcgAtN[i] / _sumWeights;
                        return dcg;
                    }
                }

                public Double[][] GroupDcg
                {
                    get
                    {
                        if (_groupDcg == null)
                            return null;
                        return _groupDcg.ToArray();
                    }
                }

                public Double[][] GroupNdcg
                {
                    get
                    {
                        if (_groupNdcg == null)
                            return null;
                        return _groupNdcg.ToArray();
                    }
                }

                public Double[][] GroupMaxDcg
                {
                    get
                    {
                        if (_groupMaxDcg == null)
                            return null;
                        return _groupMaxDcg.ToArray();
                    }
                }

                public Counters(Double[] labelGains, int truncationLevel, bool groupSummary)
                {
                    Contracts.Assert(truncationLevel > 0);
                    Contracts.AssertValue(labelGains);

                    TruncationLevel = truncationLevel;
                    _sumDcgAtN = new Double[TruncationLevel];
                    _sumNdcgAtN = new Double[TruncationLevel];

                    _groupDcgCur = new Double[TruncationLevel];
                    _groupMaxDcgCur = new Double[TruncationLevel];
                    if (groupSummary)
                    {
                        _groupNdcg = new List<Double[]>();
                        _groupDcg = new List<Double[]>();
                        _groupMaxDcg = new List<Double[]>();
                    }

                    _queryLabels = new List<short>();
                    _queryOutputs = new List<Single>();
                    _labelGains = labelGains;
                }

                public void Update(short label, Single output)
                {
                    _queryLabels.Add(label);
                    _queryOutputs.Add(output);
                }

                public void UpdateGroup(Single weight)
                {
                    RankerUtils.QueryMaxDcg(_labelGains, TruncationLevel, _queryLabels, _queryOutputs, _groupMaxDcgCur);
                    if (_groupMaxDcg != null)
                    {
                        var maxDcg = new Double[TruncationLevel];
                        Array.Copy(_groupMaxDcgCur, maxDcg, TruncationLevel);
                        _groupMaxDcg.Add(maxDcg);
                    }

                    RankerUtils.QueryDcg(_labelGains, TruncationLevel, _queryLabels, _queryOutputs, _groupDcgCur);
                    if (_groupDcg != null)
                    {
                        var groupDcg = new Double[TruncationLevel];
                        Array.Copy(_groupDcgCur, groupDcg, TruncationLevel);
                        _groupDcg.Add(groupDcg);
                    }

                    var groupNdcg = new Double[TruncationLevel];
                    for (int t = 0; t < TruncationLevel; t++)
                    {
                        Double ndcg = _groupMaxDcgCur[t] > 0 ? _groupDcgCur[t] / _groupMaxDcgCur[t] * 100 : 0;
                        _sumNdcgAtN[t] += ndcg * weight;
                        _sumDcgAtN[t] += _groupDcgCur[t] * weight;
                        groupNdcg[t] = ndcg;
                    }
                    _sumWeights += weight;

                    if (_groupNdcg != null)
                        _groupNdcg.Add(groupNdcg);

                    _queryLabels.Clear();
                    _queryOutputs.Clear();
                }
            }

            private Single _currentQueryWeight;

            private ValueGetter<Single> _labelGetter;
            private ValueGetter<Single> _scoreGetter;
            private ValueGetter<Single> _weightGetter;
            private Func<bool> _newGroupDel;
            private Action _groupSbUpdate;
            private StringBuilder _groupSb;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;
            public readonly bool Weighted;
            public readonly List<DvText> GroupId;
            private int _groupSize;

            public Aggregator(IHostEnvironment env, Double[] labelGains, int truncationLevel, bool groupSummary, bool weighted, string stratName)
                : base(env, stratName)
            {
                Host.AssertValue(labelGains);
                Host.Assert(truncationLevel > 0);

                UnweightedCounters = new Counters(labelGains, truncationLevel, groupSummary);
                Weighted = weighted;
                WeightedCounters = Weighted ? new Counters(labelGains, truncationLevel, false) : null;

                _currentQueryWeight = Single.NaN;

                if (groupSummary)
                    GroupId = new List<DvText>();
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                Contracts.Assert(PassNum < 1);
                Contracts.AssertValue(schema.Label);
                Contracts.AssertValue(schema.Group);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
                _scoreGetter = row.GetGetter<Single>(score.Index);
                _newGroupDel = RowCursorUtils.GetIsNewGroupDelegate(row, schema.Group.Index);
                if (schema.Weight != null)
                    _weightGetter = row.GetGetter<Single>(schema.Weight.Index);

                if (UnweightedCounters.GroupSummary)
                {
                    ValueGetter<StringBuilder> groupIdBuilder = RowCursorUtils.GetGetterAsStringBuilder(row, schema.Group.Index);
                    _groupSbUpdate = () => groupIdBuilder(ref _groupSb);
                }
                else
                    _groupSbUpdate = () => { };
            }

            public override void ProcessRow()
            {
                if (_newGroupDel())
                {
                    if (_groupSize > 0)
                    {
                        ProcessGroup();
                        _groupSize = 0;
                    }
                    _groupSbUpdate();
                }

                Single label = 0;
                Single score = 0;
                _labelGetter(ref label);
                _scoreGetter(ref score);

                if (Single.IsNaN(score))
                {
                    NumBadScores++;
                    return;
                }

                UnweightedCounters.Update((short)label, score);
                if (WeightedCounters != null)
                    WeightedCounters.Update((short)label, score);
                _groupSize++;
                Single weight = 1;
                if (_weightGetter != null)
                {
                    _weightGetter(ref weight);
                    if (Single.IsNaN(_currentQueryWeight))
                        _currentQueryWeight = weight;
                    else
                        Contracts.Check(weight == _currentQueryWeight, "Weights within query differ");
                }
            }

            private void ProcessGroup()
            {
                UnweightedCounters.UpdateGroup(1);
                if (WeightedCounters != null)
                    WeightedCounters.UpdateGroup(_currentQueryWeight);
                if (GroupId != null)
                    GroupId.Add(new DvText(_groupSb.ToString()));
                _currentQueryWeight = Single.NaN;
            }

            protected override void FinishPassCore()
            {
                base.FinishPassCore();
                if (_groupSize > 0)
                    ProcessGroup();
            }

            public ValueGetter<VBuffer<DvText>> GetGroupSummarySlotNames(string prefix)
            {
                return
                    (ref VBuffer<DvText> dst) =>
                    {
                        var values = dst.Values;
                        if (Utils.Size(values) < UnweightedCounters.TruncationLevel)
                            values = new DvText[UnweightedCounters.TruncationLevel];

                        for (int i = 0; i < UnweightedCounters.TruncationLevel; i++)
                            values[i] = new DvText(string.Format("{0}@{1}", prefix, i + 1));
                        dst = new VBuffer<DvText>(UnweightedCounters.TruncationLevel, values);
                    };
            }

            public void GetSlotNames(ref VBuffer<DvText> slotNames)
            {
                var values = slotNames.Values;
                if (Utils.Size(values) < UnweightedCounters.TruncationLevel)
                    values = new DvText[UnweightedCounters.TruncationLevel];

                for (int i = 0; i < UnweightedCounters.TruncationLevel; i++)
                    values[i] = new DvText(string.Format("@{0}", i + 1));
                slotNames = new VBuffer<DvText>(UnweightedCounters.TruncationLevel, values);
            }
        }
    }

    public sealed class RankerPerInstanceTransform : IDataTransform
    {
        public const string LoaderSignature = "RankerPerInstTransform";
        private const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RNK INST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public const string Ndcg = "NDCG";
        public const string Dcg = "DCG";
        public const string MaxDcg = "MaxDCG";

        private readonly Transform _transform;

        public IDataView Source { get { return _transform.Source; } }

        public bool CanShuffle { get { return _transform.CanShuffle; } }

        public ISchema Schema { get { return _transform.Schema; } }

        public RankerPerInstanceTransform(IHostEnvironment env, IDataView input, string labelCol, string scoreCol, string groupCol,
                int truncationLevel, Double[] labelGains)
        {
            _transform = new Transform(env, input, labelCol, scoreCol, groupCol, truncationLevel, labelGains);
        }

        private RankerPerInstanceTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            _transform = new Transform(env, ctx, input);
        }

        public static RankerPerInstanceTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new RankerPerInstanceTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _transform.Save(ctx);
        }

        public long? GetRowCount(bool lazy = true)
        {
            return _transform.GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _transform.GetRowCursor(needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _transform.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        private sealed class Transform : PerGroupTransformBase<short, Single, Transform.RowCursorState>
        {
            private sealed class Bindings : BindingsBase
            {
                private readonly ColumnType _outputType;
                private readonly ColumnType _slotNamesType;
                private readonly int _truncationLevel;
                private readonly MetadataUtils.MetadataGetter<VBuffer<DvText>> _slotNamesGetter;

                public Bindings(IExceptionContext ectx, ISchema input, bool user, string labelCol, string scoreCol, string groupCol,
                    int truncationLevel)
                    : base(ectx, input, labelCol, scoreCol, groupCol, user, Ndcg, Dcg, MaxDcg)
                {
                    _truncationLevel = truncationLevel;
                    _outputType = new VectorType(NumberType.R8, _truncationLevel);
                    _slotNamesType = new VectorType(TextType.Instance, _truncationLevel);
                    _slotNamesGetter = SlotNamesGetter;
                }

                protected override ColumnType GetColumnTypeCore(int iinfo)
                {
                    Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                    return _outputType;
                }

                protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
                {
                    Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                    var types = base.GetMetadataTypesCore(iinfo);
                    types = types.Prepend(_slotNamesType.GetPair(MetadataUtils.Kinds.SlotNames));
                    return types;
                }

                protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
                {
                    Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                    if (kind == MetadataUtils.Kinds.SlotNames)
                        return _slotNamesType;
                    return base.GetMetadataTypeCore(kind, iinfo);
                }

                protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
                {
                    Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                    if (kind == MetadataUtils.Kinds.SlotNames)
                    {
                        _slotNamesGetter.Marshal(iinfo, ref value);
                        return;
                    }
                    base.GetMetadataCore(kind, iinfo, ref value);
                }

                private void SlotNamesGetter(int iinfo, ref VBuffer<DvText> dst)
                {
                    Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                    var values = dst.Values;
                    if (Utils.Size(values) < _truncationLevel)
                        values = new DvText[_truncationLevel];
                    for (int i = 0; i < _truncationLevel; i++)
                        values[i] =
                            new DvText(string.Format("{0}@{1}", iinfo == NdcgCol ? Ndcg : iinfo == DcgCol ? Dcg : MaxDcg,
                                i + 1));
                    dst = new VBuffer<DvText>(_truncationLevel, values);
                }
            }

            private const int NdcgCol = 0;
            private const int DcgCol = 1;
            private const int MaxDcgCol = 2;

            private readonly Bindings _bindings;
            private readonly int _truncationLevel;
            private readonly Double[] _labelGains;

            public Transform(IHostEnvironment env, IDataView input, string labelCol, string scoreCol, string groupCol,
                int truncationLevel, Double[] labelGains)
                : base(env, input, labelCol, scoreCol, groupCol, RegistrationName)
            {
                Host.CheckParam(0 < truncationLevel && truncationLevel < 100, nameof(truncationLevel),
                    "Truncation level must be between 1 and 99");
                Host.CheckValue(labelGains, nameof(labelGains));

                _truncationLevel = truncationLevel;
                _labelGains = labelGains;
                _bindings = new Bindings(Host, Source.Schema, true, LabelCol, ScoreCol, GroupCol, _truncationLevel);
            }

            public Transform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
                : base(env, ctx, input, RegistrationName)
            {
                // *** Binary format ***
                // base
                // int: _truncationLevel
                // int: _labelGains.Length
                // double[]: _labelGains

                _truncationLevel = ctx.Reader.ReadInt32();
                Host.CheckDecode(0 < _truncationLevel && _truncationLevel < 100);
                _labelGains = ctx.Reader.ReadDoubleArray();
                _bindings = new Bindings(Host, input.Schema, false, LabelCol, ScoreCol, GroupCol, _truncationLevel);
            }

            public override void Save(ModelSaveContext ctx)
            {
                Host.AssertValue(ctx);

                // *** Binary format ***
                // base
                // int: _truncationLevel
                // int: _labelGains.Length
                // double[]: _labelGains

                base.Save(ctx);
                Host.Assert(0 < _truncationLevel && _truncationLevel < 100);
                ctx.Writer.Write(_truncationLevel);
                ctx.Writer.WriteDoubleArray(_labelGains);
            }

            protected override BindingsBase GetBindings()
            {
                return _bindings;
            }

            protected override Delegate[] CreateGetters(RowCursorState state, Func<int, bool> predicate)
            {
                var getters = new Delegate[_bindings.InfoCount];

                if (predicate(NdcgCol))
                {
                    var ndcg = state.NdcgCur;
                    ValueGetter<VBuffer<Double>> fn =
                        (ref VBuffer<Double> dst) => Copy(ndcg, ref dst);
                    getters[NdcgCol] = fn;
                }
                if (predicate(DcgCol))
                {
                    var dcg = state.DcgCur;
                    ValueGetter<VBuffer<Double>> fn = (ref VBuffer<Double> dst) => Copy(dcg, ref dst);
                    getters[DcgCol] = fn;
                }
                if (predicate(MaxDcgCol))
                {
                    var maxDcg = state.MaxDcgCur;
                    ValueGetter<VBuffer<Double>> fn = (ref VBuffer<Double> dst) => Copy(maxDcg, ref dst);
                    getters[MaxDcgCol] = fn;
                }
                return getters;
            }

            private void Copy(Double[] src, ref VBuffer<Double> dst)
            {
                Host.AssertValue(src);
                var values = dst.Values;
                if (Utils.Size(values) < src.Length)
                    values = new Double[src.Length];
                src.CopyTo(values, 0);
                dst = new VBuffer<Double>(src.Length, values);
            }

            protected override ValueGetter<short> GetLabelGetter(IRow row)
            {
                var lb = RowCursorUtils.GetLabelGetter(row, _bindings.LabelIndex);
                return
                    (ref short dst) =>
                    {
                        Single label = 0;
                        lb(ref label);
                        dst = (short)label;
                    };
            }

            protected override ValueGetter<Single> GetScoreGetter(IRow row)
            {
                return row.GetGetter<Single>(_bindings.ScoreIndex);
            }

            protected override RowCursorState InitializeState(IRow input)
            {
                return new RowCursorState(_truncationLevel);
            }

            protected override void ProcessExample(RowCursorState state, short label, Single score)
            {
                state.QueryLabels.Add(label);
                state.QueryOutputs.Add(score);
            }

            protected override void UpdateState(RowCursorState state)
            {
                // Calculate the current group DCG, NDCG and MaxDcg.
                RankerUtils.QueryMaxDcg(_labelGains, _truncationLevel, state.QueryLabels, state.QueryOutputs,
                    state.MaxDcgCur);
                RankerUtils.QueryDcg(_labelGains, _truncationLevel, state.QueryLabels, state.QueryOutputs, state.DcgCur);
                for (int t = 0; t < _truncationLevel; t++)
                {
                    Double ndcg = state.MaxDcgCur[t] > 0 ? state.DcgCur[t] / state.MaxDcgCur[t] * 100 : 0;
                    state.NdcgCur[t] = ndcg;
                }
                state.QueryLabels.Clear();
                state.QueryOutputs.Clear();
            }

            public sealed class RowCursorState
            {
                public readonly List<short> QueryLabels;
                public readonly List<Single> QueryOutputs;

                public readonly Double[] NdcgCur;
                public readonly Double[] DcgCur;
                public readonly Double[] MaxDcgCur;

                public RowCursorState(int truncationLevel)
                {
                    Contracts.Assert(0 < truncationLevel && truncationLevel < 100);

                    QueryLabels = new List<short>();
                    QueryOutputs = new List<Single>();

                    NdcgCur = new Double[truncationLevel];
                    DcgCur = new Double[truncationLevel];
                    MaxDcgCur = new Double[truncationLevel];
                }
            }
        }
    }

    public sealed class RankerMamlEvaluator : MamlEvaluatorBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for the group ID", ShortName = "group")]
            public string GroupIdColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum truncation level for computing (N)DCG", ShortName = "t")]
            public int DcgTruncationLevel = 3;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label relevance gains", ShortName = "gains")]
            public string LabelGains = "0,3,7,15,31";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Group summary filename", ShortName = "gsf", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string GroupSummaryFilename;
        }

        private readonly RankerEvaluator _evaluator;
        private readonly string _groupIdCol;

        private readonly string _groupSummaryFilename;

        protected override IEvaluator Evaluator { get { return _evaluator; } }

        public RankerMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.Ranking, "RankerMamlEvaluator")
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.GroupSummaryFilename, nameof(args.GroupSummaryFilename));

            var evalArgs = new RankerEvaluator.Arguments();
            evalArgs.DcgTruncationLevel = args.DcgTruncationLevel;
            evalArgs.LabelGains = args.LabelGains;
            evalArgs.OutputGroupSummary = !string.IsNullOrEmpty(args.GroupSummaryFilename);

            _evaluator = new RankerEvaluator(Host, evalArgs);
            _groupSummaryFilename = args.GroupSummaryFilename;
            _groupIdCol = args.GroupIdColumn;
        }

        protected override IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            var cols = base.GetInputColumnRolesCore(schema);
            var groupIdCol = EvaluateUtils.GetColName(_groupIdCol, schema.Group, DefaultColumnNames.GroupId);
            return cols.Prepend(RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Group, groupIdCol));
        }

        protected override void PrintAdditionalMetricsCore(IChannel ch, Dictionary<string, IDataView>[] metrics)
        {
            ch.AssertNonEmpty(metrics);

            if (!string.IsNullOrEmpty(_groupSummaryFilename))
            {
                IDataView gs;
                if (!TryGetGroupSummaryMetrics(metrics, out gs))
                    throw ch.Except("Did not find group summary metrics");

                ch.Trace("Saving group-summary results");
                // If the data view contains stratification columns, filter so that only the overall metrics
                // will be present, and drop them.
                gs = MetricWriter.GetNonStratifiedMetrics(Host, gs);
                MetricWriter.SavePerInstance(Host, ch, _groupSummaryFilename, gs);
                ch.Done();
            }
        }

        private bool TryGetGroupSummaryMetrics(Dictionary<string, IDataView>[] metrics, out IDataView gs)
        {
            Host.AssertNonEmpty(metrics);

            if (metrics.Length == 1)
                return metrics[0].TryGetValue(RankerEvaluator.GroupSummary, out gs);

            gs = null;
            var gsList = new List<IDataView>();
            for (int i = 0; i < metrics.Length; i++)
            {
                IDataView idv;
                if (!metrics[i].TryGetValue(RankerEvaluator.GroupSummary, out idv))
                    return false;

                idv = EvaluateUtils.AddFoldIndex(Host, idv, i, metrics.Length);
                gsList.Add(idv);
            }
            gs = AppendRowsDataView.Create(Host, gsList[0].Schema, gsList.ToArray());
            return true;
        }

        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckValue(schema.Label, nameof(schema), "Data must contain a label column");
            Host.CheckValue(schema.Group, nameof(schema), "Data must contain a group column");

            // The ranking evaluator outputs the label, group key and score columns.
            yield return schema.Group.Name;
            yield return schema.Label.Name;
            var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.Ranking);
            yield return scoreInfo.Name;

            // Return the output columns.
            yield return RankerPerInstanceTransform.Ndcg;
            yield return RankerPerInstanceTransform.Dcg;
            yield return RankerPerInstanceTransform.MaxDcg;
        }
    }

    internal static class RankerUtils
    {
        private static volatile Double[] _discountMap;
        public static Double[] DiscountMap
        {
            get
            {
                if (_discountMap == null)
                {
                    var discountMap = new Double[100]; //Hard to believe anyone would set truncation Level higher than 100
                    for (int i = 0; i < discountMap.Length; i++)
                    {
                        discountMap[i] = 1 / Math.Log(2 + i);
                    }
                    Interlocked.CompareExchange(ref _discountMap, discountMap, null);
                }
                return _discountMap;
            }
        }

        /// <summary>te
        /// Calculates natural-based max DCG at all truncations from 1 to trunc
        /// </summary>
        public static void QueryMaxDcg(Double[] labelGains, int truncationLevel,
            List<short> queryLabels, List<Single> queryOutputs, Double[] groupMaxDcgCur)
        {
            Contracts.Assert(Utils.Size(groupMaxDcgCur) == truncationLevel);

            int relevancyLevel = labelGains.Length;

            int[] labelCounts = new int[relevancyLevel];

            int maxTrunc = Math.Min(truncationLevel, queryLabels.Count);

            if (maxTrunc == 0)
            {
                for (int t = 0; t < truncationLevel; t++)
                    groupMaxDcgCur[t] = Double.NaN;
            }
            else
            {
                for (int l = 0; l < queryLabels.Count; l++)
                    labelCounts[queryLabels[l]]++;

                int topLabel = labelGains.Length - 1;
                while (labelCounts[topLabel] == 0)
                    topLabel--;

                groupMaxDcgCur[0] = labelGains[topLabel] * DiscountMap[0];
                labelCounts[topLabel]--;
                for (int t = 1; t < maxTrunc; t++)
                {
                    while (labelCounts[topLabel] == 0)
                        topLabel--;
                    groupMaxDcgCur[t] = groupMaxDcgCur[t - 1] + labelGains[topLabel] * DiscountMap[t];
                    labelCounts[topLabel]--;
                }
                for (int t = maxTrunc; t < truncationLevel; t++)
                    groupMaxDcgCur[t] = groupMaxDcgCur[t - 1];
            }
        }

        public static void QueryDcg(Double[] labelGains, int truncationLevel,
            List<short> queryLabels, List<Single> queryOutputs, Double[] groupDcgCur)
        {
            // calculate the permutation
            int count = queryLabels.Count;
            int[] permutation = Utils.GetIdentityPermutation(count);
            Array.Sort(permutation, GetCompareItems(queryLabels, queryOutputs));

            if (count > truncationLevel)
                count = truncationLevel;
            Double dcg = 0;
            for (int t = 0; t < count; ++t)
            {
                dcg = dcg + labelGains[queryLabels[permutation[t]]] * DiscountMap[t];
                groupDcgCur[t] = dcg;
            }
            for (int t = count; t < truncationLevel; ++t)
                groupDcgCur[t] = dcg;
        }

        // Used for sorting.
        private static Comparison<int> GetCompareItems(List<short> queryLabels, List<Single> queryOutputs)
        {
            return
                (i, j) =>
                {
                    Contracts.Assert(0 <= i && i < queryLabels.Count && i < queryOutputs.Count);
                    Contracts.Assert(0 <= j && j < queryLabels.Count && j < queryOutputs.Count);

                    if (queryOutputs[i] > queryOutputs[j])
                        return -1;
                    if (queryOutputs[i] < queryOutputs[j])
                        return 1;
                    if (queryLabels[i] < queryLabels[j])
                        return -1;
                    if (queryLabels[i] > queryLabels[j])
                        return 1;
                    return i.CompareTo(j);

                };
        }
    }

    public static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.RankerEvaluator", Desc = "Evaluates a ranking scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput Ranking(IHostEnvironment env, RankerMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateRanker");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            ISchema schema = input.Data.Schema;
            string groupId = TrainUtils.MatchNameOrDefaultOrNull(host, schema,
                nameof(RankerMamlEvaluator.Arguments.GroupIdColumn),
                input.GroupIdColumn, DefaultColumnNames.GroupId);
            var evaluator = new RankerMamlEvaluator(host, input);
            var data = new RoleMappedData(input.Data, label, null, groupId, weight, name);
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
