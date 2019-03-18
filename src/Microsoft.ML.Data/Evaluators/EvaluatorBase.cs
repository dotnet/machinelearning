// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is a base class for TLC evaluators. It implements both of the <see cref="IEvaluator"/> methods: <see cref="Evaluate"/> and
    ///  <see cref="GetPerInstanceMetricsCore"/>. Note that the input <see cref="RoleMappedData"/> is assumed to contain all the column
    /// roles needed for evaluation, including the score column.
    /// </summary>
    [BestFriend]
    internal abstract partial class EvaluatorBase<TAgg> : IEvaluator
        where TAgg : EvaluatorBase<TAgg>.AggregatorBase
    {
        protected readonly IHost Host;

        [BestFriend]
        private protected EvaluatorBase(IHostEnvironment env, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
        }

        Dictionary<string, IDataView> IEvaluator.Evaluate(RoleMappedData data)
        {
            CheckColumnTypes(data.Schema);
            Func<int, bool> activeCols = GetActiveCols(data.Schema);
            var agg = GetAggregator(data.Schema);
            AggregatorDictionaryBase[] dictionaries = GetAggregatorDictionaries(data.Schema);

            var dict = ProcessData(data.Data, data.Schema, activeCols, agg, dictionaries);
            agg.GetWarnings(dict, Host);
            return dict;
        }

        /// <summary>
        /// Checks the column types of the evaluator's input columns. The base class implementation checks only the type
        /// of the weight column, and all other columns should be checked by the deriving classes in <see cref="CheckCustomColumnTypesCore"/>.
        /// </summary>
        [BestFriend]
        private protected void CheckColumnTypes(RoleMappedSchema schema)
        {
            // Check the weight column type.
            if (schema.Weight.HasValue)
                EvaluateUtils.CheckWeightType(Host, schema.Weight.Value.Type);
            CheckScoreAndLabelTypes(schema);
            // Check the other column types.
            CheckCustomColumnTypesCore(schema);
        }

        /// <summary>
        /// Check that the types of the score and label columns are as expected by the evaluator. The <see cref="RoleMappedSchema"/>
        /// is assumed to contain the label column (if it exists) and the score column.
        /// Access the label column with the <see cref="RoleMappedSchema.Label"/> property, and the score column with the
        /// <see cref="RoleMappedSchema.GetUniqueColumn"/> or <see cref="RoleMappedSchema.GetColumns"/> methods.
        /// </summary>
        [BestFriend]
        private protected abstract void CheckScoreAndLabelTypes(RoleMappedSchema schema);

        /// <summary>
        /// Check the types of any other columns needed by the evaluator. Only override if the evaluator uses
        /// columns other than label, score and weight.
        /// </summary>
        [BestFriend]
        private protected virtual void CheckCustomColumnTypesCore(RoleMappedSchema schema)
        {
        }

        private Func<int, bool> GetActiveCols(RoleMappedSchema schema)
        {
            Func<int, bool> pred = GetActiveColsCore(schema);
            var stratCols = schema.GetColumns(MamlEvaluatorBase.Strat);
            var stratIndices = Utils.Size(stratCols) > 0 ? new HashSet<int>(stratCols.Select(col => col.Index)) : new HashSet<int>();
            return i => pred(i) || stratIndices.Contains(i);
        }

        /// <summary>
        /// Used in the Evaluate() method, to get the predicate for cursoring over the data.
        /// The base class implementation activates the score column, the label column if it exists, the weight column if it exists
        /// and the stratification columns.
        /// Override if other input columns need to be activated.
        /// </summary>
        [BestFriend]
        private protected virtual Func<int, bool> GetActiveColsCore(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);
            int label = schema.Label?.Index ?? -1;
            int weight = schema.Weight?.Index ?? -1;
            return i => i == score.Index || i == label || i == weight;
        }

        /// <summary>
        /// Get an aggregator for the specific evaluator given the current RoleMappedSchema.
        /// </summary>
        private TAgg GetAggregator(RoleMappedSchema schema)
        {
            return GetAggregatorCore(schema, "");
        }

        /// <summary>
        /// For each stratification column, get an aggregator dictionary.
        /// </summary>
        private AggregatorDictionaryBase[] GetAggregatorDictionaries(RoleMappedSchema schema)
        {
            var list = new List<AggregatorDictionaryBase>();
            var stratCols = schema.GetColumns(MamlEvaluatorBase.Strat);
            if (Utils.Size(stratCols) > 0)
            {
                Func<string, TAgg> createAgg = stratName => GetAggregatorCore(schema, stratName);
                foreach (var stratCol in stratCols)
                    list.Add(AggregatorDictionaryBase.Create(schema, stratCol.Name, stratCol.Type, createAgg));
            }
            return list.ToArray();
        }

        [BestFriend]
        private protected abstract TAgg GetAggregatorCore(RoleMappedSchema schema, string stratName);

        // This method does as many passes over the data as needed by the evaluator, and computes the metrics, outputting the
        // results in a dictionary from the metric kind (overal/per-fold/confusion matrix/PR-curves etc.), to a data view containing
        // the metric. If there are stratified metrics, an additional column is added to the data view containing the
        // stratification value as text in the format "column x = y".
        private Dictionary<string, IDataView> ProcessData(IDataView data, RoleMappedSchema schema,
            Func<int, bool> activeColsIndices, TAgg aggregator, AggregatorDictionaryBase[] dictionaries)
        {
            Func<bool> finishPass =
                () =>
                {
                    var need = aggregator.FinishPass();
                    foreach (var agg in dictionaries.SelectMany(dict => dict.GetAll()))
                        need |= agg.FinishPass();
                    return need;
                };

            bool needMorePasses = aggregator.Start();

            var activeCols = data.Schema.Where(x => activeColsIndices(x.Index));
            // REVIEW: Add progress reporting.
            while (needMorePasses)
            {
                using (var cursor = data.GetRowCursor(activeCols))
                {
                    if (aggregator.IsActive())
                        aggregator.InitializeNextPass(cursor, schema);
                    for (int i = 0; i < Utils.Size(dictionaries); i++)
                    {
                        dictionaries[i].Reset(cursor);

                        foreach (var agg in dictionaries[i].GetAll())
                        {
                            if (agg.IsActive())
                                agg.InitializeNextPass(cursor, schema);
                        }
                    }
                    while (cursor.MoveNext())
                    {
                        if (aggregator.IsActive())
                            aggregator.ProcessRow();
                        for (int i = 0; i < Utils.Size(dictionaries); i++)
                        {
                            var agg = dictionaries[i].Get();
                            if (agg.IsActive())
                                agg.ProcessRow();
                        }
                    }
                }
                needMorePasses = finishPass();
            }

            Action<uint, ReadOnlyMemory<char>, TAgg> addAgg;
            Func<Dictionary<string, IDataView>> consolidate;
            GetAggregatorConsolidationFuncs(aggregator, dictionaries, out addAgg, out consolidate);

            uint stratColKey = 0;
            addAgg(stratColKey, default, aggregator);
            for (int i = 0; i < Utils.Size(dictionaries); i++)
            {
                var dict = dictionaries[i];
                stratColKey++;
                foreach (var agg in dict.GetAll())
                    addAgg(stratColKey, agg.StratName.AsMemory(), agg);
            }
            return consolidate();
        }

        /// <summary>
        /// This method returns two functions used to create the data views of metrics computed by the different
        /// aggregators (the overall one, and any stratified ones if they exist). The <paramref name="addAgg"/>
        /// function is called for every aggregator, and it is where the aggregators should finish their aggregations
        /// and the aggregator results should be stored. The <paramref name="consolidate"/> function
        /// is called after <paramref name="addAgg"/> has been called on all the aggregators, and it returns
        /// the dictionary of metric data views.
        /// </summary>
        [BestFriend]
        private protected abstract void GetAggregatorConsolidationFuncs(TAgg aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, TAgg> addAgg, out Func<Dictionary<string, IDataView>> consolidate);

        [BestFriend]
        private protected ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetKeyValueGetter(AggregatorDictionaryBase[] dictionaries)
        {
            if (Utils.Size(dictionaries) == 0)
                return null;
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, dictionaries.Length);
                    for (int i = 0; i < dictionaries.Length; i++)
                        editor.Values[i] = dictionaries[i].ColName.AsMemory();
                    dst = editor.Commit();
                };
        }

        IDataTransform IEvaluator.GetPerInstanceMetrics(RoleMappedData data) => GetPerInstanceMetricsCore(data);

        [BestFriend]
        internal abstract IDataTransform GetPerInstanceMetricsCore(RoleMappedData data);

        public abstract IEnumerable<MetricColumn> GetOverallMetricColumns();

        /// <summary>
        /// This is a helper class for evaluators deriving from EvaluatorBase, used for computing aggregate metrics.
        /// Aggregators should keep track of the number of passes done. The <see cref="InitializeNextPass"/> method should get
        /// the input getters of the given IRow that are needed for the current pass, assuming that all the needed column
        /// information is stored in the given <see cref="RoleMappedSchema"/>.
        /// In <see cref="ProcessRow"/> the aggregator should call the getters once, and process the input as needed.
        /// <see cref="FinishPass"/> increments the pass count after each pass.
        /// </summary>
        public abstract class AggregatorBase
        {
            public readonly string StratName;

            protected long NumUnlabeledInstances;
            protected long NumBadScores;
            protected long NumBadWeights;

            protected readonly IHost Host;

            protected int PassNum;

            [BestFriend]
            private protected AggregatorBase(IHostEnvironment env, string stratName)
            {
                Contracts.AssertValue(env);
                Host = env.Register("Aggregator");
                Host.AssertValueOrNull(stratName);

                PassNum = -1;
                StratName = stratName;
            }

            public bool Start()
            {
                Host.Check(PassNum == -1, "Start() should only be called before processing any data.");
                PassNum = 0;
                return IsActive();
            }

            /// <summary>
            /// This method should get the getters of the new IRow that are needed for the next pass.
            /// </summary>
            [BestFriend]
            internal abstract void InitializeNextPass(DataViewRow row, RoleMappedSchema schema);

            /// <summary>
            /// Call the getters once, and process the input as necessary.
            /// </summary>
            public abstract void ProcessRow();

            /// <summary>
            /// Increment the pass count. Return true if additional passes are needed.
            /// </summary>
            public bool FinishPass()
            {
                FinishPassCore();
                PassNum++;
                return IsActive();
            }

            // REVIEW: A more proper way to do this is to make this method protected, and have the AggregatorDictionary
            // class maintain the information about which aggregator is done and have the Get() method return either the appropriate aggregator
            // or null.
            public virtual bool IsActive()
            {
                return PassNum < 1;
            }

            protected virtual void FinishPassCore()
            {
                Host.Assert(PassNum < 1);
            }

            /// <summary>
            /// Returns a dictionary from metric kinds to data views containing the metrics.
            /// </summary>
            //public abstract Dictionary<string, IDataView> Finish();

            public void GetWarnings(Dictionary<string, IDataView> dict, IHostEnvironment env)
            {
                var warnings = GetWarningsCore();
                if (Utils.Size(warnings) > 0)
                {
                    var dvBldr = new ArrayDataViewBuilder(env);
                    dvBldr.AddColumn(MetricKinds.ColumnNames.WarningText, TextDataViewType.Instance,
                        warnings.Select(s => s.AsMemory()).ToArray());
                    dict.Add(MetricKinds.Warnings, dvBldr.GetDataView());
                }
            }

            protected virtual List<string> GetWarningsCore()
            {
                var warnings = new List<string>();
                if (NumUnlabeledInstances > 0)
                    warnings.Add(string.Format("Encountered {0} unlabeled instances during testing.", NumUnlabeledInstances));

                if (NumBadWeights > 0)
                    warnings.Add(string.Format("Encountered {0} non-finite weights during testing. These weights have been replaced with 1.", NumBadWeights));

                if (NumBadScores > 0)
                {
                    warnings.Add(string.Format("The predictor produced non-finite prediction values on {0} instances during testing. " +
                        "Possible causes: abnormal data or the predictor is numerically unstable.", NumBadScores));
                }
                return warnings;
            }
        }

        // This class is a dictionary for aggregators that are used to compute aggregate metrics on stratified subsets
        // of the data. The dictionary holds a getter for a stratification column, and when the Get() method is called,
        // it calls this getter, and returns the appropriate aggregator based on the value in the stratification column.
        // When a new value is encountered, it uses a callback for creating a new aggregator.
        protected abstract class AggregatorDictionaryBase
        {
            private protected DataViewRow Row;
            private protected readonly Func<string, TAgg> CreateAgg;
            private protected readonly RoleMappedSchema Schema;

            public string ColName { get; }

            public abstract int Count { get; }

            private protected AggregatorDictionaryBase(RoleMappedSchema schema, string stratCol, Func<string, TAgg> createAgg)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertNonWhiteSpace(stratCol);
                Contracts.AssertValue(createAgg);

                Schema = schema;
                CreateAgg = createAgg;
                ColName = stratCol;
            }

            /// <summary>
            /// Gets the stratification column getter for the new IRow.
            /// </summary>
            public abstract void Reset(DataViewRow row);

            internal static AggregatorDictionaryBase Create(RoleMappedSchema schema, string stratCol, DataViewType stratType,
                Func<string, TAgg> createAgg)
            {
                Contracts.AssertNonWhiteSpace(stratCol);
                Contracts.AssertValue(createAgg);

                if (stratType.GetKeyCount() == 0 && !(stratType is TextDataViewType))
                {
                    throw Contracts.ExceptUserArg(nameof(MamlEvaluatorBase.ArgumentsBase.StratColumns),
                        "Stratification column '{stratCol}' has type '{stratType}', but must be a known count key or text");
                }
                return Utils.MarshalInvoke(CreateDictionary<int>, stratType.RawType, schema, stratCol, stratType, createAgg);
            }

            private static AggregatorDictionaryBase CreateDictionary<TStrat>(RoleMappedSchema schema, string stratCol,
                DataViewType stratType, Func<string, TAgg> createAgg)
            {
                return new GenericAggregatorDictionary<TStrat>(schema, stratCol, stratType, createAgg);
            }

            /// <summary>
            /// This method calls the getter of the stratification column, and returns the aggregator corresponding to
            /// the stratification value.
            /// </summary>
            /// <returns></returns>
            public abstract TAgg Get();

            /// <summary>
            /// This method returns the aggregators corresponding to all the stratification values seen so far.
            /// </summary>
            public abstract IEnumerable<TAgg> GetAll();

            private sealed class GenericAggregatorDictionary<TStrat> : AggregatorDictionaryBase
            {
                private readonly Dictionary<TStrat, TAgg> _dict;
                private ValueGetter<TStrat> _stratGetter;

                // This is used to get the current stratification value in the Get() method.
                private TStrat _value;

                public override int Count => _dict.Count;

                public GenericAggregatorDictionary(RoleMappedSchema schema, string stratCol, DataViewType stratType, Func<string, TAgg> createAgg)
                    : base(schema, stratCol, createAgg)
                {
                    Contracts.Assert(stratType.RawType == typeof(TStrat));
                    _dict = new Dictionary<TStrat, TAgg>();
                }

                public override void Reset(DataViewRow row)
                {
                    Row = row;
                    var col = row.Schema.GetColumnOrNull(ColName);
                    Contracts.Assert(col.HasValue);
                    _stratGetter = row.GetGetter<TStrat>(col.Value);
                    Contracts.AssertValue(_stratGetter);
                }

                public override TAgg Get()
                {
                    _stratGetter(ref _value);

                    TAgg agg;
                    if (!_dict.TryGetValue(_value, out agg))
                    {
                        // REVIEW: Consider adding a specific implementation for key types
                        // that would call _createAgg with the key value instead of the raw value.
                        agg = CreateAgg(_value.ToString());
                        agg.Start();
                        agg.InitializeNextPass(Row, Schema);
                        _dict.Add(_value, agg);
                    }
                    return agg;
                }

                public override IEnumerable<TAgg> GetAll()
                {
                    return _dict.Select(kvp => kvp.Value);
                }
            }
        }
    }

    [BestFriend]
    internal abstract class RowToRowEvaluatorBase<TAgg> : EvaluatorBase<TAgg>
        where TAgg : EvaluatorBase<TAgg>.AggregatorBase
    {
        [BestFriend]
        private protected RowToRowEvaluatorBase(IHostEnvironment env, string registrationName)
            : base(env, registrationName)
        {
        }

        internal override IDataTransform GetPerInstanceMetricsCore(RoleMappedData data)
        {
            var mapper = CreatePerInstanceRowMapper(data.Schema);
            return new RowToRowMapperTransform(Host, data.Data, mapper, null);
        }

        [BestFriend]
        private protected abstract IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema);
    }

    /// <summary>
    /// This is a helper class for creating the per-instance IDV.
    /// </summary>
    [BestFriend]
    internal abstract class PerInstanceEvaluatorBase : IRowMapper
    {
        protected readonly IHost Host;
        protected readonly string ScoreCol;
        protected readonly string LabelCol;
        protected readonly int ScoreIndex;
        protected readonly int LabelIndex;

        protected PerInstanceEvaluatorBase(IHostEnvironment env, DataViewSchema schema, string scoreCol, string labelCol)
        {
            Contracts.AssertValue(env);
            Contracts.AssertNonEmpty(scoreCol);

            Host = env.Register("PerInstanceRowMapper");
            ScoreCol = scoreCol;
            LabelCol = labelCol;

            if (!string.IsNullOrEmpty(LabelCol) && !schema.TryGetColumnIndex(LabelCol, out LabelIndex))
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", LabelCol);
            if (!schema.TryGetColumnIndex(ScoreCol, out ScoreIndex))
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", ScoreCol);
        }

        protected PerInstanceEvaluatorBase(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema schema)
        {
            Host = env.Register("PerInstanceRowMapper");

            // *** Binary format **
            // int: Id of the score column name
            // int: Id of the label column name

            ScoreCol = ctx.LoadNonEmptyString();
            LabelCol = ctx.LoadStringOrNull();
            if (!string.IsNullOrEmpty(LabelCol) && !schema.TryGetColumnIndex(LabelCol, out LabelIndex))
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", LabelCol);
            if (!schema.TryGetColumnIndex(ScoreCol, out ScoreIndex))
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", ScoreCol);
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        /// <summary>
        /// Derived class, for example A, should overwrite <see cref="SaveModel"/> so that ((<see cref="ICanSaveModel"/>)A).Save(ctx) can correctly dump A.
        /// </summary>
        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            // *** Binary format **
            // int: Id of the score column name
            // int: Id of the label column name

            ctx.SaveNonEmptyString(ScoreCol);
            ctx.SaveStringOrNull(LabelCol);
        }

        Func<int, bool> IRowMapper.GetDependencies(Func<int, bool> activeOutput)
            => GetDependenciesCore(activeOutput);

        [BestFriend]
        private protected abstract Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput);

        DataViewSchema.DetachedColumn[] IRowMapper.GetOutputColumns()
            => GetOutputColumnsCore();

        [BestFriend]
        private protected abstract DataViewSchema.DetachedColumn[] GetOutputColumnsCore();

        Delegate[] IRowMapper.CreateGetters(DataViewRow input, Func<int, bool> activeCols, out Action disposer)
            => CreateGettersCore(input, activeCols, out disposer);

        [BestFriend]
        private protected abstract Delegate[] CreateGettersCore(DataViewRow input, Func<int, bool> activeCols, out Action disposer);

        public ITransformer GetTransformer()
        {
            throw Host.ExceptNotSupp();
        }
    }
}
