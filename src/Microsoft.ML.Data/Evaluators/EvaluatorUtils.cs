// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal static class EvaluateUtils
    {
        public struct AggregatedMetric
        {
            public double Sum;
            public double SumSq;
            public string Name;
        }

        private static class DefaultEvaluatorTable
        {
            private static volatile Dictionary<string, Func<IHostEnvironment, IMamlEvaluator>> _knownEvaluatorFactories;

            public static Dictionary<string, Func<IHostEnvironment, IMamlEvaluator>> Instance
            {
                get
                {
                    Dictionary<string, Func<IHostEnvironment, IMamlEvaluator>> result = _knownEvaluatorFactories;
                    if (result == null)
                    {
                        var tmp = new Dictionary<string, Func<IHostEnvironment, IMamlEvaluator>>
                        {
                            { AnnotationUtils.Const.ScoreColumnKind.BinaryClassification, env => new BinaryClassifierMamlEvaluator(env, new BinaryClassifierMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification, env => new MulticlassClassificationMamlEvaluator(env, new MulticlassClassificationMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.Regression, env => new RegressionMamlEvaluator(env, new RegressionMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.MultiOutputRegression, env => new MultiOutputRegressionMamlEvaluator(env, new MultiOutputRegressionMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.QuantileRegression, env => new QuantileRegressionMamlEvaluator(env, new QuantileRegressionMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.Ranking, env => new RankingMamlEvaluator(env, new RankingMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.Clustering, env => new ClusteringMamlEvaluator(env, new ClusteringMamlEvaluator.Arguments()) },
                            { AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection, env => new AnomalyDetectionMamlEvaluator(env, new AnomalyDetectionMamlEvaluator.Arguments()) }
                        };
                        //tmp.Add(MetadataUtils.Const.ScoreColumnKind.SequenceClassification, "SequenceClassifierEvaluator");
                        Interlocked.CompareExchange(ref _knownEvaluatorFactories, tmp, null);
                        result = _knownEvaluatorFactories;
                    }
                    return result;
                }
            }
        }

        public static IMamlEvaluator GetEvaluator(IHostEnvironment env, DataViewSchema schema)
        {
            Contracts.CheckValueOrNull(env);
            ReadOnlyMemory<char> tmp = default;
            schema.GetMaxAnnotationKind(out int col, AnnotationUtils.Kinds.ScoreColumnSetId, CheckScoreColumnKindIsKnown);
            if (col >= 0)
            {
                schema[col].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref tmp);
                var kind = tmp.ToString();
                var map = DefaultEvaluatorTable.Instance;
                // The next assert is guaranteed because it is checked in CheckScoreColumnKindIsKnown which is the lambda passed to GetMaxMetadataKind.
                Contracts.Assert(map.ContainsKey(kind));
                return map[kind](env);
            }

            schema.GetMaxAnnotationKind(out col, AnnotationUtils.Kinds.ScoreColumnSetId, CheckScoreColumnKind);
            if (col >= 0)
            {
                schema[col].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref tmp);
                throw env.ExceptUserArg(nameof(EvaluateCommand.Arguments.Evaluator), "No default evaluator found for score column kind '{0}'.", tmp.ToString());
            }

            throw env.ExceptParam(nameof(schema), "No score columns have been automatically detected.");
        }

        // Lambda used as validator/filter in calls to GetMaxMetadataKind.
        private static bool CheckScoreColumnKindIsKnown(DataViewSchema schema, int col)
        {
            var columnType = schema[col].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnKind)?.Type;
            if (columnType == null || !(columnType is TextDataViewType))
                return false;
            ReadOnlyMemory<char> tmp = default;
            schema[col].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref tmp);
            var map = DefaultEvaluatorTable.Instance;
            return map.ContainsKey(tmp.ToString());
        }

        // Lambda used as validator/filter in calls to GetMaxMetadataKind.
        private static bool CheckScoreColumnKind(DataViewSchema schema, int col)
        {
            var columnType = schema[col].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnKind)?.Type;
            return columnType != null && columnType is TextDataViewType;
        }

        /// <summary>
        /// Find the score column to use. If <paramref name="name"/> is specified, that is used. Otherwise, this searches
        /// for the most recent score set of the given <paramref name="kind"/>. If there is no such score set and
        /// <paramref name="defName"/> is specifed it uses <paramref name="defName"/>. Otherwise, it throws.
        /// </summary>
        public static DataViewSchema.Column GetScoreColumn(IExceptionContext ectx, DataViewSchema schema, string name, string argName, string kind,
            string valueKind = AnnotationUtils.Const.ScoreValueKind.Score, string defName = null)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValueOrNull(name);
            ectx.CheckNonEmpty(argName, nameof(argName));
            ectx.CheckNonEmpty(kind, nameof(kind));
            ectx.CheckNonEmpty(valueKind, nameof(valueKind));

            if (!string.IsNullOrWhiteSpace(name))
            {
#pragma warning disable MSML_ContractsNameUsesNameof // This utility method is meant to reflect the argument name of whatever is calling it, so we take that as a parameter, rather than using nameof directly as in most cases.
                var col = schema.GetColumnOrNull(name);
                if (!col.HasValue)
                    throw ectx.ExceptUserArg(argName, "Score column is missing");
#pragma warning restore MSML_ContractsNameUsesNameof
                return col.Value;
            }

            var maxSetNum = schema.GetMaxAnnotationKind(out int colTmp, AnnotationUtils.Kinds.ScoreColumnSetId,
                (s, c) => IsScoreColumnKind(ectx, s, c, kind));

            ReadOnlyMemory<char> tmp = default;
            foreach (var colIdx in schema.GetColumnSet(AnnotationUtils.Kinds.ScoreColumnSetId, maxSetNum))
            {
                var col = schema[colIdx];
#if DEBUG
                col.Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref tmp);
                ectx.Assert(ReadOnlyMemoryUtils.EqualsStr(kind, tmp));
#endif
                // REVIEW: What should this do about hidden columns? Currently we ignore them.
                if (col.IsHidden)
                    continue;
                if (col.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreValueKind)?.Type == TextDataViewType.Instance)
                {
                    col.Annotations.GetValue(AnnotationUtils.Kinds.ScoreValueKind, ref tmp);
                    if (ReadOnlyMemoryUtils.EqualsStr(valueKind, tmp))
                        return col;
                }
            }

            if (!string.IsNullOrWhiteSpace(defName) && schema.GetColumnOrNull(defName) is DataViewSchema.Column defCol)
                return defCol;

#pragma warning disable MSML_ContractsNameUsesNameof
            throw ectx.ExceptUserArg(argName, "Score column is missing");
#pragma warning restore MSML_ContractsNameUsesNameof
        }

        /// <summary>
        /// Find the optional auxilliary score column to use. If <paramref name="name"/> is specified, that is used.
        /// Otherwise, if <paramref name="colScore"/> is part of a score set, this looks in the score set for a column
        /// with the given <paramref name="valueKind"/>. If none is found, it returns <see langword="null"/>.
        /// </summary>
        public static DataViewSchema.Column? GetOptAuxScoreColumn(IExceptionContext ectx, DataViewSchema schema, string name, string argName,
            int colScore, string valueKind, Func<DataViewType, bool> testType)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValueOrNull(name);
            ectx.CheckNonEmpty(argName, nameof(argName));
            ectx.CheckParam(0 <= colScore && colScore < schema.Count, nameof(colScore));
            ectx.CheckNonEmpty(valueKind, nameof(valueKind));

            if (!string.IsNullOrWhiteSpace(name))
            {
#pragma warning disable MSML_ContractsNameUsesNameof
                var col = schema.GetColumnOrNull(name);
                if (!col.HasValue)
                    throw ectx.ExceptUserArg(argName, "{0} column is missing", valueKind);
                if (!testType(col.Value.Type))
                    throw ectx.ExceptUserArg(argName, "{0} column has incompatible type", valueKind);
#pragma warning restore MSML_ContractsNameUsesNameof
                return col.Value;
            }

            // Get the score column set id from colScore.
            var type = schema[colScore].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnSetId)?.Type;
            if (!(type is KeyType) || type.RawType != typeof(uint))
            {
                // scoreCol is not part of a score column set, so can't determine an aux column.
                return null;
            }
            uint setId = 0;
            schema[colScore].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnSetId, ref setId);

            ReadOnlyMemory<char> tmp = default;
            foreach (var colIdx in schema.GetColumnSet(AnnotationUtils.Kinds.ScoreColumnSetId, setId))
            {
                // REVIEW: What should this do about hidden columns? Currently we ignore them.
                var col = schema[colIdx];
                if (col.IsHidden)
                    continue;

                if (col.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreValueKind)?.Type == TextDataViewType.Instance)
                {
                    col.Annotations.GetValue(AnnotationUtils.Kinds.ScoreValueKind, ref tmp);
                    if (ReadOnlyMemoryUtils.EqualsStr(valueKind, tmp) && testType(col.Type))
                        return col;
                }
            }

            // Didn't find it in the score column set.
            return null;
        }

        private static bool IsScoreColumnKind(IExceptionContext ectx, DataViewSchema schema, int col, string kind)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckParam(0 <= col && col < schema.Count, nameof(col));
            ectx.CheckNonEmpty(kind, nameof(kind));

            var type = schema[col].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnKind)?.Type;
            if (type == null || !(type is TextDataViewType))
                return false;
            var tmp = default(ReadOnlyMemory<char>);
            schema[col].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref tmp);
            return ReadOnlyMemoryUtils.EqualsStr(kind, tmp);
        }

        /// <summary>
        /// If <paramref name="str"/> is non-empty, returns it. Otherwise if <paramref name="info"/> is non-<see langword="null"/>,
        /// returns its <see cref="DataViewSchema.Column.Name"/>. Otherwise, returns <paramref name="def"/>.
        /// </summary>
        public static string GetColName(string str, DataViewSchema.Column? info, string def)
        {
            Contracts.CheckValueOrNull(str);
            Contracts.CheckValueOrNull(def);

            if (!string.IsNullOrEmpty(str))
                return str;
            return info?.Name ?? def;
        }

        public static void CheckWeightType(IExceptionContext ectx, DataViewType type)
        {
            ectx.AssertValue(type);
            if (type != NumberDataViewType.Single)
                throw ectx.ExceptUserArg(nameof(EvaluateCommand.Arguments.WeightColumn), "Incompatible Weight column. Weight column type must be {0}.", NumberDataViewType.Single);
        }

        /// <summary>
        /// Helper method to get an IEnumerable of double metrics from an overall metrics IDV produced by an evaluator.
        /// </summary>
        public static IEnumerable<KeyValuePair<string, double>> GetMetrics(IDataView metricsView, bool getVectorMetrics = true)
        {
            Contracts.CheckValue(metricsView, nameof(metricsView));
            var schema = metricsView.Schema;

            // Figure out whether there is an "IsWeighted" column.
            int isWeightedCol;
            var hasWeighted = schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out isWeightedCol);

            // Figure out whether there are stratification columns.
            int stratCol;
            int stratVal = -1;
            bool hasStrats;
            if (hasStrats = schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol))
            {
                if (!schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal))
                {
                    throw Contracts.Except("If data contains a '{0}' column, it must also contain a '{1}' column",
                        MetricKinds.ColumnNames.StratCol, MetricKinds.ColumnNames.StratVal);
                }
            }

            using (var cursor = metricsView.GetRowCursorForAllColumns())
            {
                bool isWeighted = false;
                ValueGetter<bool> isWeightedGetter;
                if (hasWeighted)
                    isWeightedGetter = cursor.GetGetter<bool>(schema[isWeightedCol]);
                else
                    isWeightedGetter = (ref bool dst) => dst = false;

                ValueGetter<uint> stratColGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema[stratCol].Type;
                    stratColGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratColGetter = (ref uint dst) => dst = 0;

                // We currently have only double valued or vector of double valued metrics.
                var colCount = schema.Count;
                var getters = new ValueGetter<double>[colCount];
                var vBufferGetters = getVectorMetrics ? new ValueGetter<VBuffer<double>>[colCount] : null;

                for (int i = 0; i < schema.Count; i++)
                {
                    var column = schema[i];
                    if (column.IsHidden || hasWeighted && i == isWeightedCol ||
                        hasStrats && (i == stratCol || i == stratVal))
                        continue;

                    var type = schema[i].Type;
                    if (type == NumberDataViewType.Double || type == NumberDataViewType.Single)
                        getters[i] = RowCursorUtils.GetGetterAs<double>(NumberDataViewType.Double, cursor, i);
                    else if (type is VectorType vectorType
                        && vectorType.IsKnownSize
                        && vectorType.ItemType == NumberDataViewType.Double
                        && getVectorMetrics)
                        vBufferGetters[i] = cursor.GetGetter<VBuffer<double>>(column);
                }

                Double metricVal = 0;
                VBuffer<double> metricVals = default(VBuffer<double>);
                uint strat = 0;
                bool foundRow = false;
                while (cursor.MoveNext())
                {
                    isWeightedGetter(ref isWeighted);
                    if (isWeighted)
                        continue;

                    stratColGetter(ref strat);
                    if (strat > 0)
                        continue;

                    // There should only be one row where isWeighted is false and strat=0.
                    Contracts.Check(!foundRow, "Multiple metric rows found in metrics data view.");

                    foundRow = true;
                    for (int i = 0; i < colCount; i++)
                    {
                        if (hasWeighted && i == isWeightedCol || hasStrats && (i == stratCol || i == stratVal))
                            continue;

                        if (getters[i] != null)
                        {
                            getters[i](ref metricVal);
                            // For R8 valued columns the metric name is the column name.
                            yield return new KeyValuePair<string, double>(schema[i].Name, metricVal);
                        }
                        else if (getVectorMetrics && vBufferGetters[i] != null)
                        {
                            vBufferGetters[i](ref metricVals);

                            // For R8 vector valued columns the names of the metrics are the column name,
                            // followed by the slot name if it exists, or "Label_i" if it doesn't.
                            VBuffer<ReadOnlyMemory<char>> names = default;
                            var size = schema[i].Type.GetVectorSize();
                            var slotNamesType = schema[i].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType;
                            if (slotNamesType != null && slotNamesType.Size == size && slotNamesType.ItemType is TextDataViewType)
                                schema[i].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref names);
                            else
                            {
                                var namesArray = new ReadOnlyMemory<char>[size];
                                for (int j = 0; j < size; j++)
                                    namesArray[j] = string.Format("({0})", j).AsMemory();
                                names = new VBuffer<ReadOnlyMemory<char>>(size, namesArray);
                            }
                            var colName = schema[i].Name;
                            foreach (var metric in metricVals.Items(all: true))
                            {
                                yield return new KeyValuePair<string, double>(
                                    string.Format("{0} {1}", colName, names.GetItemOrDefault(metric.Key)), metric.Value);
                            }
                        }
                    }
                }
            }
        }

        private static IDataView AddTextColumn<TSrc>(IHostEnvironment env, IDataView input, string inputColName, string outputColName,
            DataViewType typeSrc, string value, string registrationName)
        {
            Contracts.Check(typeSrc.RawType == typeof(TSrc));
            return LambdaColumnMapper.Create(env, registrationName, input, inputColName, outputColName, typeSrc, TextDataViewType.Instance,
                (in TSrc src, ref ReadOnlyMemory<char> dst) => dst = value.AsMemory());
        }

        /// <summary>
        /// Add a text column containing a fold index to a data view.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="input">The data view to which we add the column</param>
        /// <param name="curFold">The current fold this data view belongs to.</param>
        /// <returns>The input data view with an additional text column containing the current fold index.</returns>
        public static IDataView AddFoldIndex(IHostEnvironment env, IDataView input, int curFold)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.CheckParam(curFold >= 0, nameof(curFold));

            // We use the first column in the data view as an input column to the LambdaColumnMapper,
            // because it must have an input.
            int inputCol = 0;
            while (inputCol < input.Schema.Count && input.Schema[inputCol].IsHidden)
                inputCol++;
            env.Assert(inputCol < input.Schema.Count);

            var inputColName = input.Schema[0].Name;
            var inputColType = input.Schema[0].Type;
            return Utils.MarshalInvoke(AddTextColumn<int>, inputColType.RawType, env,
                input, inputColName, MetricKinds.ColumnNames.FoldIndex, inputColType, $"Fold {curFold}", "FoldName");
        }

        private static IDataView AddKeyColumn<TSrc>(IHostEnvironment env, IDataView input, string inputColName, string outputColName,
            DataViewType typeSrc, int keyCount, int value, string registrationName, ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter)
        {
            Contracts.Check(typeSrc.RawType == typeof(TSrc));
            return LambdaColumnMapper.Create(env, registrationName, input, inputColName, outputColName, typeSrc,
                new KeyType(typeof(uint), keyCount), (in TSrc src, ref uint dst) =>
                {
                    if (value < 0 || value > keyCount)
                        dst = 0;
                    else
                        dst = (uint)value;
                }, keyValueGetter);
        }

        /// <summary>
        /// Add a key type column containing a fold index to a data view.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="input">The data view to which we add the column</param>
        /// <param name="curFold">The current fold this data view belongs to.</param>
        /// <param name="numFolds">The total number of folds.</param>
        /// <returns>The input data view with an additional key type column containing the current fold index.</returns>
        public static IDataView AddFoldIndex(IHostEnvironment env, IDataView input, int curFold, int numFolds)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.CheckParam(curFold >= 0, nameof(curFold));
            env.CheckParam(numFolds > 0, nameof(numFolds));

            // We use the first column in the data view as an input column to the LambdaColumnMapper,
            // because it must have an input.
            int inputCol = 0;
            while (inputCol < input.Schema.Count && input.Schema[inputCol].IsHidden)
                inputCol++;
            env.Assert(inputCol < input.Schema.Count);

            var inputColName = input.Schema[inputCol].Name;
            var inputColType = input.Schema[inputCol].Type;
            return Utils.MarshalInvoke(AddKeyColumn<int>, inputColType.RawType, env,
                input, inputColName, MetricKinds.ColumnNames.FoldIndex,
                inputColType, numFolds, curFold + 1, "FoldIndex", default(ValueGetter<VBuffer<ReadOnlyMemory<char>>>));
        }

        /// <summary>
        /// This method takes an array of data views and a specified input vector column, and adds a new output column to each of the data views.
        /// First, we find the union set of the slot names in the different data views. Next we define a new vector column for each
        /// data view, indexed by the union of the slot names. For each data view, every slot value is the value in the slot corresponding
        /// to its slot name in the original column. If a reconciled slot name does not exist in an input column, the value in the output
        /// column is def.
        /// </summary>
        public static void ReconcileSlotNames<T>(IHostEnvironment env, IDataView[] views, string columnName, PrimitiveDataViewType itemType, T def = default(T))
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckValue(itemType, nameof(itemType));
            Contracts.CheckParam(typeof(T) == itemType.RawType, nameof(itemType), "Generic type does not match the item type");

            var numIdvs = views.Length;
            var slotNames = new Dictionary<string, int>();
            var maps = new int[numIdvs][];
            var slotNamesCur = default(VBuffer<ReadOnlyMemory<char>>);
            var typeSrc = new DataViewType[numIdvs];
            // Create mappings from the original slots to the reconciled slots.
            for (int i = 0; i < numIdvs; i++)
            {
                var idv = views[i];
                int col;
                if (!idv.Schema.TryGetColumnIndex(columnName, out col))
                    throw env.Except("Data view number {0} does not contain column '{1}'", i, columnName);

                var type = typeSrc[i] = idv.Schema[col].Type;
                if (!idv.Schema[col].HasSlotNames(type.GetVectorSize()))
                    throw env.Except("Column '{0}' in data view number {1} did not contain slot names metadata", columnName, i);
                idv.Schema[col].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref slotNamesCur);

                var map = maps[i] = new int[slotNamesCur.Length];
                foreach (var kvp in slotNamesCur.Items(true))
                {
                    var index = kvp.Key;
                    var name = kvp.Value.ToString();
                    if (!slotNames.ContainsKey(name))
                        slotNames[name] = slotNames.Count;
                    map[index] = slotNames[name];
                }
            }

            var reconciledSlotNames = new VBuffer<ReadOnlyMemory<char>>(slotNames.Count, slotNames.Keys.Select(k => k.AsMemory()).ToArray());
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter =
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    reconciledSlotNames.CopyTo(ref dst);
                };

            // For each input data view, create the reconciled key column by wrapping it in a LambdaColumnMapper.
            for (int i = 0; i < numIdvs; i++)
            {
                var map = maps[i];

                ValueMapper<VBuffer<T>, VBuffer<T>> mapper;
                if (def.Equals(default(T)))
                {
                    mapper =
                        (in VBuffer<T> src, ref VBuffer<T> dst) =>
                        {
                            Contracts.Assert(src.Length == Utils.Size(map));
                            var editor = VBufferEditor.Create(ref dst, slotNames.Count);

                            foreach (var kvp in src.Items())
                                editor.Values[map[kvp.Key]] = kvp.Value;
                            dst = editor.Commit();
                        };
                }
                else
                {
                    // Create a list of the slots in the reconciled output column that do not correspond to any slots
                    // in the input column, so we can populate them with NAs.
                    var mappedIndices = new bool[slotNames.Count];
                    for (int j = 0; j < map.Length; j++)
                        mappedIndices[map[j]] = true;
                    var naIndices = new List<int>();
                    for (int j = 0; j < mappedIndices.Length; j++)
                    {
                        if (!mappedIndices[j])
                            naIndices.Add(j);
                    }
                    mapper =
                        (in VBuffer<T> src, ref VBuffer<T> dst) =>
                        {
                            Contracts.Assert(src.Length == Utils.Size(map));
                            var editor = VBufferEditor.Create(ref dst, slotNames.Count);

                            foreach (var kvp in src.Items(true))
                                editor.Values[map[kvp.Key]] = kvp.Value;
                            foreach (var j in naIndices)
                                editor.Values[j] = def;
                            dst = editor.Commit();
                        };
                }

                var typeDst = new VectorType(itemType, slotNames.Count);
                views[i] = LambdaColumnMapper.Create(env, "ReconciledSlotNames", views[i],
                    columnName, columnName, typeSrc[i], typeDst, mapper, slotNamesGetter: slotNamesGetter);
            }
        }

        private static int[][] MapKeys<T>(DataViewSchema[] schemas, string columnName, bool isVec,
            int[] indices, Dictionary<ReadOnlyMemory<char>, int> reconciledKeyNames)
        {
            Contracts.AssertValue(indices);
            Contracts.AssertValue(reconciledKeyNames);

            var dvCount = schemas.Length;
            var keyValueMappers = new int[dvCount][];
            var keyNamesCur = default(VBuffer<T>);
            for (int i = 0; i < dvCount; i++)
            {
                var schema = schemas[i];
                if (!schema.TryGetColumnIndex(columnName, out indices[i]))
                    throw Contracts.Except($"Schema number {i} does not contain column '{columnName}'");

                var type = schema[indices[i]].Type;
                var keyValueType = schema[indices[i]].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type;
                VectorType vectorType = type as VectorType;
                bool typeIsVector = vectorType != null;
                if (typeIsVector != isVec)
                    throw Contracts.Except($"Column '{columnName}' in schema number {i} does not have the correct type");
                DataViewType keyValueItemType = (keyValueType as VectorType)?.ItemType ?? keyValueType;
                if (keyValueItemType == null || keyValueItemType.RawType != typeof(T))
                    throw Contracts.Except($"Column '{columnName}' in schema number {i} does not have the correct type of key values");
                DataViewType typeItemType = vectorType?.ItemType ?? type;
                if (!(typeItemType is KeyType itemKeyType) || typeItemType.RawType != typeof(uint))
                    throw Contracts.Except($"Column '{columnName}' must be a U4 key type, but is '{typeItemType}'");

                schema[indices[i]].GetKeyValues(ref keyNamesCur);

                keyValueMappers[i] = new int[itemKeyType.Count];
                foreach (var kvp in keyNamesCur.Items(true))
                {
                    var key = kvp.Key;
                    var name = kvp.Value.ToString().AsMemory();
                    if (!reconciledKeyNames.ContainsKey(name))
                        reconciledKeyNames[name] = reconciledKeyNames.Count;
                    keyValueMappers[i][key] = reconciledKeyNames[name];
                }
            }
            return keyValueMappers;
        }

        /// <summary>
        /// This method takes an array of data views and a specified input key column, and adds a new output column to each of the data views.
        /// First, we find the union set of the key values in the different data views. Next we define a new key column for each
        /// data view, with the union of the key values as the new key values. For each data view, the value in the output column is the value
        /// corresponding to the key value in the original column.
        /// </summary>
        public static void ReconcileKeyValues(IHostEnvironment env, IDataView[] views, string columnName, DataViewType keyValueType)
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckNonEmpty(columnName, nameof(columnName));

            var dvCount = views.Length;

            // Create mappings from the original key types to the reconciled key type.
            var indices = new int[dvCount];
            var keyNames = new Dictionary<ReadOnlyMemory<char>, int>();
            // We use MarshalInvoke so that we can call MapKeys with the correct generic: keyValueType.RawType.
            var keyValueMappers = Utils.MarshalInvoke(MapKeys<int>, keyValueType.RawType, views.Select(view => view.Schema).ToArray(), columnName, false, indices, keyNames);
            var keyType = new KeyType(typeof(uint), keyNames.Count);
            var keyNamesVBuffer = new VBuffer<ReadOnlyMemory<char>>(keyNames.Count, keyNames.Keys.ToArray());
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter =
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        keyNamesVBuffer.CopyTo(ref dst);

            // For each input data view, create the reconciled key column by wrapping it in a LambdaColumnMapper.
            for (int i = 0; i < dvCount; i++)
            {
                var keyMapperCur = keyValueMappers[i];
                ValueMapper<uint, uint> mapper =
                    (in uint src, ref uint dst) =>
                    {
                        if (src == 0 || src > keyMapperCur.Length)
                            dst = 0;
                        else
                            dst = (uint)keyMapperCur[src - 1] + 1;
                    };
                views[i] = LambdaColumnMapper.Create(env, "ReconcileKeyValues", views[i], columnName, columnName,
                    views[i].Schema[indices[i]].Type, keyType, mapper, keyValueGetter);
            }
        }

        /// <summary>
        /// This method takes an array of data views and a specified input key column, and adds a new output column to each of the data views.
        /// First, we find the union set of the key values in the different data views. Next we define a new key column for each
        /// data view, with the union of the key values as the new key values. For each data view, the value in the output column is the value
        /// corresponding to the key value in the original column.
        /// </summary>
        public static void ReconcileKeyValuesWithNoNames(IHostEnvironment env, IDataView[] views, string columnName, ulong keyCount)
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckNonEmpty(columnName, nameof(columnName));

            var keyType = new KeyType(typeof(uint), keyCount);

            // For each input data view, create the reconciled key column by wrapping it in a LambdaColumnMapper.
            for (int i = 0; i < views.Length; i++)
            {
                if (!views[i].Schema.TryGetColumnIndex(columnName, out var index))
                    throw env.Except($"Data view {i} doesn't contain a column '{columnName}'");
                ValueMapper<uint, uint> mapper =
                    (in uint src, ref uint dst) =>
                    {
                        if (src > keyCount)
                            dst = 0;
                        else
                            dst = src;
                    };
                views[i] = LambdaColumnMapper.Create(env, "ReconcileKeyValues", views[i], columnName, columnName,
                    views[i].Schema[index].Type, keyType, mapper);
            }
        }

        /// <summary>
        /// This method is similar to <see cref="ReconcileKeyValues"/>, but it reconciles the key values over vector
        /// input columns.
        /// </summary>
        public static void ReconcileVectorKeyValues(IHostEnvironment env, IDataView[] views, string columnName, DataViewType keyValueType)
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckNonEmpty(columnName, nameof(columnName));

            var dvCount = views.Length;

            var keyNames = new Dictionary<ReadOnlyMemory<char>, int>();
            var columnIndices = new int[dvCount];
            var keyValueMappers = Utils.MarshalInvoke(MapKeys<int>, keyValueType.RawType, views.Select(view => view.Schema).ToArray(), columnName, true, columnIndices, keyNames);
            var keyType = new KeyType(typeof(uint), keyNames.Count);
            var keyNamesVBuffer = new VBuffer<ReadOnlyMemory<char>>(keyNames.Count, keyNames.Keys.ToArray());
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter =
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        keyNamesVBuffer.CopyTo(ref dst);

            for (int i = 0; i < dvCount; i++)
            {
                var keyMapperCur = keyValueMappers[i];
                ValueMapper<VBuffer<uint>, VBuffer<uint>> mapper =
                    (in VBuffer<uint> src, ref VBuffer<uint> dst) =>
                    {
                        var srcValues = src.GetValues();
                        var editor = VBufferEditor.Create(
                            ref dst,
                            src.Length,
                            srcValues.Length);
                        if (src.IsDense)
                        {
                            for (int j = 0; j < src.Length; j++)
                            {
                                if (srcValues[j] == 0 || srcValues[j] > keyMapperCur.Length)
                                    editor.Values[j] = 0;
                                else
                                    editor.Values[j] = (uint)keyMapperCur[srcValues[j] - 1] + 1;
                            }
                        }
                        else
                        {
                            var srcIndices = src.GetIndices();
                            for (int j = 0; j < srcValues.Length; j++)
                            {
                                if (srcValues[j] == 0 || srcValues[j] > keyMapperCur.Length)
                                    editor.Values[j] = 0;
                                else
                                    editor.Values[j] = (uint)keyMapperCur[srcValues[j] - 1] + 1;
                                editor.Indices[j] = srcIndices[j];
                            }
                        }
                        dst = editor.Commit();
                    };

                ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter = null;
                var type = views[i].Schema[columnIndices[i]].Type;
                if (views[i].Schema[columnIndices[i]].HasSlotNames(type.GetVectorSize()))
                {
                    var schema = views[i].Schema;
                    int index = columnIndices[i];
                    slotNamesGetter =
                        (ref VBuffer<ReadOnlyMemory<char>> dst) => schema[index].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref dst);
                }
                views[i] = LambdaColumnMapper.Create(env, "ReconcileKeyValues", views[i], columnName, columnName,
                    type, new VectorType(keyType, type as VectorType), mapper, keyValueGetter, slotNamesGetter);
            }
        }

        /// <summary>
        /// This method gets the per-instance metrics from multiple scored data views and either returns them as an
        /// array or combines them into a single data view, based on user specifications.
        /// </summary>
        /// <param name="env">A host environment.</param>
        /// <param name="eval">The evaluator to use for getting the per-instance metrics.</param>
        /// <param name="collate">If true, data views are combined into a single data view. Otherwise, data views
        /// are returned as an array.</param>
        /// <param name="outputFoldIndex">If true, a column containing the fold index is added to the returned data views.</param>
        /// <param name="perInstance">The array of scored data views to evaluate. These are passed as <see cref="RoleMappedData"/>
        /// so that the evaluator can know the role mappings it needs.</param>
        /// <param name="variableSizeVectorColumnNames">A list of column names that are not included in the combined data view
        /// since their types do not match.</param>
        /// <returns></returns>
        public static IDataView[] ConcatenatePerInstanceDataViews(IHostEnvironment env, IMamlEvaluator eval, bool collate, bool outputFoldIndex, RoleMappedData[] perInstance, out string[] variableSizeVectorColumnNames)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(eval, nameof(eval));
            env.CheckNonEmpty(perInstance, nameof(perInstance));

            Func<RoleMappedData, int, IDataView> getPerInstance =
                (rmd, i) =>
                {
                    var perInst = eval.GetPerInstanceDataViewToSave(rmd);

                    if (!outputFoldIndex)
                        return perInst;

                    // If the fold index is requested, add a column containing it. We use the first column in the data view
                    // as an input column to the LambdaColumnMapper, because it must have an input.
                    return AddFoldIndex(env, perInst, i, perInstance.Length);
                };

            var foldDataViews = perInstance.Select(getPerInstance).ToArray();
            if (collate)
            {
                var combined = AppendPerInstanceDataViews(env, perInstance[0].Schema.Label?.Name, foldDataViews, out variableSizeVectorColumnNames);
                return new[] { combined };
            }
            else
            {
                variableSizeVectorColumnNames = new string[0];
                return foldDataViews.ToArray();
            }
        }

        /// <summary>
        /// Create an output data view that is the vertical concatenation of the metric data views.
        /// </summary>
        public static IDataView ConcatenateOverallMetrics(IHostEnvironment env, IDataView[] metrics)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(metrics, nameof(metrics));

            if (metrics.Length == 1)
                return metrics[0];

            var overallList = new List<IDataView>();
            for (int i = 0; i < metrics.Length; i++)
            {
                // Add a fold-name column. We add it as a text column, since it is only used for saving the result summary file.
                var idv = AddFoldIndex(env, metrics[i], i);
                overallList.Add(idv);
            }
            return AppendRowsDataView.Create(env, overallList[0].Schema, overallList.ToArray());
        }

        private static IDataView AppendPerInstanceDataViews(IHostEnvironment env, string labelColName,
            IEnumerable<IDataView> foldDataViews, out string[] variableSizeVectorColumnNames)
        {
            Contracts.AssertValue(env);
            env.AssertValue(foldDataViews);

            // Make sure there are no variable size vector columns.
            // This is a dictionary from the column name to its vector size.
            var vectorSizes = new Dictionary<string, int>();
            var firstDvSlotNames = new Dictionary<string, VBuffer<ReadOnlyMemory<char>>>();
            DataViewType labelColKeyValuesType = null;
            var firstDvKeyWithNamesColumns = new List<string>();
            var firstDvKeyNoNamesColumns = new Dictionary<string, ulong>();
            var firstDvVectorKeyColumns = new List<string>();
            var variableSizeVectorColumnNamesList = new List<string>();
            var list = new List<IDataView>();
            int dvNumber = 0;
            foreach (var dv in foldDataViews)
            {
                var hidden = new List<int>();
                for (int i = 0; i < dv.Schema.Count; i++)
                {
                    if (dv.Schema[i].IsHidden)
                    {
                        hidden.Add(i);
                        continue;
                    }

                    var type = dv.Schema[i].Type;
                    var name = dv.Schema[i].Name;
                    ulong typeKeyCount = type.GetKeyCount();
                    if (type is VectorType vectorType)
                    {
                        if (dvNumber == 0)
                        {
                            if (dv.Schema[i].HasKeyValues())
                                firstDvVectorKeyColumns.Add(name);
                            // Store the slot names of the 1st idv and use them as baseline.
                            if (dv.Schema[i].HasSlotNames(vectorType.Size))
                            {
                                VBuffer<ReadOnlyMemory<char>> slotNames = default;
                                dv.Schema[i].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref slotNames);
                                firstDvSlotNames.Add(name, slotNames);
                            }
                        }

                        int cachedSize;
                        if (vectorSizes.TryGetValue(name, out cachedSize))
                        {
                            VBuffer<ReadOnlyMemory<char>> slotNames;
                            // In the event that no slot names were recorded here, then slotNames will be
                            // the default, length 0 vector.
                            firstDvSlotNames.TryGetValue(name, out slotNames);
                            if (!VerifyVectorColumnsMatch(cachedSize, i, dv, vectorType, in slotNames))
                                variableSizeVectorColumnNamesList.Add(name);
                        }
                        else
                            vectorSizes.Add(name, vectorType.Size);
                    }
                    else if (dvNumber == 0 && name == labelColName)
                    {
                        // The label column can be a key. Reconcile the key values, and wrap with a KeyToValue transform.
                        labelColKeyValuesType = dv.Schema[i].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type;
                    }
                    else if (dvNumber == 0 && dv.Schema[i].HasKeyValues())
                        firstDvKeyWithNamesColumns.Add(name);
                    else if (type.GetKeyCount() > 0 && name != labelColName && !dv.Schema[i].HasKeyValues())
                    {
                        // For any other key column (such as GroupId) we do not reconcile the key values, we only convert to U4.
                        if (!firstDvKeyNoNamesColumns.ContainsKey(name))
                            firstDvKeyNoNamesColumns[name] = typeKeyCount;
                        if (firstDvKeyNoNamesColumns[name] < typeKeyCount)
                            firstDvKeyNoNamesColumns[name] = typeKeyCount;
                    }
                }
                var idv = dv;
                if (hidden.Count > 0)
                {
                    var args = new ChooseColumnsByIndexTransform.Options();
                    args.Drop = true;
                    args.Indices = hidden.ToArray();
                    idv = new ChooseColumnsByIndexTransform(env, args, idv);
                }
                list.Add(idv);
                dvNumber++;
            }
            variableSizeVectorColumnNames = variableSizeVectorColumnNamesList.ToArray();

            var views = list.ToArray();
            foreach (var keyCol in firstDvKeyWithNamesColumns)
                ReconcileKeyValues(env, views, keyCol, TextDataViewType.Instance);
            if (labelColKeyValuesType != null)
                ReconcileKeyValues(env, views, labelColName, labelColKeyValuesType.GetItemType());
            foreach (var keyCol in firstDvKeyNoNamesColumns)
                ReconcileKeyValuesWithNoNames(env, views, keyCol.Key, keyCol.Value);
            foreach (var vectorKeyCol in firstDvVectorKeyColumns)
                ReconcileVectorKeyValues(env, views, vectorKeyCol, TextDataViewType.Instance);

            Func<IDataView, int, IDataView> keyToValue =
                (idv, i) =>
                {
                    foreach (var keyCol in AnnotationUtils.Prepend(firstDvVectorKeyColumns.Concat(firstDvKeyWithNamesColumns), labelColName))
                    {
                        if (keyCol == labelColName && labelColKeyValuesType == null)
                            continue;

                        idv = new KeyToValueMappingTransformer(env, keyCol).Transform(idv);
                        var hidden = FindHiddenColumns(idv.Schema, keyCol);
                        idv = new ChooseColumnsByIndexTransform(env, new ChooseColumnsByIndexTransform.Options() { Drop = true, Indices = hidden.ToArray() }, idv);
                    }
                    foreach (var keyCol in firstDvKeyNoNamesColumns)
                    {
                        var hidden = FindHiddenColumns(idv.Schema, keyCol.Key);
                        idv = new ChooseColumnsByIndexTransform(env, new ChooseColumnsByIndexTransform.Options() { Drop = true, Indices = hidden.ToArray() }, idv);
                    }
                    return idv;
                };

            Func<IDataView, IDataView> selectDropNonVarLenthCol =
                (idv) =>
                {
                    foreach (var variableSizeVectorColumnName in variableSizeVectorColumnNamesList)
                    {
                        int index;
                        idv.Schema.TryGetColumnIndex(variableSizeVectorColumnName, out index);
                        var vectorType = idv.Schema[index].Type as VectorType;
                        env.AssertValue(vectorType);

                        idv = Utils.MarshalInvoke(AddVarLengthColumn<int>, vectorType.ItemType.RawType, env, idv,
                                 variableSizeVectorColumnName, vectorType);

                        // Drop the old column that does not have variable length.
                        idv = ColumnSelectingTransformer.CreateDrop(env, idv, variableSizeVectorColumnName);
                    }
                    return idv;
                };

            return AppendRowsDataView.Create(env, null, views.Select(keyToValue).Select(selectDropNonVarLenthCol).ToArray());
        }

        private static IEnumerable<int> FindHiddenColumns(DataViewSchema schema, string colName)
        {
            for (int i = 0; i < schema.Count; i++)
            {
                if (schema[i].IsHidden && schema[i].Name == colName)
                    yield return i;
            }
        }

        private static bool VerifyVectorColumnsMatch(int cachedSize, int col, IDataView dv,
            VectorType type, in VBuffer<ReadOnlyMemory<char>> firstDvSlotNames)
        {
            if (cachedSize != type.Size)
                return false;

            // If we detect mismatch it a sign that slots reshuffling has happened.
            if (dv.Schema[col].HasSlotNames(type.Size))
            {
                // Verify that slots match with slots from 1st idv.
                VBuffer<ReadOnlyMemory<char>> currSlotNames = default;
                dv.Schema[col].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref currSlotNames);

                if (currSlotNames.Length != firstDvSlotNames.Length)
                    return false;
                else
                {
                    var result = true;
                    VBufferUtils.ForEachEitherDefined(in currSlotNames, in firstDvSlotNames,
                        (slot, val1, val2) => result = result && val1.Span.SequenceEqual(val2.Span));
                    return result;
                }
            }
            else
            {
                // If we don't have slot names, then the first dataview should not have had slot names either.
                return firstDvSlotNames.Length == 0;
            }
        }

        private static IDataView AddVarLengthColumn<TSrc>(IHostEnvironment env, IDataView idv, string variableSizeVectorColumnName, VectorType typeSrc)
        {
            return LambdaColumnMapper.Create(env, "ChangeToVarLength", idv, variableSizeVectorColumnName,
                       variableSizeVectorColumnName + "_VarLength", typeSrc, new VectorType((PrimitiveDataViewType)typeSrc.ItemType),
                       (in VBuffer<TSrc> src, ref VBuffer<TSrc> dst) => src.CopyTo(ref dst));
        }

        private static List<string> GetMetricNames(IChannel ch, DataViewSchema schema, DataViewRow row, Func<int, bool> ignoreCol,
            ValueGetter<double>[] getters, ValueGetter<VBuffer<double>>[] vBufferGetters)
        {
            ch.AssertValue(schema);
            ch.AssertValue(row);
            ch.Assert(Utils.Size(getters) == schema.Count);
            ch.Assert(Utils.Size(vBufferGetters) == schema.Count);

            // Get the names of the metrics. For R8 valued columns the metric name is the column name. For R8 vector valued columns
            // the names of the metrics are the column name, followed by the slot name if it exists, or "Label_i" if it doesn't.
            VBuffer<ReadOnlyMemory<char>> names = default;
            int metricCount = 0;
            var metricNames = new List<string>();
            for (int i = 0; i < schema.Count; i++)
            {
                if (schema[i].IsHidden || ignoreCol(i))
                    continue;

                var type = schema[i].Type;
                var metricName = row.Schema[i].Name;
                if (type is NumberDataViewType)
                {
                    getters[i] = RowCursorUtils.GetGetterAs<double>(NumberDataViewType.Double, row, i);
                    metricNames.Add(metricName);
                    metricCount++;
                }
                else if (type is VectorType vectorType && vectorType.ItemType == NumberDataViewType.Double)
                {
                    if (vectorType.Size == 0)
                    {
                        ch.Warning("Vector metric '{0}' has different lengths in different folds and will not be averaged for overall results.", metricName);
                        continue;
                    }

                    vBufferGetters[i] = row.GetGetter<VBuffer<double>>(schema[i]);
                    metricCount += vectorType.Size;
                    var slotNamesType = schema[i].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType;
                    if (slotNamesType != null && slotNamesType.Size == vectorType.Size && slotNamesType.ItemType is TextDataViewType)
                        schema[i].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref names);
                    else
                    {
                        var editor = VBufferEditor.Create(ref names, vectorType.Size);
                        for (int j = 0; j < vectorType.Size; j++)
                            editor.Values[j] = string.Format("Label_{0}", j).AsMemory();
                        names = editor.Commit();
                    }
                    foreach (var name in names.Items(all: true))
                        metricNames.Add(string.Format("{0}{1}", metricName, name.Value));
                }
            }
            ch.Assert(metricNames.Count == metricCount);
            return metricNames;
        }

        internal static IDataView GetOverallMetricsData(IHostEnvironment env, IDataView data, int numFolds, out AggregatedMetric[] agg,
            out AggregatedMetric[] weightedAgg)
        {
            agg = ComputeMetricsSum(env, data, numFolds, out int isWeightedCol, out int stratCol, out int stratVal, out int foldCol, out weightedAgg);

            var nonAveragedCols = new List<string>();
            var avgMetrics = GetAverageToDataView(env, data.Schema, agg, weightedAgg, numFolds, stratCol, stratVal,
                isWeightedCol, foldCol, numFolds > 1, nonAveragedCols);

            var idvList = new List<IDataView>() { avgMetrics };

            var hasStrat = stratCol >= 0;
            if (numFolds > 1 || hasStrat)
            {
                if (Utils.Size(nonAveragedCols) > 0)
                {
                    data = ColumnSelectingTransformer.CreateDrop(env, data, nonAveragedCols.ToArray());
                }
                idvList.Add(data);
            }

            var overall = AppendRowsDataView.Create(env, avgMetrics.Schema, idvList.ToArray());

            // If there are stratified results, apply a KeyToValue transform to get the stratification column
            // names from the key column.
            if (hasStrat)
                overall = new KeyToValueMappingTransformer(env, MetricKinds.ColumnNames.StratCol).Transform(overall);
            return overall;
        }

        internal static AggregatedMetric[] ComputeMetricsSum(IHostEnvironment env, IDataView data, int numFolds, out int isWeightedCol,
            out int stratCol, out int stratVal, out int foldCol, out AggregatedMetric[] weightedAgg)
        {
            var isWeightedColumn = data.Schema.GetColumnOrNull(MetricKinds.ColumnNames.IsWeighted);
            var hasWeighted = isWeightedColumn.HasValue;
            var hasStrats = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out int scol);
            var hasStratVals = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out int svalcol);
            env.Assert(hasStrats == hasStratVals);
            var hasFoldCol = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.FoldIndex, out int fcol);

            isWeightedCol = hasWeighted ? isWeightedColumn.Value.Index : -1;
            stratCol = hasStrats ? scol : -1;
            stratVal = hasStratVals ? svalcol : -1;
            foldCol = hasFoldCol ? fcol : -1;

            // We currently have only double valued or vector of double valued metrics.
            int colCount = data.Schema.Count;
            var getters = new ValueGetter<double>[colCount];
            var vBufferGetters = new ValueGetter<VBuffer<double>>[colCount];
            int numResults = 0;
            int numWeightedResults = 0;
            AggregatedMetric[] agg;
            using (var cursor = data.GetRowCursorForAllColumns())
            {
                bool isWeighted = false;
                ValueGetter<bool> isWeightedGetter;
                if (hasWeighted)
                    isWeightedGetter = cursor.GetGetter<bool>(isWeightedColumn.Value);
                else
                    isWeightedGetter = (ref bool dst) => dst = false;

                ValueGetter<uint> stratColGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema[stratCol].Type;
                    stratColGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratColGetter = (ref uint dst) => dst = 0;

                // Get the names of the metrics. For R8 valued columns the metric name is the column name. For R8 vector valued columns
                // the names of the metrics are the column name, followed by the slot name if it exists, or "Label_i" if it doesn't.
                List<string> metricNames;
                using (var ch = env.Register("GetMetricsAsString").Start("Get Metric Names"))
                {
                    metricNames = GetMetricNames(ch, data.Schema, cursor,
                        i => hasWeighted && i == isWeightedColumn.Value.Index || hasStrats && (i == scol || i == svalcol) ||
                            hasFoldCol && i == fcol, getters, vBufferGetters);
                }
                agg = new AggregatedMetric[metricNames.Count];

                Double metricVal = 0;
                VBuffer<Double> metricVals = default(VBuffer<Double>);
                if (hasWeighted)
                    weightedAgg = new AggregatedMetric[metricNames.Count];
                else
                    weightedAgg = null;
                uint strat = 0;
                while (cursor.MoveNext())
                {
                    stratColGetter(ref strat);
                    // REVIEW: how to print stratified results?
                    if (strat > 0)
                        continue;

                    isWeightedGetter(ref isWeighted);
                    if (isWeighted)
                    {
                        // If !average, we should have only one relevant row.
                        if (numWeightedResults > numFolds)
                            throw Contracts.Except("Multiple weighted rows found in metrics data view.");

                        numWeightedResults++;
                        UpdateSums(isWeightedCol, stratCol, stratVal, weightedAgg, numFolds > 1, metricNames, hasWeighted,
                            hasStrats, colCount, getters, vBufferGetters, ref metricVal, ref metricVals);
                    }
                    else
                    {
                        // If !average, we should have only one relevant row.
                        if (numResults > numFolds)
                            throw Contracts.Except("Multiple unweighted rows found in metrics data view.");

                        numResults++;
                        UpdateSums(isWeightedCol, stratCol, stratVal, agg, numFolds > 1, metricNames, hasWeighted, hasStrats,
                            colCount, getters, vBufferGetters, ref metricVal, ref metricVals);
                    }

                    if (numResults == numFolds && (!hasWeighted || numWeightedResults == numFolds))
                        break;
                }
            }
            return agg;
        }

        private static void UpdateSums(int isWeightedCol, int stratCol, int stratVal, AggregatedMetric[] aggregated, bool hasStdev, List<string> metricNames, bool hasWeighted, bool hasStrats, int colCount, ValueGetter<double>[] getters, ValueGetter<VBuffer<double>>[] vBufferGetters, ref double metricVal, ref VBuffer<double> metricVals)
        {
            int iMetric = 0;
            for (int i = 0; i < colCount; i++)
            {
                if (hasWeighted && i == isWeightedCol || hasStrats && (i == stratCol || i == stratVal))
                    continue;

                if (getters[i] == null && vBufferGetters[i] == null)
                {
                    // REVIEW: What to do with metrics that are not doubles?
                    continue;
                }
                if (getters[i] != null)
                {
                    getters[i](ref metricVal);
                    aggregated[iMetric].Sum += metricVal;
                    if (hasStdev)
                        aggregated[iMetric].SumSq += metricVal * metricVal;
                    aggregated[iMetric].Name = metricNames[iMetric];
                    iMetric++;
                }
                else
                {
                    Contracts.AssertValue(vBufferGetters[i]);
                    vBufferGetters[i](ref metricVals);
                    foreach (var metric in metricVals.Items(all: true))
                    {
                        aggregated[iMetric].Sum += metric.Value;
                        if (hasStdev)
                            aggregated[iMetric].SumSq += metric.Value * metric.Value;
                        aggregated[iMetric].Name = metricNames[iMetric];
                        iMetric++;
                    }
                }
            }
            Contracts.Assert(iMetric == metricNames.Count);
        }

        internal static IDataView GetAverageToDataView(IHostEnvironment env, DataViewSchema schema, AggregatedMetric[] agg, AggregatedMetric[] weightedAgg,
            int numFolds, int stratCol, int stratVal, int isWeightedCol, int foldCol, bool hasStdev, List<string> nonAveragedCols = null)
        {
            Contracts.AssertValue(env);

            int colCount = schema.Count;

            var dvBldr = new ArrayDataViewBuilder(env);
            var weightedDvBldr = isWeightedCol >= 0 ? new ArrayDataViewBuilder(env) : null;

            int iMetric = 0;
            for (int i = 0; i < colCount; i++)
            {
                if (schema[i].IsHidden)
                    continue;

                var type = schema[i].Type;
                var name = schema[i].Name;
                if (i == stratCol)
                {
                    int typeKeyCount = type.GetKeyCountAsInt32(env);

                    var keyValuesType = schema[i].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorType;
                    if (keyValuesType == null || !(keyValuesType.ItemType is TextDataViewType) ||
                        keyValuesType.Size != typeKeyCount)
                    {
                        throw env.Except("Column '{0}' must have key values metadata",
                            MetricKinds.ColumnNames.StratCol);
                    }

                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getKeyValues =
                        (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            schema[stratCol].GetKeyValues(ref dst);
                            Contracts.Assert(dst.IsDense);
                        };

                    var keys = foldCol >= 0 ? new uint[] { 0, 0 } : new uint[] { 0 };
                    dvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, getKeyValues, type.GetKeyCount(), keys);
                    weightedDvBldr?.AddColumn(MetricKinds.ColumnNames.StratCol, getKeyValues, type.GetKeyCount(), keys);
                }
                else if (i == stratVal)
                {
                    //REVIEW: Not sure if empty string makes sense here.
                    var stratVals = foldCol >= 0 ? new[] { "".AsMemory(), "".AsMemory() } : new[] { "".AsMemory() };
                    dvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextDataViewType.Instance, stratVals);
                    weightedDvBldr?.AddColumn(MetricKinds.ColumnNames.StratVal, TextDataViewType.Instance, stratVals);
                }
                else if (i == isWeightedCol)
                {
                    env.AssertValue(weightedDvBldr);
                    dvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BooleanDataViewType.Instance, foldCol >= 0 ? new[] { false, false } : new[] { false });
                    weightedDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BooleanDataViewType.Instance, foldCol >= 0 ? new[] { true, true } : new[] { true });
                }
                else if (i == foldCol)
                {
                    var foldVals = new[] { "Average".AsMemory(), "Standard Deviation".AsMemory() };
                    dvBldr.AddColumn(MetricKinds.ColumnNames.FoldIndex, TextDataViewType.Instance, foldVals);
                    weightedDvBldr?.AddColumn(MetricKinds.ColumnNames.FoldIndex, TextDataViewType.Instance, foldVals);
                }
                else if (type is NumberDataViewType)
                {
                    dvBldr.AddScalarColumn(schema, agg, hasStdev, numFolds, iMetric);
                    weightedDvBldr?.AddScalarColumn(schema, weightedAgg, hasStdev, numFolds, iMetric);
                    iMetric++;
                }
                else if (type is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType == NumberDataViewType.Double)
                {
                    dvBldr.AddVectorColumn(env, schema, agg, hasStdev, numFolds, iMetric, i, vectorType, name);
                    weightedDvBldr?.AddVectorColumn(env, schema, weightedAgg, hasStdev, numFolds, iMetric, i, vectorType, name);
                    iMetric += vectorType.Size;
                }
                else
                    nonAveragedCols?.Add(name);
            }
            var idv = dvBldr.GetDataView();
            if (weightedDvBldr != null)
                idv = AppendRowsDataView.Create(env, idv.Schema, idv, weightedDvBldr.GetDataView());
            return idv;
        }

        private static void AddVectorColumn(this ArrayDataViewBuilder dvBldr, IHostEnvironment env, DataViewSchema schema,
            AggregatedMetric[] agg, bool hasStdev, int numFolds, int iMetric, int i, VectorType type, string columnName)
        {
            var vectorMetrics = new double[type.Size];
            env.Assert(vectorMetrics.Length > 0);
            for (int j = 0; j < vectorMetrics.Length; j++)
                vectorMetrics[j] = agg[iMetric + j].Sum / numFolds;
            double[] vectorStdevMetrics = null;
            if (hasStdev)
            {
                vectorStdevMetrics = new double[type.Size];
                for (int j = 0; j < vectorStdevMetrics.Length; j++)
                    vectorStdevMetrics[j] = Math.Sqrt(agg[iMetric + j].SumSq / numFolds - vectorMetrics[j] * vectorMetrics[j]);
            }
            var names = new ReadOnlyMemory<char>[type.Size];
            for (int j = 0; j < names.Length; j++)
                names[j] = agg[iMetric + j].Name.AsMemory();
            var slotNames = new VBuffer<ReadOnlyMemory<char>>(type.Size, names);
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames = (ref VBuffer<ReadOnlyMemory<char>> dst) => dst = slotNames;
            if (vectorStdevMetrics != null)
            {
                env.AssertValue(vectorStdevMetrics);
                dvBldr.AddColumn(columnName, getSlotNames, NumberDataViewType.Double, new[] { vectorMetrics, vectorStdevMetrics });
            }
            else
                dvBldr.AddColumn(columnName, getSlotNames, NumberDataViewType.Double, new[] { vectorMetrics });
        }

        private static void AddScalarColumn(this ArrayDataViewBuilder dvBldr, DataViewSchema schema, AggregatedMetric[] agg, bool hasStdev, int numFolds, int iMetric)
        {
            Contracts.AssertValue(dvBldr);

            var avg = agg[iMetric].Sum / numFolds;
            if (hasStdev)
                dvBldr.AddColumn(agg[iMetric].Name, NumberDataViewType.Double, avg, Math.Sqrt(agg[iMetric].SumSq / numFolds - avg * avg));
            else
                dvBldr.AddColumn(agg[iMetric].Name, NumberDataViewType.Double, avg);
        }

        /// <summary>
        /// Takes a data view containing one or more rows of metrics, and returns a data view containing additional
        /// rows with the average and the standard deviation of the metrics in the input data view.
        /// </summary>
        public static IDataView CombineFoldMetricsDataViews(IHostEnvironment env, IDataView data, int numFolds)
        {
            return GetOverallMetricsData(env, data, numFolds, out var _, out var _);
        }
    }

    internal static class MetricWriter
    {
        /// <summary>
        /// Get the confusion tables as strings to be printed to the Console.
        /// </summary>
        /// <param name="host">The host is used for getting the random number generator for sampling classes</param>
        /// <param name="confusionDataView">The data view containing the confusion matrix. It should contain a text column
        /// with the label names named "LabelNames", and an R8 vector column named "Count" containing the counts: in the row
        /// corresponding to label i, slot j should contain the number of class i examples that were predicted as j by the predictor.</param>
        /// <param name="weightedConfusionTable">If there is an R8 vector column named "Weight" containing the weighted counts, this parameter
        /// is assigned the string representation of the weighted confusion table. Otherwise it is assigned null.</param>
        /// <param name="binary">Indicates whether the confusion table is for binary classification.</param>
        /// <param name="sample">Indicates how many classes to sample from the confusion table (-1 indicates no sampling)</param>
        public static string GetConfusionTable(IHost host, IDataView confusionDataView, out string weightedConfusionTable, bool binary = true, int sample = -1)
        {
            host.CheckValue(confusionDataView, nameof(confusionDataView));
            host.CheckParam(sample == -1 || sample >= 2, nameof(sample), "Should be -1 to indicate no sampling, or at least 2");

            // Get the class names.
            int countCol;
            host.Check(confusionDataView.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.Count, out countCol), "Did not find the count column");
            var type = confusionDataView.Schema[countCol].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType;
            host.Check(type != null && type.IsKnownSize && type.ItemType is TextDataViewType, "The Count column does not have a text vector metadata of kind SlotNames.");

            var labelNames = default(VBuffer<ReadOnlyMemory<char>>);
            confusionDataView.Schema[countCol].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref labelNames);
            host.Check(labelNames.IsDense, "Slot names vector must be dense");

            int numConfusionTableLabels = sample < 0 ? labelNames.Length : Math.Min(labelNames.Length, sample);

            // Sample the classes. We choose a random permutation, keep the first 'sample' indices and drop the rest.
            // The labelIndexToConfIndexMap array indicates for each class its index in the confusion table, or -1 if it is dropped
            var labelIndexToConfIndexMap = new int[labelNames.Length];
            if (numConfusionTableLabels < labelNames.Length)
            {
                var tempPerm = Utils.GetRandomPermutation(host.Rand, labelNames.Length);
                var sampledIndices = tempPerm.Skip(labelNames.Length - numConfusionTableLabels).OrderBy(i => i);

                for (int i = 0; i < labelIndexToConfIndexMap.Length; i++)
                    labelIndexToConfIndexMap[i] = -1;
                int countNotDropped = 0;
                foreach (var i in sampledIndices)
                    labelIndexToConfIndexMap[i] = countNotDropped++;
            }
            else
            {
                for (int i = 0; i < labelNames.Length; i++)
                    labelIndexToConfIndexMap[i] = i;
            }

            double[] precisionSums;
            double[] recallSums;
            var confusionTable = GetConfusionTableAsArray(confusionDataView, countCol, labelNames.Length,
                labelIndexToConfIndexMap, numConfusionTableLabels, out precisionSums, out recallSums);

            var predictedLabelNames = GetPredictedLabelNames(in labelNames, labelIndexToConfIndexMap);
            var confusionTableString = GetConfusionTableAsString(confusionTable, recallSums, precisionSums,
               predictedLabelNames,
               sampled: numConfusionTableLabels < labelNames.Length, binary: binary);

            int weightIndex;
            if (confusionDataView.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.Weight, out weightIndex))
            {
                confusionTable = GetConfusionTableAsArray(confusionDataView, weightIndex, labelNames.Length,
                   labelIndexToConfIndexMap, numConfusionTableLabels, out precisionSums, out recallSums);
                weightedConfusionTable = GetConfusionTableAsString(confusionTable, recallSums, precisionSums,
                    predictedLabelNames,
                    sampled: numConfusionTableLabels < labelNames.Length, prefix: "Weighted ", binary: binary);
            }
            else
                weightedConfusionTable = null;

            return confusionTableString;
        }

        private static List<ReadOnlyMemory<char>> GetPredictedLabelNames(in VBuffer<ReadOnlyMemory<char>> labelNames, int[] labelIndexToConfIndexMap)
        {
            List<ReadOnlyMemory<char>> result = new List<ReadOnlyMemory<char>>();
            var values = labelNames.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (labelIndexToConfIndexMap[i] >= 0)
                {
                    result.Add(values[i]);
                }
            }
            return result;
        }

        // This methods is given a data view and a column index of the counts, and computes three arrays: the confusion table,
        // the per class recall and the per class precision.
        private static double[][] GetConfusionTableAsArray(IDataView confusionDataView, int countIndex, int numClasses,
            int[] labelIndexToConfIndexMap, int numConfusionTableLabels, out double[] precisionSums, out double[] recallSums)
        {
            var confusionTable = new Double[numConfusionTableLabels][];
            for (int i = 0; i < numConfusionTableLabels; i++)
                confusionTable[i] = new Double[numConfusionTableLabels];

            precisionSums = new Double[numConfusionTableLabels];
            recallSums = new Double[numConfusionTableLabels];

            int stratCol;
            var hasStrat = confusionDataView.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            using (var cursor = confusionDataView.GetRowCursor(confusionDataView.Schema.Where(col => col.Index == countIndex || hasStrat && col.Index == stratCol)))
            {
                var type = cursor.Schema[countIndex].Type as VectorType;
                Contracts.Check(type != null && type.IsKnownSize && type.ItemType == NumberDataViewType.Double);
                var countGetter = cursor.GetGetter<VBuffer<double>>(cursor.Schema[countIndex]);
                ValueGetter<uint> stratGetter = null;
                if (hasStrat)
                {
                    var stratType = cursor.Schema[stratCol].Type;
                    stratGetter = RowCursorUtils.GetGetterAs<uint>(stratType, cursor, stratCol);
                }

                var count = default(VBuffer<double>);
                int numRows = -1;
                while (cursor.MoveNext())
                {
                    uint strat = 0;
                    if (stratGetter != null)
                        stratGetter(ref strat);
                    if (strat > 0)
                        continue;

                    numRows++;
                    if (labelIndexToConfIndexMap[numRows] < 0)
                        continue;

                    countGetter(ref count);
                    if (count.Length != numClasses)
                        throw Contracts.Except("Expected {0} values in 'Count' column, but got {1}.", numClasses, count.Length);
                    int row = labelIndexToConfIndexMap[numRows];
                    foreach (var val in count.Items())
                    {
                        var index = val.Key;
                        if (labelIndexToConfIndexMap[index] < 0)
                            continue;

                        confusionTable[row][labelIndexToConfIndexMap[index]] = val.Value;
                        precisionSums[labelIndexToConfIndexMap[index]] += val.Value;
                        recallSums[row] += val.Value;
                    }

                    if (numRows == numClasses - 1)
                        break;
                }
            }
            return confusionTable;
        }

        /// <summary>
        /// This method returns the per-fold metrics as a string. If weighted metrics are present they are returned in a separate string.
        /// </summary>
        /// <param name="env">An IHostEnvironment.</param>
        /// <param name="fold">The data view containing the per-fold metrics. Each row in the data view represents a set of metrics
        /// calculated either on the whole dataset or on a subset of it defined by a stratification column. If the data view contains
        /// stratified metrics, it must contain two text columns named "StratCol" and "StratVal", containing the stratification column
        /// name, and a text description of the value. In this case, the value of column StratVal in the row corresponding to the entire
        /// dataset should contain the text "overall", and the value of column StratCol should be DvText.NA. If weighted metrics are present
        /// then the data view should also contain a bool column named "IsWeighted".</param>
        /// <param name="weightedMetrics">If the IsWeighted column exists, this is assigned the string representation of the weighted
        /// metrics. Otherwise it is assigned null.</param>
        public static string GetPerFoldResults(IHostEnvironment env, IDataView fold, out string weightedMetrics)
        {
            return GetFoldMetricsAsString(env, fold, out weightedMetrics);
        }

        private static string GetOverallMetricsAsString(double[] sumMetrics, double[] sumSqMetrics, int numFolds, bool weighted, bool average, List<string> metricNames)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < metricNames.Count; i++)
            {
                var avg = sumMetrics[i] / numFolds;
                sb.Append(string.Format("{0}{1}: ", weighted ? "Weighted " : "", metricNames[i]).PadRight(20));
                sb.Append(string.Format(CultureInfo.InvariantCulture, "{0,7:N6}", avg));
                if (average)
                {
                    Contracts.Assert(sumSqMetrics != null || numFolds == 1);
                    sb.AppendLine(string.Format(" ({0:N4})", numFolds == 1 ? 0 :
                        Math.Sqrt(sumSqMetrics[i] / numFolds - avg * avg)));
                }
                else
                    sb.AppendLine();
            }
            return sb.ToString();
        }

        // This method returns a string representation of a set of metrics. If there are stratification columns, it looks for columns named
        // StratCol and StratVal, and outputs the metrics in the rows with NA in the StratCol column. If weighted is true, it looks
        // for a bool column named "IsWeighted" and outputs the metrics in the rows with a value of true in that column.
        // If nonAveragedCols is non-null, it computes the average and standard deviation over all the relevant rows and populates
        // nonAveragedCols with columns that are either hidden, or are not of a type that we can display (i.e., either a numeric column,
        // or a known length vector of doubles).
        // If average is false, no averaging is done, and instead we check that there is exactly one relevant row. Otherwise, we
        // add the vector columns of variable length of the list of non-averagable columns if nonAveragedCols is not null.
        private static string GetFoldMetricsAsString(IHostEnvironment env, IDataView data, out string weightedMetricsString)
        {
            var metrics = EvaluateUtils.ComputeMetricsSum(env, data, 1, out int isWeightedCol, out int stratCol,
                out int stratVal, out int foldCol, out var weightedMetrics);

            var sb = new StringBuilder();
            var weightedSb = isWeightedCol >= 0 ? new StringBuilder() : null;
            for (int i = 0; i < metrics.Length; i++)
            {
                sb.Append($"{metrics[i].Name}: ".PadRight(20));
                sb.Append(string.Format(CultureInfo.InvariantCulture, "{0,7:N6}", metrics[i].Sum));
                weightedSb?.Append($"Weighted {weightedMetrics[i].Name}: ".PadRight(20));
                weightedSb?.Append(string.Format(CultureInfo.InvariantCulture, "{0,7:N6}", weightedMetrics[i].Sum));
                sb.AppendLine();
                weightedSb?.AppendLine();
            }

            weightedMetricsString = weightedSb?.ToString();
            return sb.ToString();
        }

        // Get a string representation of a confusion table.
        private static string GetConfusionTableAsString(double[][] confusionTable, double[] rowSums, double[] columnSums,
            List<ReadOnlyMemory<char>> predictedLabelNames, string prefix = "", bool sampled = false, bool binary = true)
        {
            int numLabels = Utils.Size(confusionTable);

            int colWidth = numLabels == 2 ? 8 : 5;
            int maxNameLen = predictedLabelNames.Max(name => name.Length);
            // If the names are too long to fit in the column header, we back off to using class indices
            // in the header. This will also require putting the indices in the row, but it's better than
            // the alternative of having ambiguous abbreviated column headers, or having a table potentially
            // too wide to fit in a console.
            bool useNumbersInHeader = maxNameLen > colWidth;

            int rowLabelLen = maxNameLen;
            int rowDigitLen = 0;
            if (useNumbersInHeader)
            {
                // The row label will also include the index, so a user can easily match against the header.
                // In such a case, a label like "Foo" would be presented as something like "5. Foo".
                rowDigitLen = Math.Max(predictedLabelNames.Count - 1, 0).ToString().Length;
                Contracts.Assert(rowDigitLen >= 1);
                rowLabelLen += rowDigitLen + 2;
            }
            Contracts.Assert((rowDigitLen == 0) == !useNumbersInHeader);

            // The "PREDICTED" in the table, at length 9, dictates the amount of additional padding that will
            // be necessary on account of label names.
            int paddingLen = Math.Max(9, rowLabelLen);
            string pad = new string(' ', paddingLen - 9);
            string rowLabelFormat = null;
            if (useNumbersInHeader)
            {
                int namePadLen = paddingLen - (rowDigitLen + 2);
                rowLabelFormat = string.Format("{{0,{0}}}. {{1,{1}}} ||", rowDigitLen, namePadLen);
            }
            else
                rowLabelFormat = string.Format("{{1,{0}}} ||", paddingLen);

            var sb = new StringBuilder();
            if (numLabels == 2 && binary)
            {
                var positiveCaps = predictedLabelNames[0].ToString().ToUpper();

                var numTruePos = confusionTable[0][0];
                var numFalseNeg = confusionTable[0][1];
                var numTrueNeg = confusionTable[1][1];
                var numFalsePos = confusionTable[1][0];
                sb.AppendFormat("{0}TEST {1} RATIO:\t{2:N4} ({3:F1}/({3:F1}+{4:F1}))", prefix, positiveCaps,
                    1.0 * (numTruePos + numFalseNeg) / (numTruePos + numTrueNeg + numFalseNeg + numFalsePos),
                    numTruePos + numFalseNeg, numFalsePos + numTrueNeg);
            }

            sb.AppendLine();
            sb.AppendFormat("{0}Confusion table", prefix);
            if (sampled)
                sb.AppendLine(" (sampled)");
            else
                sb.AppendLine();

            sb.AppendFormat("          {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append(numLabels > 2 ? "========" : "===========");
            sb.AppendLine();
            sb.AppendFormat("PREDICTED {0}||", pad);
            string format = string.Format(" {{{0},{1}}} |", useNumbersInHeader ? 0 : 1, colWidth);
            for (int i = 0; i < numLabels; i++)
                sb.AppendFormat(format, i, predictedLabelNames[i]);
            sb.AppendLine(" Recall");
            sb.AppendFormat("TRUTH     {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append(numLabels > 2 ? "========" : "===========");

            sb.AppendLine();

            string format2 = string.Format(" {{0,{0}:{1}}} |", colWidth,
                string.IsNullOrWhiteSpace(prefix) ? "N0" : "F1");
            for (int i = 0; i < numLabels; i++)
            {
                sb.AppendFormat(rowLabelFormat, i, predictedLabelNames[i]);
                for (int j = 0; j < numLabels; j++)
                    sb.AppendFormat(format2, confusionTable[i][j]);
                Double recall = rowSums[i] > 0 ? confusionTable[i][i] / rowSums[i] : 0;
                sb.AppendFormat(" {0,5:F4}", recall);
                sb.AppendLine();
            }
            sb.AppendFormat("          {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append(numLabels > 2 ? "========" : "===========");
            sb.AppendLine();
            sb.AppendFormat("Precision {0}||", pad);
            format = string.Format("{{0,{0}:N4}} |", colWidth + 1);
            for (int i = 0; i < numLabels; i++)
            {
                Double precision = columnSums[i] > 0 ? confusionTable[i][i] / columnSums[i] : 0;
                sb.AppendFormat(format, precision);
            }
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Print the overall results to the Console. The overall data view should contain rows from all the folds being averaged.
        /// If filename is not null then also save the results to the specified file. The first row in the file is the averaged
        /// results, followed by the results of each fold.
        /// </summary>
        public static void PrintOverallMetrics(IHostEnvironment env, IChannel ch, string filename, IDataView overall, int numFolds)
        {
            var overallWithAvg = EvaluateUtils.GetOverallMetricsData(env, overall, numFolds, out var agg, out var weightedAgg);

            var sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine("OVERALL RESULTS");
            sb.AppendLine("---------------------------------------");

            var nonAveragedCols = new List<string>();
            if (weightedAgg != null)
                sb.Append(GetOverallMetricsAsString(weightedAgg.Select(x => x.Sum).ToArray(), weightedAgg.Select(x => x.SumSq).ToArray(), numFolds, true, true, weightedAgg.Select(x => x.Name).ToList()));
            sb.Append(GetOverallMetricsAsString(agg.Select(x => x.Sum).ToArray(), agg.Select(x => x.SumSq).ToArray(), numFolds, false, true, agg.Select(x => x.Name).ToList()));
            sb.AppendLine("\n---------------------------------------");
            ch.Info(sb.ToString());

            if (!string.IsNullOrEmpty(filename))
            {
                using (var file = env.CreateOutputFile(filename))
                {
                    var saverArgs = new TextSaver.Arguments() { Dense = true, Silent = true };
                    DataSaverUtils.SaveDataView(ch, new TextSaver(env, saverArgs), overallWithAvg, file);
                }
            }
        }

        private static string PadLeft(string str, int totalLength)
        {
            if (str.Length > totalLength)
                return str.Substring(0, totalLength - 1).PadRight(totalLength, '.');
            return str.PadLeft(totalLength);
        }

        /// <summary>
        /// Searches for a warning dataview in the given dictionary, and if present, prints the warnings to the given channel. The warning dataview
        /// should contain a text column named "WarningText".
        /// </summary>
        public static void PrintWarnings(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            IDataView warnings;
            if (metrics.TryGetValue(MetricKinds.Warnings, out warnings))
            {
                var warningTextColumn = warnings.Schema.GetColumnOrNull(MetricKinds.ColumnNames.WarningText);
                if (warningTextColumn !=null && warningTextColumn.HasValue && warningTextColumn.Value.Type is TextDataViewType)
                {
                    using (var cursor = warnings.GetRowCursor(warnings.Schema[MetricKinds.ColumnNames.WarningText]))
                    {
                        var warning = default(ReadOnlyMemory<char>);
                        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(warningTextColumn.Value);
                        while (cursor.MoveNext())
                        {
                            getter(ref warning);
                            ch.Warning(warning.ToString());
                        }
                    }
                }
            }
        }

        /// <summary>
        ///  Save the given data view using text saver.
        /// </summary>
        public static void SavePerInstance(IHostEnvironment env, IChannel ch, string filename, IDataView data,
            bool dense = true, bool saveSchema = false)
        {
            using (var file = env.CreateOutputFile(filename))
            {
                DataSaverUtils.SaveDataView(ch,
                    new TextSaver(env, new TextSaver.Arguments() { OutputSchema = saveSchema, Dense = dense, Silent = true }),
                    data, file);
            }
        }

        /// <summary>
        /// Filter out the stratified results from overall and drop the stratification columns.
        /// </summary>
        public static IDataView GetNonStratifiedMetrics(IHostEnvironment env, IDataView data)
        {
            int stratCol;
            if (!data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol))
                return data;
            var type = data.Schema[stratCol].Type;
            env.Check(type.GetKeyCount() > 0, "Expected a known count key type stratification column");
            var filterArgs = new NAFilter.Arguments();
            filterArgs.Columns = new[] { MetricKinds.ColumnNames.StratCol };
            filterArgs.Complement = true;
            data = new NAFilter(env, filterArgs, data);

            int stratVal;
            var found = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            env.Check(found, "If stratification column exist, data view must also contain a StratVal column");

            data = ColumnSelectingTransformer.CreateDrop(env, data, data.Schema[stratCol].Name, data.Schema[stratVal].Name);
            return data;
        }
    }

    /// <summary>
    /// This is a list of string constants denoting 'standard' metric kinds.
    /// </summary>
    [BestFriend]
    internal static class MetricKinds
    {
        /// <summary>
        /// This data view contains the confusion matrix for N-class classification. It has N rows, and each row has
        /// the following columns:
        /// * Count (vector indicating how many examples of this class were predicted as each one of the classes). This column
        /// should have metadata containing the class names.
        /// * (Optional) Weight (vector with the total weight of the examples of this class that were predicted as each one of the classes).
        /// </summary>
        public const string ConfusionMatrix = "ConfusionMatrix";

        /// <summary>
        /// This is a data view with 'global' dataset-wise metrics in its columns. It has one row containing the overall metrics,
        /// and optionally more rows for weighted metrics, and stratified metrics.
        /// </summary>
        public const string OverallMetrics = "OverallMetrics";

        /// <summary>
        /// This data view contains a single text column, with warnings about bad input values encountered by the evaluator during
        /// the aggregation of metrics. Each warning is in a separate row.
        /// </summary>
        public const string Warnings = "Warnings";

        /// <summary>
        /// Names for the columns in the data views output by evaluators.
        /// </summary>
        public sealed class ColumnNames
        {
            public const string WarningText = "WarningText";
            public const string IsWeighted = "IsWeighted";
            public const string Count = "Count";
            public const string Weight = "Weight";
            public const string StratCol = "StratCol";
            public const string StratVal = "StratVal";
            public const string FoldIndex = "Fold Index";
        }
    }
}
