// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public static class EvaluateUtils
    {
        private static class DefaultEvaluatorTable
        {
            private static volatile Dictionary<string, string> _knownEvaluatorLoadNames;

            public static Dictionary<string, string> Instance
            {
                get
                {
                    if (_knownEvaluatorLoadNames == null)
                    {
                        var tmp = new Dictionary<string, string>();
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.BinaryClassification, BinaryClassifierEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.MultiClassClassification, MultiClassClassifierEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.Regression, RegressionEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.MultiOutputRegression, MultiOutputRegressionEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.QuantileRegression, QuantileRegressionEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.Ranking, RankerEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.Clustering, ClusteringEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.AnomalyDetection, AnomalyDetectionEvaluator.LoadName);
                        tmp.Add(MetadataUtils.Const.ScoreColumnKind.SequenceClassification, "SequenceClassifierEvaluator");
                        Interlocked.CompareExchange(ref _knownEvaluatorLoadNames, tmp, null);
                    }
                    return _knownEvaluatorLoadNames;
                }
            }
        }

        public static SubComponent<IMamlEvaluator, SignatureMamlEvaluator> GetEvaluatorType(IExceptionContext ectx, ISchema schema)
        {
            Contracts.CheckValueOrNull(ectx);
            DvText tmp = default(DvText);
            int col;
            schema.GetMaxMetadataKind(out col, MetadataUtils.Kinds.ScoreColumnSetId, CheckScoreColumnKindIsKnown);
            if (col >= 0)
            {
                schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, col, ref tmp);
                var kind = tmp.ToString();
                var map = DefaultEvaluatorTable.Instance;
                // The next assert is guaranteed because it is checked in CheckScoreColumnKindIsKnown which is the lambda passed to GetMaxMetadataKind.
                Contracts.Assert(map.ContainsKey(kind));
                return new SubComponent<IMamlEvaluator, SignatureMamlEvaluator>(map[kind]);
            }

            schema.GetMaxMetadataKind(out col, MetadataUtils.Kinds.ScoreColumnSetId, CheckScoreColumnKind);
            if (col >= 0)
            {
                schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, col, ref tmp);
                throw ectx.ExceptUserArg(nameof(EvaluateCommand.Arguments.Evaluator), "No default evaluator found for score column kind '{0}'.", tmp.ToString());
            }

            throw ectx.ExceptParam(nameof(schema), "No score columns have been automatically detected.");
        }

        // Lambda used as validator/filter in calls to GetMaxMetadataKind.
        private static bool CheckScoreColumnKindIsKnown(ISchema schema, int col)
        {
            var columnType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, col);
            if (columnType == null || !columnType.IsText)
                return false;
            DvText tmp = default(DvText);
            schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, col, ref tmp);
            var map = DefaultEvaluatorTable.Instance;
            return map.ContainsKey(tmp.ToString());
        }

        // Lambda used as validator/filter in calls to GetMaxMetadataKind.
        private static bool CheckScoreColumnKind(ISchema schema, int col)
        {
            var columnType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, col);
            return columnType != null && columnType.IsText;
        }

        /// <summary>
        /// Find the score column to use. If name is specified, that is used. Otherwise, this searches for the
        /// most recent score set of the given kind. If there is no such score set and defName is specifed it
        /// uses defName. Otherwise, it throws.
        /// </summary>
        public static ColumnInfo GetScoreColumnInfo(IExceptionContext ectx, ISchema schema, string name, string argName, string kind,
            string valueKind = MetadataUtils.Const.ScoreValueKind.Score, string defName = null)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValueOrNull(name);
            ectx.CheckNonEmpty(argName, nameof(argName));
            ectx.CheckNonEmpty(kind, nameof(kind));
            ectx.CheckNonEmpty(valueKind, nameof(valueKind));

            int colTmp;
            ColumnInfo info;
            if (!string.IsNullOrWhiteSpace(name))
            {
#pragma warning disable TLC_ContractsNameUsesNameof
                if (!ColumnInfo.TryCreateFromName(schema, name, out info))
                    throw ectx.ExceptUserArg(argName, "Score column is missing");
#pragma warning restore TLC_ContractsNameUsesNameof
                return info;
            }

            var maxSetNum = schema.GetMaxMetadataKind(out colTmp, MetadataUtils.Kinds.ScoreColumnSetId,
                (s, c) => IsScoreColumnKind(ectx, s, c, kind));

            DvText tmp = default(DvText);
            foreach (var col in schema.GetColumnSet(MetadataUtils.Kinds.ScoreColumnSetId, maxSetNum))
            {
#if DEBUG
                schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, col, ref tmp);
                ectx.Assert(tmp.EqualsStr(kind));
#endif
                // REVIEW: What should this do about hidden columns? Currently we ignore them.
                if (schema.IsHidden(col))
                    continue;
                if (schema.TryGetMetadata(TextType.Instance, MetadataUtils.Kinds.ScoreValueKind, col, ref tmp) &&
                    tmp.EqualsStr(valueKind))
                {
                    return ColumnInfo.CreateFromIndex(schema, col);
                }
            }

            if (!string.IsNullOrWhiteSpace(defName) && ColumnInfo.TryCreateFromName(schema, defName, out info))
                return info;

#pragma warning disable TLC_ContractsNameUsesNameof
            throw ectx.ExceptUserArg(argName, "Score column is missing");
#pragma warning restore TLC_ContractsNameUsesNameof
        }

        /// <summary>
        /// Find the optional auxilliary score column to use. If name is specified, that is used.
        /// Otherwise, if colScore is part of a score set, this looks in the score set for a column
        /// with the given valueKind. If none is found, it returns null.
        /// </summary>
        public static ColumnInfo GetOptAuxScoreColumnInfo(IExceptionContext ectx, ISchema schema, string name, string argName,
            int colScore, string valueKind, Func<ColumnType, bool> testType)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValueOrNull(name);
            ectx.CheckNonEmpty(argName, nameof(argName));
            ectx.CheckParam(0 <= colScore && colScore < schema.ColumnCount, nameof(colScore));
            ectx.CheckNonEmpty(valueKind, nameof(valueKind));

            if (!string.IsNullOrWhiteSpace(name))
            {
                ColumnInfo info;
#pragma warning disable TLC_ContractsNameUsesNameof
                if (!ColumnInfo.TryCreateFromName(schema, name, out info))
                    throw ectx.ExceptUserArg(argName, "{0} column is missing", valueKind);
                if (!testType(info.Type))
                    throw ectx.ExceptUserArg(argName, "{0} column has incompatible type", valueKind);
#pragma warning restore TLC_ContractsNameUsesNameof
                return info;
            }

            // Get the score column set id from colScore.
            var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnSetId, colScore);
            if (type == null || !type.IsKey || type.RawKind != DataKind.U4)
            {
                // scoreCol is not part of a score column set, so can't determine an aux column.
                return null;
            }
            uint setId = 0;
            schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnSetId, colScore, ref setId);

            DvText tmp = default(DvText);
            foreach (var col in schema.GetColumnSet(MetadataUtils.Kinds.ScoreColumnSetId, setId))
            {
                // REVIEW: What should this do about hidden columns? Currently we ignore them.
                if (schema.IsHidden(col))
                    continue;
                if (schema.TryGetMetadata(TextType.Instance, MetadataUtils.Kinds.ScoreValueKind, col, ref tmp) &&
                    tmp.EqualsStr(valueKind))
                {
                    var res = ColumnInfo.CreateFromIndex(schema, col);
                    if (testType(res.Type))
                        return res;
                }
            }

            // Didn't find it in the score column set.
            return null;
        }

        public static bool IsScoreColumnKind(IExceptionContext ectx, ISchema schema, int col, string kind)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckParam(0 <= col && col < schema.ColumnCount, nameof(col));
            ectx.CheckNonEmpty(kind, nameof(kind));

            var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, col);
            if (type == null || !type.IsText)
                return false;
            var tmp = default(DvText);
            schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, col, ref tmp);
            return tmp.EqualsStr(kind);
        }

        /// <summary>
        /// If str is non-empty, returns it. Otherwise if info is non-null, returns info.Name.
        /// Otherwise, returns def.
        /// </summary>
        public static string GetColName(string str, ColumnInfo info, string def)
        {
            Contracts.CheckValueOrNull(str);
            Contracts.CheckValueOrNull(info);
            Contracts.CheckValueOrNull(def);

            if (!string.IsNullOrEmpty(str))
                return str;
            if (info != null)
                return info.Name;
            return def;
        }

        public static void CheckWeightType(IExceptionContext ectx, ColumnType type)
        {
            ectx.AssertValue(type);
            if (type != NumberType.Float)
                throw ectx.ExceptUserArg(nameof(EvaluateCommand.Arguments.WeightColumn), "Incompatible Weight column. Weight column type must be {0}.", NumberType.Float);
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

            using (var cursor = metricsView.GetRowCursor(col => true))
            {
                DvBool isWeighted = DvBool.False;
                ValueGetter<DvBool> isWeightedGetter;
                if (hasWeighted)
                    isWeightedGetter = cursor.GetGetter<DvBool>(isWeightedCol);
                else
                    isWeightedGetter = (ref DvBool dst) => dst = DvBool.False;

                ValueGetter<uint> stratColGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema.GetColumnType(stratCol);
                    stratColGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratColGetter = (ref uint dst) => dst = 0;

                // We currently have only double valued or vector of double valued metrics.
                var colCount = schema.ColumnCount;
                var getters = new ValueGetter<double>[colCount];
                var vBufferGetters = getVectorMetrics ? new ValueGetter<VBuffer<double>>[colCount] : null;

                for (int i = 0; i < schema.ColumnCount; i++)
                {
                    if (schema.IsHidden(i) || hasWeighted && i == isWeightedCol ||
                        hasStrats && (i == stratCol || i == stratVal))
                        continue;

                    var type = schema.GetColumnType(i);
                    if (type == NumberType.R8 || type == NumberType.R4)
                        getters[i] = RowCursorUtils.GetGetterAs<double>(NumberType.R8, cursor, i);
                    else if (type.IsKnownSizeVector && type.ItemType == NumberType.R8 && getVectorMetrics)
                        vBufferGetters[i] = cursor.GetGetter<VBuffer<double>>(i);
                }

                Double metricVal = 0;
                VBuffer<double> metricVals = default(VBuffer<double>);
                uint strat = 0;
                bool foundRow = false;
                while (cursor.MoveNext())
                {
                    isWeightedGetter(ref isWeighted);
                    if (isWeighted.IsTrue)
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
                            yield return new KeyValuePair<string, double>(schema.GetColumnName(i), metricVal);
                        }
                        else if (getVectorMetrics && vBufferGetters[i] != null)
                        {
                            vBufferGetters[i](ref metricVals);

                            // For R8 vector valued columns the names of the metrics are the column name, 
                            // followed by the slot name if it exists, or "Label_i" if it doesn't. 
                            VBuffer<DvText> names = default(VBuffer<DvText>);
                            var size = schema.GetColumnType(i).VectorSize;
                            var slotNamesType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, i);
                            if (slotNamesType != null && slotNamesType.VectorSize == size && slotNamesType.ItemType.IsText)
                                schema.GetMetadata(MetadataUtils.Kinds.SlotNames, i, ref names);
                            else
                            {
                                var namesArray = new DvText[size];
                                for (int j = 0; j < size; j++)
                                    namesArray[j] = new DvText(string.Format("({0})", j));
                                names = new VBuffer<DvText>(size, namesArray);
                            }
                            var colName = schema.GetColumnName(i);
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

        public static IDataView AddTextColumn<TSrc>(IHostEnvironment env, IDataView input, string inputColName, string outputColName,
            ColumnType typeSrc, string value, string registrationName)
        {
            Contracts.Check(typeSrc.RawType == typeof(TSrc));
            return LambdaColumnMapper.Create(env, registrationName, input, inputColName, outputColName, typeSrc, TextType.Instance,
                (ref TSrc src, ref DvText dst) => dst = new DvText(value));
        }

        public static IDataView AddKeyColumn<TSrc>(IHostEnvironment env, IDataView input, string inputColName, string outputColName,
            ColumnType typeSrc, int keyCount, int value, string registrationName, ValueGetter<VBuffer<DvText>> keyValueGetter)
        {
            Contracts.Check(typeSrc.RawType == typeof(TSrc));
            return LambdaColumnMapper.Create(env, registrationName, input, inputColName, outputColName, typeSrc,
                new KeyType(DataKind.U4, 0, keyCount), (ref TSrc src, ref uint dst) =>
                {
                    if (value < 0 || value > keyCount)
                        dst = 0;
                    else
                        dst = (uint)value;
                }, keyValueGetter);
        }

        /// <summary>
        /// This method takes an array of data views and a specified input vector column, and adds a new output column to each of the data views.
        /// First, we find the union set of the slot names in the different data views. Next we define a new vector column for each 
        /// data view, indexed by the union of the slot names. For each data view, every slot value is the value in the slot corresponding
        /// to its slot name in the original column. If a reconciled slot name does not exist in an input column, the value in the output
        /// column is def.
        /// </summary>
        public static void ReconcileSlotNames<T>(IHostEnvironment env, IDataView[] views, string columnName, PrimitiveType itemType, T def = default(T))
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckValue(itemType, nameof(itemType));
            Contracts.CheckParam(typeof(T) == itemType.RawType, nameof(itemType), "Generic type does not match the item type");

            var numIdvs = views.Length;
            var slotNames = new Dictionary<DvText, int>();
            var maps = new int[numIdvs][];
            var slotNamesCur = default(VBuffer<DvText>);
            var typeSrc = new ColumnType[numIdvs];
            // Create mappings from the original slots to the reconciled slots.
            for (int i = 0; i < numIdvs; i++)
            {
                var idv = views[i];
                int col;
                if (!idv.Schema.TryGetColumnIndex(columnName, out col))
                    throw env.Except("Data view number {0} does not contain column '{1}'", i, columnName);

                var type = typeSrc[i] = idv.Schema.GetColumnType(col);
                if (!idv.Schema.HasSlotNames(col, type.VectorSize))
                    throw env.Except("Column '{0}' in data view number {1} did not contain slot names metadata", columnName, i);
                idv.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, col, ref slotNamesCur);

                var map = maps[i] = new int[slotNamesCur.Length];
                foreach (var kvp in slotNamesCur.Items(true))
                {
                    var index = kvp.Key;
                    var name = kvp.Value;
                    if (!slotNames.ContainsKey(name))
                        slotNames[name] = slotNames.Count;
                    map[index] = slotNames[name];
                }
            }

            var reconciledSlotNames = new VBuffer<DvText>(slotNames.Count, slotNames.Keys.ToArray());
            ValueGetter<VBuffer<DvText>> slotNamesGetter =
                (ref VBuffer<DvText> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < reconciledSlotNames.Length)
                        values = new DvText[reconciledSlotNames.Length];

                    Array.Copy(reconciledSlotNames.Values, values, reconciledSlotNames.Length);
                    dst = new VBuffer<DvText>(reconciledSlotNames.Length, values, dst.Indices);
                };

            // For each input data view, create the reconciled key column by wrapping it in a LambdaColumnMapper.
            for (int i = 0; i < numIdvs; i++)
            {
                var map = maps[i];

                ValueMapper<VBuffer<T>, VBuffer<T>> mapper;
                if (def.Equals(default(T)))
                {
                    mapper =
                        (ref VBuffer<T> src, ref VBuffer<T> dst) =>
                        {
                            Contracts.Assert(src.Length == Utils.Size(map));

                            var values = dst.Values;
                            if (Utils.Size(values) < slotNames.Count)
                                values = new T[slotNames.Count];
                            foreach (var kvp in src.Items())
                                values[map[kvp.Key]] = kvp.Value;
                            dst = new VBuffer<T>(slotNames.Count, values, dst.Indices);
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
                        (ref VBuffer<T> src, ref VBuffer<T> dst) =>
                        {
                            Contracts.Assert(src.Length == Utils.Size(map));
                            var values = dst.Values;
                            if (Utils.Size(values) < slotNames.Count)
                                values = new T[slotNames.Count];

                            foreach (var kvp in src.Items(true))
                                values[map[kvp.Key]] = kvp.Value;
                            foreach (var j in naIndices)
                                values[j] = def;
                            dst = new VBuffer<T>(slotNames.Count, values, dst.Indices);
                        };
                }

                var typeDst = new VectorType(itemType, slotNames.Count);
                views[i] = LambdaColumnMapper.Create(env, "ReconciledSlotNames", views[i],
                    columnName, columnName, typeSrc[i], typeDst, mapper, slotNamesGetter: slotNamesGetter);
            }
        }

        private static int[][] MapKeys(ISchema[] schemas, string columnName, bool isVec,
            out int[] indices, out Dictionary<DvText, int> reconciledKeyNames)
        {
            var dvCount = schemas.Length;
            var keyValueMappers = new int[dvCount][];
            var keyNamesCur = default(VBuffer<DvText>);
            indices = new int[dvCount];
            reconciledKeyNames = new Dictionary<DvText, int>();
            for (int i = 0; i < dvCount; i++)
            {
                var schema = schemas[i];
                if (!schema.TryGetColumnIndex(columnName, out indices[i]))
                    throw Contracts.Except($"Schema number {i} does not contain column '{columnName}'");

                var type = schema.GetColumnType(indices[i]);
                if (type.IsVector != isVec)
                    throw Contracts.Except($"Column '{columnName}' in schema number {i} does not have the correct type");
                if (!schema.HasKeyNames(indices[i], type.ItemType.KeyCount))
                    throw Contracts.Except($"Column '{columnName}' in schema number {i} does not have text key values");
                if (!type.ItemType.IsKey || type.ItemType.RawKind != DataKind.U4)
                    throw Contracts.Except($"Column '{columnName}' must be a U4 key type, but is '{type.ItemType}'");

                schema.GetMetadata(MetadataUtils.Kinds.KeyValues, indices[i], ref keyNamesCur);

                keyValueMappers[i] = new int[type.ItemType.KeyCount];
                foreach (var kvp in keyNamesCur.Items(true))
                {
                    var key = kvp.Key;
                    var name = kvp.Value;
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
        public static void ReconcileKeyValues(IHostEnvironment env, IDataView[] views, string columnName)
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckNonEmpty(columnName, nameof(columnName));

            var dvCount = views.Length;

            Dictionary<DvText, int> keyNames;
            int[] indices;
            // Create mappings from the original key types to the reconciled key type.
            var keyValueMappers = MapKeys(views.Select(view => view.Schema).ToArray(), columnName, false, out indices, out keyNames);
            var keyType = new KeyType(DataKind.U4, 0, keyNames.Count);
            var keyNamesVBuffer = new VBuffer<DvText>(keyNames.Count, keyNames.Keys.ToArray());
            ValueGetter<VBuffer<DvText>> keyValueGetter =
                    (ref VBuffer<DvText> dst) =>
                        dst = new VBuffer<DvText>(keyNamesVBuffer.Length, keyNamesVBuffer.Count, keyNamesVBuffer.Values, keyNamesVBuffer.Indices);

            // For each input data view, create the reconciled key column by wrapping it in a LambdaColumnMapper.
            for (int i = 0; i < dvCount; i++)
            {
                var keyMapperCur = keyValueMappers[i];
                ValueMapper<uint, uint> mapper =
                    (ref uint src, ref uint dst) =>
                    {
                        if (src == 0 || src > keyMapperCur.Length)
                            dst = 0;
                        else
                            dst = (uint)keyMapperCur[src - 1] + 1;
                    };
                views[i] = LambdaColumnMapper.Create(env, "ReconcileKeyValues", views[i], columnName, columnName,
                    views[i].Schema.GetColumnType(indices[i]), keyType, mapper, keyValueGetter);
            }
        }

        /// <summary>
        /// This method is similar to <see cref="ReconcileKeyValues"/>, but it reconciles the key values over vector
        /// input columns.
        /// </summary>
        public static void ReconcileVectorKeyValues(IHostEnvironment env, IDataView[] views, string columnName)
        {
            Contracts.CheckNonEmpty(views, nameof(views));
            Contracts.CheckNonEmpty(columnName, nameof(columnName));

            var dvCount = views.Length;

            Dictionary<DvText, int> keyNames;
            int[] columnIndices;
            var keyValueMappers = MapKeys(views.Select(view => view.Schema).ToArray(), columnName, true, out columnIndices, out keyNames);
            var keyType = new KeyType(DataKind.U4, 0, keyNames.Count);
            var keyNamesVBuffer = new VBuffer<DvText>(keyNames.Count, keyNames.Keys.ToArray());
            ValueGetter<VBuffer<DvText>> keyValueGetter =
                    (ref VBuffer<DvText> dst) =>
                        dst = new VBuffer<DvText>(keyNamesVBuffer.Length, keyNamesVBuffer.Count, keyNamesVBuffer.Values, keyNamesVBuffer.Indices);

            for (int i = 0; i < dvCount; i++)
            {
                var keyMapperCur = keyValueMappers[i];
                ValueMapper<VBuffer<uint>, VBuffer<uint>> mapper =
                    (ref VBuffer<uint> src, ref VBuffer<uint> dst) =>
                    {
                        var values = dst.Values;
                        if (Utils.Size(values) < src.Count)
                            values = new uint[src.Count];
                        if (src.IsDense)
                        {
                            for (int j = 0; j < src.Length; j++)
                            {
                                if (src.Values[j] == 0 || src.Values[j] > keyMapperCur.Length)
                                    values[j] = 0;
                                else
                                    values[j] = (uint)keyMapperCur[src.Values[j] - 1] + 1;
                            }
                            dst = new VBuffer<uint>(src.Length, values, dst.Indices);
                        }
                        else
                        {
                            var indices = dst.Indices;
                            if (Utils.Size(indices) < src.Count)
                                indices = new int[src.Count];
                            for (int j = 0; j < src.Count; j++)
                            {
                                if (src.Values[j] == 0 || src.Values[j] > keyMapperCur.Length)
                                    values[j] = 0;
                                else
                                    values[j] = (uint)keyMapperCur[src.Values[j] - 1] + 1;
                                indices[j] = src.Indices[j];
                            }
                            dst = new VBuffer<uint>(src.Length, src.Count, values, indices);
                        }
                    };

                ValueGetter<VBuffer<DvText>> slotNamesGetter = null;
                var type = views[i].Schema.GetColumnType(columnIndices[i]);
                if (views[i].Schema.HasSlotNames(columnIndices[i], type.VectorSize))
                {
                    var schema = views[i].Schema;
                    int index = columnIndices[i];
                    slotNamesGetter =
                        (ref VBuffer<DvText> dst) => schema.GetMetadata(MetadataUtils.Kinds.SlotNames, index, ref dst);
                }
                views[i] = LambdaColumnMapper.Create(env, "ReconcileKeyValues", views[i], columnName, columnName,
                    type, new VectorType(keyType, type.AsVector), mapper, keyValueGetter, slotNamesGetter);
            }
        }

    }

    public static class MetricWriter
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
            var type = confusionDataView.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, countCol);
            host.Check(type != null && type.IsKnownSizeVector && type.ItemType.IsText, "The Count column does not have a text vector metadata of kind SlotNames.");

            var labelNames = default(VBuffer<DvText>);
            confusionDataView.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, countCol, ref labelNames);
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

            var confusionTableString = GetConfusionTableAsString(confusionTable, recallSums, precisionSums,
               labelNames.Values.Where((t, i) => labelIndexToConfIndexMap[i] >= 0).ToArray(),
               sampled: numConfusionTableLabels < labelNames.Count, binary: binary);

            int weightIndex;
            if (confusionDataView.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.Weight, out weightIndex))
            {
                confusionTable = GetConfusionTableAsArray(confusionDataView, weightIndex, labelNames.Length,
                   labelIndexToConfIndexMap, numConfusionTableLabels, out precisionSums, out recallSums);
                weightedConfusionTable = GetConfusionTableAsString(confusionTable, recallSums, precisionSums,
                    labelNames.Values.Where((t, i) => labelIndexToConfIndexMap[i] >= 0).ToArray(),
                    sampled: numConfusionTableLabels < labelNames.Count, prefix: "Weighted ", binary: binary);
            }
            else
                weightedConfusionTable = null;

            return confusionTableString;
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
            using (var cursor = confusionDataView.GetRowCursor(col => col == countIndex || hasStrat && col == stratCol))
            {
                var type = cursor.Schema.GetColumnType(countIndex);
                Contracts.Check(type.IsKnownSizeVector && type.ItemType == NumberType.R8);
                var countGetter = cursor.GetGetter<VBuffer<double>>(countIndex);
                ValueGetter<uint> stratGetter = null;
                if (hasStrat)
                {
                    type = cursor.Schema.GetColumnType(stratCol);
                    stratGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
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
        /// then the data view should also contain a DvBool column named "IsWeighted".</param>
        /// <param name="weightedMetrics">If the IsWeighted column exists, this is assigned the string representation of the weighted
        /// metrics. Otherwise it is assigned null.</param>
        public static string GetPerFoldResults(IHostEnvironment env, IDataView fold, out string weightedMetrics)
        {
            IDataView avgMetrics;
            int isWeightedCol;
            if (fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out isWeightedCol))
                weightedMetrics = GetMetricsAsString(env, fold, true, 1, out avgMetrics);
            else
                weightedMetrics = null;
            return GetMetricsAsString(env, fold, false, 1, out avgMetrics);
        }

        // This method returns a string representation of a set of metrics. If there are stratification columns, it looks for columns named
        // StratCol and StratVal, and outputs the metrics in the rows with NA in the StratCol column. If weighted is true, it looks 
        // for a DvBool column named "IsWeighted" and outputs the metrics in the rows with a value of true in that column. 
        // If nonAveragedCols is non-null, it computes the average and standard deviation over all the relevant rows and populates
        // nonAveragedCols with columns that are either hidden, or are not of a type that we can display (i.e., either a numeric column,
        // or a known length vector of doubles).
        // If average is false, no averaging is done, and instead we check that there is exactly one relevant row. Otherwise, we
        // add the vector columns of variable length of the list of non-averagable columns if nonAveragedCols is not null.
        private static string GetMetricsAsString(IHostEnvironment env, IDataView data, bool weighted,
            int numFolds, out IDataView avgMetricsDataView, bool average = false, List<string> nonAveragedCols = null)
        {
            int isWeightedCol;
            bool hasWeighted = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out isWeightedCol);
            // If the IsWeighted column is not present, weighted must be false.
            Contracts.Assert(hasWeighted || !weighted);

            int stratCol;
            bool hasStrats = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            int stratVal;
            bool hasStratVals = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            Contracts.Assert(hasStrats == hasStratVals);

            int foldCol;
            bool hasFoldCol = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.FoldIndex, out foldCol);

            // We currently have only double valued or vector of double valued metrics.
            var colCount = data.Schema.ColumnCount;
            var getters = new ValueGetter<double>[colCount];
            var vBufferGetters = new ValueGetter<VBuffer<double>>[colCount];

            double[] avgMetrics;
            double[] sumSqMetrics;
            List<string> metricNames;
            int numResults = 0;
            using (var cursor = data.GetRowCursor(col => true))
            {
                DvBool isWeighted = DvBool.False;
                ValueGetter<DvBool> isWeightedGetter;
                if (hasWeighted)
                    isWeightedGetter = cursor.GetGetter<DvBool>(isWeightedCol);
                else
                    isWeightedGetter = (ref DvBool dst) => dst = DvBool.False;

                ValueGetter<uint> stratColGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema.GetColumnType(stratCol);
                    stratColGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratColGetter = (ref uint dst) => dst = 0;

                // Get the names of the metrics. For R8 valued columns the metric name is the column name. For R8 vector valued columns
                // the names of the metrics are the column name, followed by the slot name if it exists, or "Label_i" if it doesn't.
                using (var ch = env.Register("GetMetricsAsString").Start("Get Metric Names"))
                {
                    metricNames = GetMetricNames(ch, data.Schema, cursor,
                        i => hasWeighted && i == isWeightedCol || hasStrats && (i == stratCol || i == stratVal) ||
                            hasFoldCol && i == foldCol, getters, vBufferGetters);
                    ch.Done();
                }

                Double metricVal = 0;
                VBuffer<Double> metricVals = default(VBuffer<Double>);
                avgMetrics = new double[metricNames.Count];
                sumSqMetrics = new double[metricNames.Count];
                uint strat = 0;
                while (cursor.MoveNext())
                {
                    isWeightedGetter(ref isWeighted);
                    if (isWeighted.IsTrue != weighted)
                        continue;

                    stratColGetter(ref strat);
                    // REVIEW: how to print stratified results?
                    if (strat > 0)
                        continue;

                    // If !average, we should have only one relevant row.
                    if (!average && numResults > 0)
                        throw Contracts.Except("Multiple {0} rows found in metrics data view.", weighted ? "weighted" : "unweighted");

                    numResults++;
                    int iMetric = 0;
                    for (int i = 0; i < colCount; i++)
                    {
                        if (hasWeighted && i == isWeightedCol || hasStrats && (i == stratCol || i == stratVal))
                            continue;

                        // REVIEW: What to do with metrics that are not doubles?
                        if (getters[i] != null)
                        {
                            getters[i](ref metricVal);
                            avgMetrics[iMetric] += metricVal;
                            if (sumSqMetrics != null)
                                sumSqMetrics[iMetric] += metricVal * metricVal;
                            iMetric++;
                        }
                        else if (vBufferGetters[i] != null)
                        {
                            vBufferGetters[i](ref metricVals);
                            foreach (var metric in metricVals.Items(all: true))
                            {
                                avgMetrics[iMetric] += metric.Value;
                                if (sumSqMetrics != null)
                                    sumSqMetrics[iMetric] += metric.Value * metric.Value;
                                iMetric++;
                            }
                        }
                    }
                    Contracts.Assert(iMetric == metricNames.Count);

                    if (numResults == numFolds)
                        break;
                }
            }

            var sb = new StringBuilder();
            for (int i = 0; i < metricNames.Count; i++)
            {
                avgMetrics[i] /= numResults;
                sb.Append(string.Format("{0}{1}: ", weighted ? "Weighted " : "", metricNames[i]).PadRight(20));
                sb.Append(string.Format(CultureInfo.InvariantCulture, "{0,7:N6}", avgMetrics[i]));
                if (average)
                {
                    Contracts.AssertValue(sumSqMetrics);
                    sb.AppendLine(string.Format(" ({0:N4})", numResults == 1 ? 0 :
                        Math.Sqrt(sumSqMetrics[i] / numResults - avgMetrics[i] * avgMetrics[i])));
                }
                else
                    sb.AppendLine();
            }

            if (average)
            {
                var dvBldr = new ArrayDataViewBuilder(env);
                int iMetric = 0;
                for (int i = 0; i < colCount; i++)
                {
                    if (hasStrats && i == stratCol)
                    {
                        var type = data.Schema.GetColumnType(i);
                        var keyValuesType = data.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, i);
                        if (keyValuesType == null || !keyValuesType.ItemType.IsText ||
                            keyValuesType.VectorSize != type.KeyCount)
                        {
                            throw env.Except("Column '{0}' must have key values metadata",
                                MetricKinds.ColumnNames.StratCol);
                        }

                        ValueGetter<VBuffer<DvText>> getKeyValues =
                            (ref VBuffer<DvText> dst) =>
                            {
                                data.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, stratCol, ref dst);
                                Contracts.Assert(dst.IsDense);
                            };

                        dvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, getKeyValues, 0, type.KeyCount, (uint)0);
                    }
                    else if (hasStratVals && i == stratVal)
                        dvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, DvText.NA);
                    else if (hasWeighted && i == isWeightedCol)
                        dvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BoolType.Instance, weighted ? DvBool.True : DvBool.False);
                    else if (hasFoldCol && i == foldCol)
                    {
                        var avg = new DvText("Average");
                        dvBldr.AddColumn(MetricKinds.ColumnNames.FoldIndex, TextType.Instance, avg);
                    }
                    else if (getters[i] != null)
                    {
                        dvBldr.AddColumn(data.Schema.GetColumnName(i), NumberType.R8, avgMetrics[iMetric]);
                        iMetric++;
                    }
                    else if (vBufferGetters[i] != null)
                    {
                        var type = data.Schema.GetColumnType(i);
                        var vectorMetrics = new double[type.VectorSize];
                        env.Assert(vectorMetrics.Length > 0);
                        Array.Copy(avgMetrics, iMetric, vectorMetrics, 0, vectorMetrics.Length);
                        var slotNamesType = data.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, i);
                        var name = data.Schema.GetColumnName(i);
                        var slotNames = default(VBuffer<DvText>);
                        if (slotNamesType != null && slotNamesType.ItemType.IsText &&
                            slotNamesType.VectorSize == type.VectorSize)
                        {
                            data.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, i, ref slotNames);
                            Contracts.Assert(slotNames.IsDense);
                            var values = slotNames.Values;
                            for (int j = 0; j < values.Length; j++)
                                values[j] = new DvText(name + values[j]);
                            slotNames = new VBuffer<DvText>(slotNames.Length, values, slotNames.Indices);
                        }
                        else
                        {
                            var values = slotNames.Values;
                            if (Utils.Size(values) < type.VectorSize)
                                values = new DvText[type.VectorSize];
                            for (int j = 0; j < type.VectorSize; j++)
                                values[j] = new DvText(name + j);
                            slotNames = new VBuffer<DvText>(type.VectorSize, values, slotNames.Indices);
                        }
                        ValueGetter<VBuffer<DvText>> getSlotNames = (ref VBuffer<DvText> dst) => dst = slotNames;
                        dvBldr.AddColumn(name, getSlotNames, NumberType.R8, new[] { vectorMetrics });
                        iMetric += vectorMetrics.Length;
                    }
                    else
                        nonAveragedCols?.Add(data.Schema.GetColumnName(i));
                }
                Contracts.Assert(iMetric == metricNames.Count);
                avgMetricsDataView = dvBldr.GetDataView();
            }
            else
                avgMetricsDataView = null;

            return sb.ToString();
        }

        private static List<string> GetMetricNames(IChannel ch, ISchema schema, IRow row, Func<int, bool> ignoreCol,
            ValueGetter<double>[] getters, ValueGetter<VBuffer<double>>[] vBufferGetters)
        {
            Contracts.AssertValue(schema);
            Contracts.AssertValue(row);
            Contracts.Assert(Utils.Size(getters) == schema.ColumnCount);
            Contracts.Assert(Utils.Size(vBufferGetters) == schema.ColumnCount);

            // Get the names of the metrics. For R8 valued columns the metric name is the column name. For R8 vector valued columns
            // the names of the metrics are the column name, followed by the slot name if it exists, or "Label_i" if it doesn't. 
            VBuffer<DvText> names = default(VBuffer<DvText>);
            int metricCount = 0;
            var metricNames = new List<string>();
            for (int i = 0; i < schema.ColumnCount; i++)
            {
                if (schema.IsHidden(i) || ignoreCol(i))
                    continue;

                var type = schema.GetColumnType(i);
                var metricName = row.Schema.GetColumnName(i);
                if (type.IsNumber)
                {
                    getters[i] = RowCursorUtils.GetGetterAs<double>(NumberType.R8, row, i);
                    metricNames.Add(metricName);
                    metricCount++;
                }
                else if (type.IsVector && type.ItemType == NumberType.R8)
                {
                    if (type.VectorSize == 0)
                    {
                        ch.Warning("Vector metric '{0}' has different lengths in different folds and will not be averaged for overall results.", metricName);
                        continue;
                    }

                    vBufferGetters[i] = row.GetGetter<VBuffer<double>>(i);
                    metricCount += type.VectorSize;
                    var slotNamesType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, i);
                    if (slotNamesType != null && slotNamesType.VectorSize == type.VectorSize && slotNamesType.ItemType.IsText)
                        schema.GetMetadata(MetadataUtils.Kinds.SlotNames, i, ref names);
                    else
                    {
                        var namesArray = names.Values;
                        if (Utils.Size(namesArray) < type.VectorSize)
                            namesArray = new DvText[type.VectorSize];
                        for (int j = 0; j < type.VectorSize; j++)
                            namesArray[j] = new DvText(string.Format("Label_{0}", j));
                        names = new VBuffer<DvText>(type.VectorSize, namesArray);
                    }
                    foreach (var name in names.Items(all: true))
                        metricNames.Add(string.Format("{0} {1}", metricName, name.Value));
                }
            }
            Contracts.Assert(metricNames.Count == metricCount);
            return metricNames;
        }

        // Get a string representation of a confusion table.
        private static string GetConfusionTableAsString(double[][] confusionTable, double[] rowSums, double[] columnSums,
            DvText[] predictedLabelNames, string prefix = "", bool sampled = false, bool binary = true)
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
                rowDigitLen = Math.Max(predictedLabelNames.Length - 1, 0).ToString().Length;
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
            var sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine("OVERALL RESULTS");
            sb.AppendLine("---------------------------------------");

            int isWeighted;
            IDataView weightedAvgMetrics = null;
            var nonAveragedCols = new List<string>();
            if (overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out isWeighted))
                sb.Append(GetMetricsAsString(env, overall, true, numFolds, out weightedAvgMetrics, true));
            IDataView avgMetrics;
            sb.AppendLine(GetMetricsAsString(env, overall, false, numFolds, out avgMetrics, true, nonAveragedCols));
            env.AssertValue(avgMetrics);
            sb.AppendLine("---------------------------------------");
            ch.Info(sb.ToString());

            if (!string.IsNullOrEmpty(filename))
            {
                using (var file = env.CreateOutputFile(filename))
                {
                    // idvList will contain all the dataviews that should be appended with AppendRowsDataView.
                    // If numResults=1, then we just save the average metrics. Otherwise, we remove all the non-metric columns
                    // (except for the IsWeighted column and FoldIndex column if present), and append to the average results.
                    var idvList = new List<IDataView>() { avgMetrics };
                    if (weightedAvgMetrics != null)
                        idvList.Add(weightedAvgMetrics);

                    int stratCol;
                    var hasStrat = overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
                    if (numFolds > 1 || hasStrat)
                    {
                        if (Utils.Size(nonAveragedCols) > 0)
                        {
                            var dropArgs = new DropColumnsTransform.Arguments() { Column = nonAveragedCols.ToArray() };
                            overall = new DropColumnsTransform(env, dropArgs, overall);
                        }
                        idvList.Add(overall);
                    }

                    var summary = AppendRowsDataView.Create(env, avgMetrics.Schema, idvList.ToArray());

                    // If there are stratified results, apply a KeyToValue transform to get the stratification column
                    // names from the key column.
                    if (hasStrat)
                    {
                        var args = new KeyToValueTransform.Arguments();
                        args.Column = new[] { new KeyToValueTransform.Column() { Source = MetricKinds.ColumnNames.StratCol }, };
                        summary = new KeyToValueTransform(env, args, summary);
                    }

                    var saverArgs = new TextSaver.Arguments() { Dense = true, Silent = true };
                    DataSaverUtils.SaveDataView(ch, new TextSaver(env, saverArgs), summary, file);
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
                int col;
                if (warnings.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.WarningText, out col) && warnings.Schema.GetColumnType(col).IsText)
                {
                    using (var cursor = warnings.GetRowCursor(c => c == col))
                    {
                        var warning = default(DvText);
                        var getter = cursor.GetGetter<DvText>(col);
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
            var type = data.Schema.GetColumnType(stratCol);
            env.Check(type.KeyCount > 0, "Expected a known count key type stratification column");
            var filterArgs = new NAFilter.Arguments();
            filterArgs.Column = new[] { MetricKinds.ColumnNames.StratCol };
            filterArgs.Complement = true;
            data = new NAFilter(env, filterArgs, data);

            int stratVal;
            var found = data.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            env.Check(found, "If stratification column exist, data view must also contain a StratVal column");

            var dropArgs = new DropColumnsTransform.Arguments();
            dropArgs.Column = new[] { data.Schema.GetColumnName(stratCol), data.Schema.GetColumnName(stratVal) };
            data = new DropColumnsTransform(env, dropArgs, data);
            return data;
        }
    }

    /// <summary>
    /// This is a list of string constants denoting 'standard' metric kinds.
    /// </summary>
    public static class MetricKinds
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
