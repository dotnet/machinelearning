// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Featurization ideas inspired from:
    /// https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
    /// </summary>
    public static class DatasetFeatureInference
    {
        public sealed class Stats
        {
            [JsonIgnore] private SummaryStatistics _statistics;

            [JsonIgnore] public double Sum;

            public Stats()
            {
                _statistics = new SummaryStatistics();
            }

            public void Add(double x)
            {
                Sum += x;
                _statistics.Add(x);
            }

            public void Add(IEnumerable<int> x)
            {
                foreach (int n in x)
                    Add(n);
            }

            [JsonProperty]
            public long Count => _statistics.RawCount;

            [JsonProperty]
            public double? NonZeroValueCount => _statistics.RawCount > 20 ? (double?)_statistics.Nonzero : null;

            [JsonProperty]
            public double? Variance => _statistics.RawCount > 20 ? (double?)_statistics.SampleVariance : null;

            [JsonProperty]
            public double? StandardDeviation => _statistics.RawCount > 20 ? (double?)_statistics.SampleStdDev : null;

            [JsonProperty]
            public double? Skewness => _statistics.RawCount > 20 ? (double?)_statistics.Skewness : null;

            [JsonProperty]
            public double? Kurtosis => _statistics.RawCount > 20 ? (double?)_statistics.Kurtosis : null;

            [JsonProperty]
            public double? Mean => _statistics.RawCount > 20 ? (double?)_statistics.Mean : null;

            [JsonIgnore]
            public double Min => _statistics.Min;

            [JsonIgnore]
            public double Max => _statistics.Max;
        }

        public sealed class Column
        {
            [JsonIgnore]
            private readonly string _name;

            [JsonProperty(Order = 1)]
            public string Name => _name?.Substring(0, Math.Min(_name.Length, 100));

            [JsonIgnore]
            public readonly ColumnPurpose ColumnPurpose;

            [JsonProperty(Order = 2)]
            public string Purpose => ColumnPurpose.ToString();

            [JsonIgnore]
            public readonly DataKind? Kind;

            [JsonProperty(Order = 2)]
            public string DataKind => Kind?.ToString();

            [JsonProperty(Order = 2)]
            public readonly int[] Indices;

            [JsonProperty(Order = 2)]
            public string IndicesCount => Indices.Length.ToString();

            public Column(string name, ColumnPurpose purpose, DataKind? dataKind, string ranges)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValue(ranges, nameof(ranges));

                _name = name;
                ColumnPurpose = purpose;
                Kind = dataKind;
                Indices = ColumnGroupingInference.GetRange(ranges);
            }
        }

        public sealed class Arguments
        {
            public readonly ReadOnlyMemory<char>[][] Data;
            public readonly Column[] Columns;
            public readonly long? ApproximateRowCount;
            public readonly long? FullFileSize;
            public readonly bool InferencedSchema;
            public readonly Guid Id;
            public readonly bool PrettyPrint;
            public Arguments(ReadOnlyMemory<char>[][] data, Column[] columns, long? fullFileSize,
                long? approximateRowCount, bool inferencedSchema, Guid id, bool prettyPrint = false)
            {
                Data = data;
                Columns = columns;
                FullFileSize = fullFileSize;
                ApproximateRowCount = approximateRowCount;
                InferencedSchema = inferencedSchema;
                Id = id;
                PrettyPrint = prettyPrint;
            }
        }

        private interface ITypeInferenceExpert
        {
            void Apply(ReadOnlyMemory<char>[][] data, Column[] columns);
            bool AddMe();
            string FeatureName();
        }

        /// <summary>
        /// Meta features about column schema:
        /// 1) Number of features such as Text, Categorical, Numerical, counted in terms of slots.
        /// 2) Column meta data such as Name, DataKind, Purpose and slot indices.
        /// 3) Feature ratio: number of categorical columns to text feature columns, etc.
        /// 4) Count of columns and slots by column purpose.
        /// 5) Log of total feature count.
        /// </summary>
        public sealed class ColumnSchema : ITypeInferenceExpert
        {
            public Column[] Columns;
            public int FeatureCount;
            public int TotalSlotCount;
            public double LogFeatureCount;
            public Ratio FeatureRatio;
            public Dictionary<string, int> ColumnTypeCount;
            public Dictionary<string, int> ColumnSlotCount;

            public struct Ratio
            {
                public double CategoricalToNumerical;
                public double NumericalToCategorical;
                public double CategoricalToText;
                public double TextToCategorical;
                public double TextToNumerical;
                public double NumericalToText;
            }

            public ColumnSchema()
            {
                ColumnTypeCount = new Dictionary<string, int>();
                ColumnSlotCount = new Dictionary<string, int>();
            }

            public bool AddMe() => ColumnTypeCount.Count > 0;

            public string FeatureName() => nameof(ColumnSchema);

            public void Apply(ReadOnlyMemory<char>[][] data, Column[] columns)
            {
                Columns = columns;
                foreach (var column in columns)
                {
                    int slotCount = column.Indices.Length;
                    TotalSlotCount += slotCount;
                    string purposeString = column.ColumnPurpose.ToString();
                    if (ColumnTypeCount.ContainsKey(purposeString))
                        ColumnTypeCount[purposeString] += 1;
                    else
                        ColumnTypeCount.Add(purposeString, 1);

                    if (ColumnSlotCount.ContainsKey(purposeString))
                        ColumnSlotCount[purposeString] += slotCount;
                    else
                        ColumnSlotCount.Add(purposeString, slotCount);

                    if (column.ColumnPurpose == ColumnPurpose.NumericFeature ||
                        column.ColumnPurpose == ColumnPurpose.CategoricalFeature ||
                        column.ColumnPurpose == ColumnPurpose.TextFeature)
                        FeatureCount += slotCount;
                }

                LogFeatureCount = Math.Log(FeatureCount, 2);
                int categoricalFeatureCount;
                ColumnSlotCount.TryGetValue(ColumnPurpose.CategoricalFeature.ToString(), out categoricalFeatureCount);
                int textFeatureCount;
                ColumnSlotCount.TryGetValue(ColumnPurpose.TextFeature.ToString(), out textFeatureCount);
                int numericFeatureCount;
                ColumnSlotCount.TryGetValue(ColumnPurpose.NumericFeature.ToString(), out numericFeatureCount);

                FeatureRatio.CategoricalToNumerical = categoricalFeatureCount * 1.0 / numericFeatureCount;
                FeatureRatio.NumericalToCategorical = numericFeatureCount * 1.0 / categoricalFeatureCount;
                FeatureRatio.CategoricalToText = categoricalFeatureCount * 1.0 / textFeatureCount;
                FeatureRatio.TextToCategorical = textFeatureCount * 1.0 / categoricalFeatureCount;
                FeatureRatio.TextToNumerical = textFeatureCount * 1.0 / numericFeatureCount;
                FeatureRatio.NumericalToText = numericFeatureCount * 1.0 / textFeatureCount;

                Contracts.Check(ColumnSlotCount.Count == ColumnTypeCount.Count);
            }
        }

        /// <summary>
        /// Meta features for Label columns:
        /// 1) Column info such as name, data kind, column purpose, slot indices.
        /// 2) Stats about label counts such as Min, Max counts, standard deviation, mean, median,
        ///    kurtosis, variance, min/max class probabilities.
        /// 3) Missing value count.
        /// </summary>
        public sealed class LabelFeatures : ITypeInferenceExpert
        {
            public struct LabelColumnFeature
            {
                public Column ColumnInfo;
                public Stats ColumnStats;
                public int MissingValueCount;
                public double MaxClassProbability;
                public double MinClassProbability;
            }

            [JsonIgnore] private bool _containsLabelColumns;

            public List<LabelColumnFeature> LabelFeature;

            public LabelFeatures()
            {
                LabelFeature = new List<LabelColumnFeature>();
            }

            private void ApplyCore(ReadOnlyMemory<char>[][] data, Column column)
            {
                _containsLabelColumns = true;
                Dictionary<string, int> histogram = new Dictionary<string, int>();
                int rowCount = data.First().Length;
                int missingValues = 0;
                for (int i = 0; i < rowCount; i++)
                {
                    string label = "";
                    foreach (var index in column.Indices)
                    {
                        if (index >= data.GetLength(0))
                            break;

                        Contracts.Check(data[index].Length > i);

                        label += data[index][i].ToString();
                    }

                    if (histogram.ContainsKey(label))
                        histogram[label] += 1;
                    else
                        histogram.Add(label, 1);
                }

                Stats stats = new Stats();
                stats.Add(histogram.Values);
                LabelFeature.Add(new LabelColumnFeature
                {
                    ColumnInfo = column,
                    MissingValueCount = missingValues,
                    MaxClassProbability = stats.Max / stats.Sum,
                    MinClassProbability = stats.Min / stats.Sum,
                    ColumnStats = stats
                });
            }

            public void Apply(ReadOnlyMemory<char>[][] data, Column[] columns)
            {
                foreach (var column in columns.Where(col => col.ColumnPurpose == ColumnPurpose.Label))
                    ApplyCore(data, column);
            }

            public bool AddMe() => _containsLabelColumns;

            public string FeatureName() => nameof(LabelFeature);
        }

        /// <summary>
        /// Meta features about missing values across the dataset.
        /// </summary>
        public sealed class MissingValues : ITypeInferenceExpert
        {
            public int NumberOfMissingValues;
            public double PercentageOfMissingValues;
            public int NumberOfInstancesWithMissingValues;
            public double PercentageOfInstancesWithMissingValues;
            public int NumberOfFeaturesWithMissingValues;
            public double PercentageOfFeaturesWithMissingValues;

            public void Apply(ReadOnlyMemory<char>[][] data, Column[] columns)
            {
                if (data.GetLength(0) == 0)
                    return;

                BitArray featuresWithMissingValues = new
                    BitArray(columns.Length > 0 ? columns.Max(col => col.Indices.Max()) + 1 : 0);

                int rowCount = data[0].Length;
                for (int i = 0; i < rowCount; i++)
                {
                    bool instanceWithMissingValue = false;
                    foreach (var column in columns)
                    {
                        foreach (int index in column.Indices)
                        {
                            if (index >= data.GetLength(0))
                                break;

                            Contracts.Check(data[index].Length > i);
                        }
                    }

                    if (instanceWithMissingValue)
                        NumberOfInstancesWithMissingValues++;
                }

                int totalSlots = columns.Sum(col => col.Indices.Length);
                PercentageOfMissingValues = NumberOfMissingValues * 1.0 / (rowCount * totalSlots);
                PercentageOfInstancesWithMissingValues = NumberOfInstancesWithMissingValues * 1.0 / rowCount;
                NumberOfFeaturesWithMissingValues = (from bool m in featuresWithMissingValues where m select m).Count();
                PercentageOfFeaturesWithMissingValues = NumberOfFeaturesWithMissingValues * 1.0 / totalSlots;
            }

            public bool AddMe() => NumberOfMissingValues > 0;

            public string FeatureName() => nameof(MissingValues);
        }

        public struct ColumnStatistics
        {
            public Column Column;
            public Stats Stats;
        }

        /// <summary>
        /// Meta features about columns:
        /// For numeric column types it caculates mean, variance, kurtosis, min, max, median, standard deviation.
        /// For non-numeric column types it caculates the same but over string length and number of spaces as seperate stats.
        /// </summary>
        public sealed class ColumnFeatures : ITypeInferenceExpert
        {
            public List<ColumnStatistics> NumericColumnFeatures;
            public List<ColumnStatistics> NonNumericColumnLengthFeature;
            public List<ColumnStatistics> NonNumericColumnSpacesFeature;
            public Dictionary<string, Stats> StatsPerColumnPurpose;
            public Dictionary<string, Stats> StatsPerColumnPurposeWithSpaces;

            public ColumnFeatures()
            {
                NumericColumnFeatures = new List<ColumnStatistics>();
                NonNumericColumnLengthFeature = new List<ColumnStatistics>();
                NonNumericColumnSpacesFeature = new List<ColumnStatistics>();
                StatsPerColumnPurpose = new Dictionary<string, Stats>();
                StatsPerColumnPurposeWithSpaces = new Dictionary<string, Stats>();
            }

            private void ApplyCore(ReadOnlyMemory<char>[][] data, Column column)
            {
                bool numericColumn = CmdParser.IsNumericType(column.Kind?.ToType());
                //Statistics for numeric column or length of the text in the case of non-numeric column.
                Stats stats = new Stats();
                //Statistics for number of spaces in the case non-numeric column.
                Stats spacesStats = new Stats();

                foreach (int index in column.Indices)
                {
                    if (index >= data.GetLength(0))
                        break;

                    foreach (ReadOnlyMemory<char> value in data[index])
                    {
                        string columnPurposeString = column.Purpose;
                        Stats statsPerPurpose;
                        Stats statsPerPurposeSpaces;
                        if (!StatsPerColumnPurpose.ContainsKey(columnPurposeString))
                        {
                            statsPerPurpose = new Stats();
                            statsPerPurposeSpaces = new Stats();
                            StatsPerColumnPurpose.Add(columnPurposeString, statsPerPurpose);
                            StatsPerColumnPurposeWithSpaces.Add(columnPurposeString, statsPerPurposeSpaces);
                        }
                        else
                        {
                            statsPerPurpose = StatsPerColumnPurpose[columnPurposeString];
                            statsPerPurposeSpaces = StatsPerColumnPurposeWithSpaces[columnPurposeString];
                        }

                        string valueString = value.ToString();
                        if (numericColumn)
                        {
                            double valueLocal;
                            if (Double.TryParse(valueString, out valueLocal))
                            {
                                stats.Add(valueLocal);
                                statsPerPurpose.Add(valueLocal);
                            }
                        }
                        else
                        {
                            stats.Add(valueString.Length);
                            statsPerPurpose.Add(valueString.Length);
                            int spacesCount = valueString.Count(c => c == ' ');
                            spacesStats.Add(spacesCount);
                            statsPerPurposeSpaces.Add(spacesCount);
                        }
                    }
                }

                if (numericColumn)
                    NumericColumnFeatures.Add(new ColumnStatistics { Column = column, Stats = stats });
                else
                {
                    NonNumericColumnLengthFeature.Add(new ColumnStatistics { Column = column, Stats = stats });
                    NonNumericColumnSpacesFeature.Add(new ColumnStatistics { Column = column, Stats = spacesStats });
                }
            }

            public void Apply(ReadOnlyMemory<char>[][] data, Column[] columns)
            {
                foreach (var column in columns)
                    ApplyCore(data, column);

                Contracts.Check(NonNumericColumnLengthFeature.Count == NonNumericColumnSpacesFeature.Count);
            }

            public bool AddMe() => NumericColumnFeatures.Count > 0 || NonNumericColumnLengthFeature.Count > 0;

            public string FeatureName() => nameof(ColumnFeatures);
        }

        private static IEnumerable<ITypeInferenceExpert> GetFeatures()
        {
            yield return new ColumnSchema();
            yield return new LabelFeatures();
            yield return new MissingValues();
            yield return new ColumnFeatures();
        }

        private struct DatasetFeatures
        {
            public long FileSize;
            public long? ApproximateRowCount;
            public bool InferencedSchema;
            public Guid Id;
            public Dictionary<string, ITypeInferenceExpert> Features;
        }

        public static string InferDatasetFeatures(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            string jsonString = "";
            if (!args.ApproximateRowCount.HasValue || args.ApproximateRowCount < 2)
                return jsonString;

            var h = env.Register("InferDatasetFeatureInference");
            using (var ch = h.Start("InferDatasetFeatureInference"))
            {
                DatasetFeatures features;
                features.FileSize = args.FullFileSize ?? 0;
                features.ApproximateRowCount = args.ApproximateRowCount;
                features.Features = new Dictionary<string, ITypeInferenceExpert>();
                features.Id = args.Id;
                features.InferencedSchema = args.InferencedSchema;

                foreach (var feature in GetFeatures())
                {
                    feature.Apply(args.Data, args.Columns);
                    if (feature.AddMe())
                        features.Features.Add(feature.FeatureName(), feature);
                }

                if (args.PrettyPrint)
                    jsonString = JsonConvert.SerializeObject(features, Newtonsoft.Json.Formatting.Indented);
                else
                    jsonString = JsonConvert.SerializeObject(features);
            }

            return jsonString;
        }
    }
}
