// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class InferenceUtils
    {
        public static IDataView Take(this IDataView data, int count)
        {
            Contracts.CheckValue(data, nameof(data));
            // REVIEW: This should take an env as a parameter, not create one.
            var env = new TlcEnvironment(0);
            var take = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeArguments { Count = count }, data);
            return CacheCore(take, env);
        }

        public static IDataView Cache(this IDataView data)
        {
            Contracts.CheckValue(data, nameof(data));
            // REVIEW: This should take an env as a parameter, not create one.
            return CacheCore(data, new TlcEnvironment(0));
        }

        private static IDataView CacheCore(IDataView data, IHostEnvironment env)
        {
            Contracts.AssertValue(data, "data");
            Contracts.AssertValue(env, "env");
            return new CacheDataView(env, data, Utils.GetIdentityPermutation(data.Schema.ColumnCount));
        }

        public static Type InferPredictorCategoryType(IDataView data, PurposeInference.Column[] columns)
        {
            List<PurposeInference.Column> labels = columns.Where(col => col.Purpose == ColumnPurpose.Label).ToList();
            if (labels.Count == 0)
                return typeof(SignatureClusteringTrainer);

            if (labels.Count > 1)
                return typeof(SignatureMultiOutputRegressorTrainer);

            PurposeInference.Column label = labels.First();
            HashSet<string> uniqueLabelValues = new HashSet<string>();
            data = data.Take(1000);
            using (var cursor = data.GetRowCursor(index => index == label.ColumnIndex))
            {
                ValueGetter<ReadOnlyMemory<char>> getter = DataViewUtils.PopulateGetterArray(cursor, new List<int> { label.ColumnIndex })[0];
                while (cursor.MoveNext())
                {
                    var currentLabel = default(ReadOnlyMemory<char>);
                    getter(ref currentLabel);
                    string currentLabelString = currentLabel.ToString();
                    if (!String.IsNullOrEmpty(currentLabelString) && !uniqueLabelValues.Contains(currentLabelString))
                        uniqueLabelValues.Add(currentLabelString);
                }
            }

            if (uniqueLabelValues.Count == 1)
                return typeof(SignatureAnomalyDetectorTrainer);

            if (uniqueLabelValues.Count == 2)
                return typeof(SignatureBinaryClassifierTrainer);

            if (uniqueLabelValues.Count > 2)
            {
                if ((label.ItemKind == DataKind.R4) &&
                    uniqueLabelValues.Any(val =>
                    {
                        float fVal;
                        return float.TryParse(val, out fVal) && (fVal > 50 || fVal < 0 || val.Contains('.'));
                    }))
                    return typeof(SignatureRegressorTrainer);

                if (label.ItemKind == DataKind.R4 ||
                    label.ItemKind == DataKind.TX ||
                    data.Schema.GetColumnType(label.ColumnIndex).IsKey)
                {
                    if (columns.Any(col => col.Purpose == ColumnPurpose.Group))
                        return typeof(SignatureRankerTrainer);
                    else
                        return typeof(SignatureMultiClassClassifierTrainer);
                }
            }

            return null;
        }

        public static ColumnGroupingInference.GroupingColumn[] InferColumnPurposes(IChannel ch, IHost env, TextFileSample sample, TextFileContents.ColumnSplitResult splitResult, out bool hasHeader)
        {
            ch.Info("Detecting column types");
            var typeInferenceResult = ColumnTypeInference.InferTextFileColumnTypes(env, sample,
                new ColumnTypeInference.Arguments
                {
                    ColumnCount = splitResult.ColumnCount,
                    Separator = splitResult.Separator,
                    AllowSparse = splitResult.AllowSparse,
                    AllowQuote = splitResult.AllowQuote,
                });

            hasHeader = typeInferenceResult.HasHeader;
            if (!typeInferenceResult.IsSuccess)
            {
                ch.Error("Couldn't detect column types.");
                return null;
            }

            ch.Info("Detecting column purposes");
            var typedLoaderArgs = new TextLoader.Arguments
            {
                Column = ColumnTypeInference.GenerateLoaderColumns(typeInferenceResult.Columns),
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote,
                HasHeader = typeInferenceResult.HasHeader
            };
            var typedLoader = new TextLoader(env, typedLoaderArgs, sample);

            var purposeInferenceResult = PurposeInference.InferPurposes(env, typedLoader,
                Utils.GetIdentityPermutation(typedLoaderArgs.Column.Length), new PurposeInference.Arguments());
            ch.Info("Detecting column grouping and generating column names");

            ColumnGroupingInference.GroupingColumn[] groupingResult = ColumnGroupingInference.InferGroupingAndNames(env, typeInferenceResult.HasHeader,
                typeInferenceResult.Columns, purposeInferenceResult.Columns).Columns;

            return groupingResult;
        }
    }

    // REVIEW: Should this also have the base type (ITrainer<...>)?
    public sealed class PredictorCategory
    {
        public readonly string Name;
        public readonly Type Signature;

        public PredictorCategory(string name, Type sig)
        {
            Name = name;
            Signature = sig;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    public sealed class PredictorsList
    {
        public static List<PredictorCategory> PredictorCategories = new List<PredictorCategory>
        {
           new PredictorCategory("Binary Classification", typeof(SignatureBinaryClassifierTrainer)),
           new PredictorCategory("Multi-class Classification", typeof(SignatureMultiClassClassifierTrainer)),
           new PredictorCategory("Ranking", typeof(SignatureRankerTrainer)),
           new PredictorCategory("Regression", typeof(SignatureRegressorTrainer)),
           new PredictorCategory("Multi-output Regression", typeof(SignatureMultiOutputRegressorTrainer)),
           new PredictorCategory("Anomaly Detection", typeof(SignatureAnomalyDetectorTrainer)),
           new PredictorCategory("Clustering", typeof(SignatureClusteringTrainer)),
           new PredictorCategory("Sequence Prediction", typeof(SignatureSequenceTrainer)),
           new PredictorCategory("Recommendation", typeof(SignatureMatrixRecommendingTrainer))
        };
    }

    public enum ColumnPurpose
    {
        Ignore = 0,
        Name = 1,
        Label = 2,
        NumericFeature = 3,
        CategoricalFeature = 4,
        TextFeature = 5,
        Weight = 6,
        Group = 7,
        ImagePath = 8
    }
}
