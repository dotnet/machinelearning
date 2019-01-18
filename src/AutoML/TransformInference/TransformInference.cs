// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Auto
{
    internal class SuggestedTransform
    {
        public readonly IEstimator<ITransformer> Estimator;
        public readonly PipelineNode PipelineNode;

        public SuggestedTransform(PipelineNode pipelineNode, IEstimator<ITransformer> estimator)
        {
            PipelineNode = pipelineNode;
            Estimator = estimator;
        }

        public SuggestedTransform Clone()
        {
            return new SuggestedTransform(PipelineNode, Estimator);
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(PipelineNode.Name);
            sb.Append("{");
            if (PipelineNode.OutColumns.Length > 1)
            {
                for (var i = 0; i < PipelineNode.OutColumns.Length; i++)
                {
                    sb.Append($" col={PipelineNode.OutColumns[i]}:{PipelineNode.InColumns[i]}");
                }
            }
            else
            {
                sb.Append($" col={PipelineNode.OutColumns[0]}:{string.Join(",", PipelineNode.InColumns)}");
            }
            if (PipelineNode.Properties != null)
            {
                foreach (var property in PipelineNode.Properties)
                {
                    sb.Append($" {property.Key}={property.Value}");
                }
            }
            sb.Append("}");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Auto-generate set of transforms for the data view, given the purposes of specified columns.
    ///
    /// The design is the same as for <see cref="ColumnTypeInference"/>: there's a sequence of 'experts'
    /// that each look at all the columns. Every expert may or may not suggest additional transforms.
    /// If the expert needs some information about the column (for example, the column values),
    /// this information is lazily calculated by the column object, not the expert itself, to allow the reuse
    /// of the same information by another expert.
    /// </summary>
    internal static class TransformInference
    {
        private const double EstimatedSampleFraction = 1.0;
        private const bool ExcludeFeaturesConcatTransforms = false;

        private const int MaxRowsToRead = 1000;

        internal class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private readonly ColumnPurpose _purpose;
            private readonly Lazy<ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private readonly Lazy<bool> _hasMissing;

            public int ColumnId { get { return _columnId; } }
            public ColumnPurpose Purpose { get { return _purpose; } }
            public ColumnType Type { get { return _type.Value; } }
            public string ColumnName { get { return _columnName.Value; } }
            public bool HasMissing { get { return _hasMissing.Value; } }

            public IntermediateColumn(IDataView data, PurposeInference.Column column)
            {
                _data = data;
                _columnId = column.ColumnIndex;
                _purpose = column.Purpose;
                _type = new Lazy<ColumnType>(() => _data.Schema[_columnId].Type);
                _columnName = new Lazy<string>(() => _data.Schema[_columnId].Name);
                _hasMissing = new Lazy<bool>(() =>
                {
                    if (Type.ItemType() != NumberType.R4)
                        return false;
                    return Type.IsVector() ? HasMissingVector() : HasMissingOne();
                });
            }

            public string GetTempColumnName(string tag = null) => _data.Schema.GetTemporaryColumnName(tag);

            private bool HasMissingOne()
            {
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<Single>(_columnId);
                    var value = default(Single);
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        if (Single.IsNaN(value))
                            return true;
                    }
                    return false;
                }
            }

            private bool HasMissingVector()
            {
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<VBuffer<Single>>(_columnId);
                    var value = default(VBuffer<Single>);
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        if (VBufferUtils.HasNaNs(value))
                            return true;
                    }
                    return false;
                }
            }

            public void GetUniqueValueCounts<T>(out int uniqueValueCount, out int singletonCount, out int rowCount)
            {
                var seen = new HashSet<string>();
                var singletons = new HashSet<string>();
                rowCount = 0;
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<T>(_columnId);
                    while (cursor.MoveNext())
                    {
                        var value = default(T);
                        getter(ref value);
                        var s = value.ToString();
                        if (seen.Add(s))
                            singletons.Add(s);
                        else
                            singletons.Remove(s);
                        rowCount++;
                    }
                    uniqueValueCount = seen.Count;
                    singletonCount = singletons.Count;
                }
            }
        }

        internal sealed class ColumnRoutingStructure : IEquatable<ColumnRoutingStructure>
        {
            public struct AnnotatedName
            {
                public string Name { get; set; }
                public bool IsNumeric { get; set; }

                public bool Equals(AnnotatedName an)
                {
                    return an.Name == Name &&
                           an.IsNumeric == IsNumeric;
                }

                public override string ToString() => $"{Name}({IsNumeric})";
            }

            public AnnotatedName[] ColumnsConsumed { get; }
            public AnnotatedName[] ColumnsProduced { get; }

            public ColumnRoutingStructure(AnnotatedName[] columnsConsumed, AnnotatedName[] columnsProduced)
            {
                ColumnsConsumed = columnsConsumed;
                ColumnsProduced = columnsProduced;
            }

            public bool Equals(ColumnRoutingStructure obj)
            {
                return obj != null &&
                       obj.ColumnsConsumed.All(cc => ColumnsConsumed.Any(cc.Equals)) &&
                       obj.ColumnsProduced.All(cp => ColumnsProduced.Any(cp.Equals));
            }
        }

        internal interface ITransformInferenceExpert
        {
            bool IncludeFeaturesOverride { get; set; }

            IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns);
        }

        public abstract class TransformInferenceExpertBase : ITransformInferenceExpert
        {
            public bool IncludeFeaturesOverride { get; set; }

            public abstract IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns);

            protected readonly MLContext Context;

            public TransformInferenceExpertBase()
            {
                Context = new MLContext();
            }
        }

        private static IEnumerable<ITransformInferenceExpert> GetExperts()
        {
            // The expert work independently of each other, the sequence is irrelevant
            // (it only determines the sequence of resulting transforms).

            // For text labels, convert to categories.
            yield return new Experts.AutoLabel();

            // For group ID column, rename to GroupId and hash, if text.
            // REVIEW: this is only sufficient if we discard the possibility of hash collisions, and don't care
            // about the group Id cardinality (we don't for ranking).
            yield return new Experts.GroupIdHashRename();

            // For name column, rename to Name (or, if multiple and text, concat and rename to Name).
            yield return new Experts.NameColumnConcatRename();

            // For boolean columns use convert transform
            yield return new Experts.Boolean();

            // For categorical columns, use Cat transform.
            yield return new Experts.Categorical();

            // For text columns, use TextTransform.
            yield return new Experts.Text();

            // If numeric column has missing values, use Missing transform.
            yield return new Experts.NumericMissing();

            // If there's more than one feature column, concat all into Features. If it isn't called 'Features', rename it.
            yield return new Experts.FeaturesColumnConcatRenameNumericOnly();

            // For text columns, also use TextTransform with Unigram + trichar.
            //yield return new Experts.TextUniGramTriGram();
        }

        internal static class Experts
        {
            internal sealed class AutoLabel : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var lastLabelColId = Array.FindLastIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (lastLabelColId < 0)
                        yield break;

                    var col = columns[lastLabelColId];
                    
                    if (col.Type.IsText())
                    {
                        yield return ValueToKeyMappingExtension.CreateSuggestedTransform(Context, col.ColumnName, DefaultColumnNames.Label);
                    }
                    else if (col.ColumnName != DefaultColumnNames.Label)
                    {
                        yield return ColumnCopyingExtension.CreateSuggestedTransform(Context, col.ColumnName, DefaultColumnNames.Label);
                    }
                }
            }

            internal sealed class GroupIdHashRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var firstGroupColId = Array.FindIndex(columns, x => x.Purpose == ColumnPurpose.Group);
                    if (firstGroupColId < 0)
                        yield break;

                    var col = columns[firstGroupColId];

                    if (col.Type.IsText())
                    {
                        // REVIEW: we could potentially apply HashJoin to vectors of text.
                        yield return OneHotHashEncodingExtension.CreateSuggestedTransform(Context, col.ColumnName, DefaultColumnNames.GroupId);
                    }
                    else if (col.ColumnName != DefaultColumnNames.GroupId)
                    {
                        yield return ColumnCopyingExtension.CreateSuggestedTransform(Context, col.ColumnName, DefaultColumnNames.GroupId);
                    }
                }
            }

            internal sealed class Categorical : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    bool foundCat = false;
                    bool foundCatHash = false;
                    var catColumnsNew = new List<string>();
                    var catHashColumnsNew = new List<string>();
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsText() || column.Purpose != ColumnPurpose.CategoricalFeature)
                            continue;

                        if (IsDictionaryOk(column, EstimatedSampleFraction))
                        {
                            foundCat = true;
                            catColumnsNew.Add(column.ColumnName);
                        }
                        else
                        {
                            foundCatHash = true;
                            catHashColumnsNew.Add(column.ColumnName);
                        }
                    }

                    if (foundCat)
                    {
                        var catColumnsArr = catColumnsNew.ToArray();
                        yield return OneHotEncodingExtension.CreateSuggestedTransform(Context, catColumnsArr, catColumnsArr);
                    }

                    if (foundCatHash)
                    {
                        var catHashColumnsNewArr = catHashColumnsNew.ToArray();
                        yield return OneHotHashEncodingExtension.CreateSuggestedTransform(Context, catHashColumnsNewArr, catHashColumnsNewArr);
                    }

                    if (!ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }

                private bool IsDictionaryOk(IntermediateColumn column, Double dataSampleFraction)
                {
                    if (column.Type.IsVector())
                        return false;
                    int total;
                    int unique;
                    int singletons;
                    // REVIEW: replace with proper Good-Turing estimation.
                    // REVIEW: This looks correct; cf. equation (8) of Katz S. "Estimation of Probabilities from
                    // Sparse Data for the Language Model Component of a Speech Recognizer" (1987), taking into account that
                    // the singleton count was estimated from a fraction of the data (and assuming the estimate is
                    // roughly the same for the entire sample).
                    column.GetUniqueValueCounts<ReadOnlyMemory<char>>(out unique, out singletons, out total);
                    var expectedUnseenValues = singletons / dataSampleFraction;
                    return expectedUnseenValues < 1000 && unique < 10000;
                }
            }

            internal sealed class Boolean : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var newColumns = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsBool() || column.Purpose != ColumnPurpose.NumericFeature)
                        {
                            continue;
                        }

                        newColumns.Add(column.ColumnName);
                    }

                    if (newColumns.Count() > 0)
                    {
                        var newColumnsArr = newColumns.ToArray();
                        yield return TypeConvertingExtension.CreateSuggestedTransform(Context, newColumnsArr, newColumnsArr);

                        // Concat featurized columns into existing feature column, if transformed at least one column.
                        if (!ExcludeFeaturesConcatTransforms)
                        {
                            yield return InferenceHelpers.GetRemainingFeatures(newColumns, columns, IncludeFeaturesOverride);
                            IncludeFeaturesOverride = true;
                        }
                    }
                }
            }

            internal static class InferenceHelpers
            {
                public static SuggestedTransform GetRemainingFeatures(List<string> newCols, IntermediateColumn[] existingColumns,
                    bool includeFeaturesOverride)
                {
                    // Pick up existing features columns, if they exist
                    var featuresColumnsCount = existingColumns.Count(col =>
                     (col.Purpose == ColumnPurpose.NumericFeature) &&
                     (col.ColumnName == DefaultColumnNames.Features));
                    if (includeFeaturesOverride || featuresColumnsCount > 0)
                        newCols.Insert(0, DefaultColumnNames.Features);
                    return ColumnConcatenatingExtension.CreateSuggestedTransform(new MLContext(), newCols.ToArray(), DefaultColumnNames.Features);
                }

                public static SuggestedTransform TextTransformUnigramTriChar(MLContext context, string srcColumn, string dstColumn)
                {
                    //var input = new TextFeaturizingEstimator(context, srcColumn, dstColumn)
                    //{
                        //WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 1 },
                        //CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3 }
                    //};
                    return TextFeaturizingExtension.CreateSuggestedTransform(context, srcColumn, dstColumn);
                }

                public static SuggestedTransform TextTransformBigramTriChar(MLContext context, string srcColumn, string dstColumn)
                {
                    //var input = new TextFeaturizingEstimator(env, srcColumn, dstColumn)
                    //{
                        //WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2 },
                        //CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3 }
                    //};
                    return TextFeaturizingExtension.CreateSuggestedTransform(context, srcColumn, dstColumn);
                }
            }

            internal sealed class Text : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsText() || column.Purpose != ColumnPurpose.TextFeature)
                            continue;

                        var columnDestSuffix = "_tf";
                        var columnNameSafe = column.ColumnName;

                        string columnDestRenamed = $"{columnNameSafe}{columnDestSuffix}";

                        featureCols.Add(columnDestRenamed);
                        yield return TextFeaturizingExtension.CreateSuggestedTransform(Context, columnNameSafe, columnDestRenamed);
                    }

                    // Concat text featurized columns into existing feature column, if transformed at least one column.
                    if (!ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }
            }

            internal sealed class TextUniGramTriGram : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType().IsText() && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName).ToArray();

                    if ((textColumnNames.Length == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Length > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcat");
                        yield return ColumnConcatenatingExtension.CreateSuggestedTransform(Context, textColumnNames, concatTextColumnName);
                    }
                    else
                    {
                        concatTextColumnName = textColumnNames.First();
                    }

                    //Get Unigram + Trichar for text transform on the concatenated text column.
                    string featureTextColumn = columns[0].GetTempColumnName("FeaturesText");
                    yield return InferenceHelpers.TextTransformUnigramTriChar(Context, concatTextColumnName, featureTextColumn);

                    //Concat text featurized column into feature column.
                    List<string> featureCols = new List<string>(new[] { featureTextColumn });
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        featureCols.Add(DefaultColumnNames.Features);

                    if (!ExcludeFeaturesConcatTransforms)
                    {
                        yield return ColumnConcatenatingExtension.CreateSuggestedTransform(Context, featureCols.ToArray(), DefaultColumnNames.Features);
                    }
                }
            }

            internal sealed class NumericMissing : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var columnsWithMissing = new List<string>();
                    foreach (var column in columns)
                    {
                        if (column.Type.ItemType() != NumberType.R4 || column.Purpose != ColumnPurpose.NumericFeature
                            || !column.HasMissing)
                        {
                            continue;
                        }

                        columnsWithMissing.Add(column.ColumnName);
                    }
                    if (columnsWithMissing.Any())
                    {
                        var columnsArr = columnsWithMissing.ToArray();
                        yield return MissingValueIndicatorExtension.CreateSuggestedTransform(Context, columnsArr, columnsArr);
                    }
                }
            }

            internal class FeaturesColumnConcatRename : TransformInferenceExpertBase
            {
                public virtual bool IgnoreColumn(ColumnPurpose purpose)
                {
                    if (purpose != ColumnPurpose.TextFeature
                        && purpose != ColumnPurpose.CategoricalFeature
                        && purpose != ColumnPurpose.NumericFeature)
                        return true;
                    return false;
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var selectedColumns = columns.Where(c => !IgnoreColumn(c.Purpose)).ToArray();
                    var colList = selectedColumns.Select(c => c.ColumnName).ToArray();
                    bool allColumnsNumeric = selectedColumns.All(c => c.Purpose == ColumnPurpose.NumericFeature && c.Type.ItemType() != BoolType.Instance);
                    bool allColumnsNonNumeric = selectedColumns.All(c => c.Purpose != ColumnPurpose.NumericFeature);

                    if (colList.Length > 0)
                    {
                        // Check if column is named features and already numeric
                        if (colList.Length == 1 && colList[0] == DefaultColumnNames.Features && allColumnsNumeric)
                        {
                            yield break;
                        }
                        
                        if (!allColumnsNumeric && !allColumnsNonNumeric)
                        {
                            yield break;
                        }
                        
                        var input = new ColumnConcatenatingEstimator(Context, DefaultColumnNames.Features, colList);
                        yield return ColumnConcatenatingExtension.CreateSuggestedTransform(Context, colList, DefaultColumnNames.Features);
                    }
                }
            }

            internal sealed class FeaturesColumnConcatRenameIgnoreText : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.CategoricalFeature && purpose != ColumnPurpose.NumericFeature);
                }
            }

            internal sealed class FeaturesColumnConcatRenameNumericOnly : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.NumericFeature);
                }
            }

            internal sealed class NameColumnConcatRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    int count = 0;
                    var colSpec = new StringBuilder();
                    var colSpecTextOnly = new List<string>();
                    var columnList = new List<string>();

                    foreach (var column in columns)
                    {
                        var columnName = new StringBuilder();
                        if (column.Purpose != ColumnPurpose.Name)
                        {
                            continue;
                        }
                        count++;

                        if (colSpec.Length > 0)
                        {
                            colSpec.Append(",");
                        }
                        colSpec.Append(column.ColumnName);
                        
                        columnName.Append(column.ColumnName);
                        columnList.Add(columnName.ToString());

                        if (column.Type.ItemType().IsText())
                        {
                            colSpecTextOnly.Add(column.ColumnName);
                        }
                    }

                    if (count == 1 && colSpec.ToString() != DefaultColumnNames.Name)
                    {
                        yield return ColumnCopyingExtension.CreateSuggestedTransform(Context, colSpec.ToString(), DefaultColumnNames.Name);
                    }
                    else if (count > 1)
                    {
                        if (string.IsNullOrWhiteSpace(colSpecTextOnly.ToString()))
                        {
                            yield break;
                        }

                        // suggested grouping name columns into one vector
                        yield return ColumnConcatenatingExtension.CreateSuggestedTransform(Context, columnList.ToArray(), DefaultColumnNames.Name);
                    }
                }
            }
        }

        /// <summary>
        /// Automatically infer transforms for the data view
        /// </summary>
        public static SuggestedTransform[] InferTransforms(MLContext env, IDataView data, PurposeInference.Column[] purposes)
        {
            data = data.Take(MaxRowsToRead);
            var cols = purposes.Where(x => !data.Schema[x.ColumnIndex].IsHidden).Select(x => new IntermediateColumn(data, x)).ToArray();
            var list = new List<SuggestedTransform>();
            var includeFeaturesOverride = false;
            foreach (var expert in GetExperts())
            {
                expert.IncludeFeaturesOverride = includeFeaturesOverride;
                SuggestedTransform[] suggestions = expert.Apply(cols).ToArray();
                includeFeaturesOverride |= expert.IncludeFeaturesOverride;

                list.AddRange(suggestions);
            }
            return list.ToArray();
        }
    }
}
