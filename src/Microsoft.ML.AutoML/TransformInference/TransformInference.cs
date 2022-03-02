// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
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
        internal class IntermediateColumn
        {
            public readonly string ColumnName;
            public readonly DataViewType Type;
            public readonly ColumnPurpose Purpose;
            public readonly ColumnDimensions Dimensions;

            public IntermediateColumn(string name, DataViewType type, ColumnPurpose purpose, ColumnDimensions dimensions)
            {
                ColumnName = name;
                Type = type;
                Purpose = purpose;
                Dimensions = dimensions;
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
            IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task);
        }

        public abstract class TransformInferenceExpertBase : ITransformInferenceExpert
        {
            public abstract IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task);

            protected readonly MLContext Context;

            public TransformInferenceExpertBase(MLContext context)
            {
                Context = context;
            }
        }

        private static IEnumerable<ITransformInferenceExpert> GetExperts(MLContext context)
        {
            // The expert work independently of each other, the sequence is irrelevant
            // (it only determines the sequence of resulting transforms).

            // For multiclass tasks, convert label column to key
            yield return new Experts.Label(context);

            // For boolean columns use convert transform
            yield return new Experts.Boolean(context);

            // For categorical columns, use Cat transform.
            yield return new Experts.Categorical(context);

            // For text columns, use TextTransform.
            yield return new Experts.Text(context);

            // If numeric column has missing values, use Missing transform.
            yield return new Experts.NumericMissing(context);

            // For recommendation tasks, convert both user and item columns as key
            yield return new Experts.RecommendationColumns(context);

            // For image columns, use image transforms.
            yield return new Experts.Image(context);

            // For groupId columns, use GroupId transforms.
            yield return new Experts.GroupId(context);
        }

        internal static class Experts
        {
            internal sealed class Label : TransformInferenceExpertBase
            {
                public Label(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    if (task != TaskKind.MulticlassClassification)
                    {
                        yield break;
                    }

                    var lastLabelColId = Array.FindLastIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (lastLabelColId < 0)
                        yield break;

                    var col = columns[lastLabelColId];

                    if (!col.Type.IsKey())
                    {
                        yield return ValueToKeyMappingExtension.CreateSuggestedTransform(Context, col.ColumnName,
                            col.ColumnName);
                    }
                }
            }

            internal sealed class RecommendationColumns : TransformInferenceExpertBase
            {
                public RecommendationColumns(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    if (task != TaskKind.Recommendation)
                    {
                        yield break;
                    }
                    foreach (var column in columns)
                    {
                        if (column.Purpose == ColumnPurpose.UserId ||
                            column.Purpose == ColumnPurpose.ItemId)
                        {
                            yield return ValueToKeyMappingExtension.CreateSuggestedTransform(Context, column.ColumnName, column.ColumnName);
                        }
                    }
                }
            }

            internal sealed class GroupId : TransformInferenceExpertBase
            {
                public GroupId(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    if (task != TaskKind.Ranking)
                    {
                        yield break;
                    }
                    foreach (var column in columns)
                    {
                        if (column.Purpose == ColumnPurpose.GroupId && !column.Type.IsKey())
                        {
                            yield return HashingExtension.CreateSuggestedTransform(Context, column.ColumnName, column.ColumnName);
                        }
                    }
                }
            }

            internal sealed class Categorical : TransformInferenceExpertBase
            {
                public Categorical(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    bool foundCat = false;
                    bool foundCatHash = false;
                    var catColumnsNew = new List<string>();
                    var catHashColumnsNew = new List<string>();

                    foreach (var column in columns)
                    {
                        if (column.Purpose != ColumnPurpose.CategoricalFeature)
                        {
                            continue;
                        }

                        if (column.Dimensions.Cardinality != null && column.Dimensions.Cardinality < 100)
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

                    var transformedColumns = new List<string>();
                    transformedColumns.AddRange(catColumnsNew);
                    transformedColumns.AddRange(catHashColumnsNew);
                }
            }

            internal sealed class Boolean : TransformInferenceExpertBase
            {
                public Boolean(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    var newColumns = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.GetItemType().IsBool() || column.Purpose != ColumnPurpose.NumericFeature)
                        {
                            continue;
                        }

                        newColumns.Add(column.ColumnName);
                    }

                    if (newColumns.Count() > 0)
                    {
                        var newColumnsArr = newColumns.ToArray();
                        yield return TypeConvertingExtension.CreateSuggestedTransform(Context, newColumnsArr, newColumnsArr);
                    }
                }
            }

            internal sealed class Text : TransformInferenceExpertBase
            {
                public Text(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.GetItemType().IsText() || column.Purpose != ColumnPurpose.TextFeature)
                            continue;

                        var columnDestSuffix = "_tf";
                        var columnNameSafe = column.ColumnName;

                        string columnDestRenamed = $"{columnNameSafe}{columnDestSuffix}";

                        featureCols.Add(columnDestRenamed);
                        yield return TextFeaturizingExtension.CreateSuggestedTransform(Context, columnNameSafe, columnDestRenamed);
                    }
                }
            }

            internal sealed class NumericMissing : TransformInferenceExpertBase
            {
                public NumericMissing(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    var columnsWithMissing = new List<string>();
                    foreach (var column in columns)
                    {
                        if (column.Type.GetItemType() == NumberDataViewType.Single
                            && column.Purpose == ColumnPurpose.NumericFeature
                            && column.Dimensions.HasMissing == true)
                        {
                            columnsWithMissing.Add(column.ColumnName);
                        }
                    }
                    if (columnsWithMissing.Any())
                    {
                        var columnsArr = columnsWithMissing.ToArray();
                        var indicatorColNames = GetNewColumnNames(columnsArr.Select(c => $"{c}_MissingIndicator"), columns).ToArray();
                        yield return MissingValueIndicatingExtension.CreateSuggestedTransform(Context, columnsArr, indicatorColNames);
                        yield return TypeConvertingExtension.CreateSuggestedTransform(Context, indicatorColNames, indicatorColNames);
                        yield return MissingValueReplacingExtension.CreateSuggestedTransform(Context, columnsArr, columnsArr);
                    }
                }
            }
            internal sealed class Image : TransformInferenceExpertBase
            {
                public Image(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, TaskKind task)
                {
                    foreach (var column in columns)
                    {
                        if (!column.Type.GetItemType().IsText() || column.Purpose != ColumnPurpose.ImagePath)
                            continue;

                        var columnDestSuffix = "_featurized";
                        string columnDestRenamed = $"{column.ColumnName}{columnDestSuffix}";

                        yield return RawByteImageLoading.CreateSuggestedTransform(Context, column.ColumnName, columnDestRenamed);
                    }
                }
            }
        }

        /// <summary>
        /// Automatically infer transforms for the data view
        /// </summary>
        public static SuggestedTransform[] InferTransforms(MLContext context, TaskKind task, DatasetColumnInfo[] columns)
        {
            var intermediateCols = columns.Where(c => c.Purpose != ColumnPurpose.Ignore)
                .Select(c => new IntermediateColumn(c.Name, c.Type, c.Purpose, c.Dimensions))
                .ToArray();

            var suggestedTransforms = new List<SuggestedTransform>();
            foreach (var expert in GetExperts(context))
            {
                SuggestedTransform[] suggestions = expert.Apply(intermediateCols, task).ToArray();
                suggestedTransforms.AddRange(suggestions);
            }

            if (task != TaskKind.Recommendation)
            {
                var finalFeaturesConcatTransform = BuildFinalFeaturesConcatTransform(context, suggestedTransforms, intermediateCols);
                if (finalFeaturesConcatTransform != null)
                {
                    suggestedTransforms.Add(finalFeaturesConcatTransform);
                }
            }
            return suggestedTransforms.ToArray();
        }

        /// <summary>
        /// Build final features concat transform, using output of all suggested experts.
        /// Take the output columns from all suggested experts (except for 'Label'), and concatenate them
        /// into one final 'Features' column that a trainer will accept.
        /// </summary>
        private static SuggestedTransform BuildFinalFeaturesConcatTransform(MLContext context, IEnumerable<SuggestedTransform> suggestedTransforms,
            IEnumerable<IntermediateColumn> intermediateCols)
        {
            // get the output column names from all suggested transforms
            var concatColNames = new List<string>();
            foreach (var suggestedTransform in suggestedTransforms)
            {
                concatColNames.AddRange(suggestedTransform.PipelineNode.OutColumns);
            }

            // include all numeric columns of type R4
            foreach (var intermediateCol in intermediateCols)
            {
                if (intermediateCol.Purpose == ColumnPurpose.NumericFeature &&
                    intermediateCol.Type.GetItemType() == NumberDataViewType.Single)
                {
                    concatColNames.Add(intermediateCol.ColumnName);
                }
            }

            // remove column with 'Label' purpose
            var labelColumnName = intermediateCols.FirstOrDefault(c => c.Purpose == ColumnPurpose.Label)?.ColumnName;
            concatColNames.Remove(labelColumnName);

            // remove column with 'GroupId' purpose
            var groupColumnName = intermediateCols.FirstOrDefault(c => c.Purpose == ColumnPurpose.GroupId)?.ColumnName;
            concatColNames.RemoveAll(s => s == groupColumnName);

            intermediateCols = intermediateCols.Where(c => c.Purpose == ColumnPurpose.NumericFeature ||
                c.Purpose == ColumnPurpose.CategoricalFeature || c.Purpose == ColumnPurpose.TextFeature ||
                c.Purpose == ColumnPurpose.ImagePath);

            if (!concatColNames.Any() || (concatColNames.Count == 1 &&
                concatColNames[0] == DefaultColumnNames.Features &&
                intermediateCols.First().Type.IsVector()))
            {
                return null;
            }

            if (concatColNames.Count() == 1 &&
                (intermediateCols.First().Type.IsVector() ||
                intermediateCols.First().Purpose == ColumnPurpose.CategoricalFeature ||
                intermediateCols.First().Purpose == ColumnPurpose.TextFeature ||
                intermediateCols.First().Purpose == ColumnPurpose.ImagePath))
            {
                return ColumnCopyingExtension.CreateSuggestedTransform(context, concatColNames.First(), DefaultColumnNames.Features);
            }

            return ColumnConcatenatingExtension.CreateSuggestedTransform(context, concatColNames.Distinct().ToArray(), DefaultColumnNames.Features);
        }

        private static IEnumerable<string> GetNewColumnNames(IEnumerable<string> desiredColNames, IEnumerable<IntermediateColumn> columns)
        {
            var newColNames = new List<string>();

            var existingColNames = new HashSet<string>(columns.Select(c => c.ColumnName));
            foreach (var desiredColName in desiredColNames)
            {
                if (!existingColNames.Contains(desiredColName))
                {
                    newColNames.Add(desiredColName);
                    continue;
                }

                for (var i = 0; ; i++)
                {
                    var newColName = $"{desiredColName}{i}";
                    if (!existingColNames.Contains(newColName))
                    {
                        newColNames.Add(newColName);
                        break;
                    }
                }
            }

            return newColNames;
        }
    }
}
