// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

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
        internal class IntermediateColumn
        {
            public readonly string ColumnName;
            public readonly ColumnType Type;
            public readonly ColumnPurpose Purpose;
            public readonly ColumnDimensions Dimensions;

            public IntermediateColumn(string name, ColumnType type, ColumnPurpose purpose, ColumnDimensions dimensions)
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
            IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns);
        }

        public abstract class TransformInferenceExpertBase : ITransformInferenceExpert
        {
            public abstract IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns);

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

            // For text labels, convert to categories.
            yield return new Experts.AutoLabel(context);

            // For group ID column, rename to GroupId and hash, if text.
            // REVIEW: this is only sufficient if we discard the possibility of hash collisions, and don't care
            // about the group Id cardinality (we don't for ranking).
            yield return new Experts.GroupIdHashRename(context);

            // For name column, rename to Name (or, if multiple and text, concat and rename to Name).
            yield return new Experts.NameColumnConcatRename(context);

            // For boolean columns use convert transform
            yield return new Experts.Boolean(context);

            // For categorical columns, use Cat transform.
            yield return new Experts.Categorical(context);

            // For text columns, use TextTransform.
            yield return new Experts.Text(context);

            // If numeric column has missing values, use Missing transform.
            yield return new Experts.NumericMissing(context);
        }

        internal static class Experts
        {
            internal sealed class AutoLabel : TransformInferenceExpertBase
            {
                public AutoLabel(MLContext context) : base(context)
                {
                }

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
                public GroupIdHashRename(MLContext context) : base(context)
                {
                }

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
                public Categorical(MLContext context) : base(context)
                {
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
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

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
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

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
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

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns)
                {
                    var columnsWithMissing = new List<string>();
                    foreach (var column in columns)
                    {
                        if (column.Type.GetItemType() == NumberType.R4
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
            
            internal sealed class NameColumnConcatRename : TransformInferenceExpertBase
            {
                public NameColumnConcatRename(MLContext context) : base(context)
                {
                }

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

                        if (column.Type.GetItemType().IsText())
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
        public static SuggestedTransform[] InferTransforms(MLContext context, (string, ColumnType, ColumnPurpose, ColumnDimensions)[] columns)
        {
            var intermediateCols = columns.Where(c => c.Item3 != ColumnPurpose.Ignore)
                .Select(c => new IntermediateColumn(c.Item1, c.Item2, c.Item3, c.Item4))
                .ToArray();

            var suggestedTransforms = new List<SuggestedTransform>();
            foreach (var expert in GetExperts(context))
            {
                SuggestedTransform[] suggestions = expert.Apply(intermediateCols).ToArray();
                suggestedTransforms.AddRange(suggestions);
            }

            var finalFeaturesConcatTransform = BuildFinalFeaturesConcatTransform(context, suggestedTransforms, intermediateCols);
            if (finalFeaturesConcatTransform != null)
            {
                suggestedTransforms.Add(finalFeaturesConcatTransform);
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
            foreach(var intermediateCol in intermediateCols)
            {
                if (intermediateCol.Purpose == ColumnPurpose.NumericFeature &&
                    intermediateCol.Type.GetItemType() == NumberType.R4)
                {
                    concatColNames.Add(intermediateCol.ColumnName);
                }
            }

            // remove 'Label' if it was ever a suggested purpose
            concatColNames.Remove(DefaultColumnNames.Label);
            concatColNames.Remove(DefaultColumnNames.GroupId);
            concatColNames.Remove(DefaultColumnNames.Name);

            intermediateCols = intermediateCols.Where(c => c.Purpose == ColumnPurpose.NumericFeature ||
                c.Purpose == ColumnPurpose.CategoricalFeature || c.Purpose == ColumnPurpose.TextFeature);

            if (!concatColNames.Any() || (concatColNames.Count == 1 &&
                concatColNames[0] == DefaultColumnNames.Features &&
                intermediateCols.First().Type.IsVector()))
            {
                return null;
            }

            if (concatColNames.Count() == 1 &&
                (intermediateCols.First().Type.IsVector() ||
                intermediateCols.First().Purpose == ColumnPurpose.CategoricalFeature ||
                intermediateCols.First().Purpose == ColumnPurpose.TextFeature))
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

                for(var i = 0; ; i++)
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
