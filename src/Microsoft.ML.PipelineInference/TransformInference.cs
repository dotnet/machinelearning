// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Auto-generate set of transforms for the data view, given the purposes of specified columns.
    ///
    /// The design is the same as for <see cref="ColumnTypeInference"/>: there's a sequence of 'experts'
    /// that each look at all the columns. Every expert may or may not suggest additional transforms.
    /// If the expert needs some information about the column (for example, the column values),
    /// this information is lazily calculated by the column object, not the expert itself, to allow the reuse
    /// of the same information by another expert.
    /// </summary>
    public static class TransformInference
    {
        public sealed class Arguments
        {
            /// <summary>
            /// Relative size of the inspected data view vs. the 'real' data size.
            /// </summary>
            public Double EstimatedSampleFraction;
            public bool ExcludeFeaturesConcatTransforms;
            public int[] ExcludedColumnIndices;
            public int AtomicIdOffset;
            public int Level;

            public Arguments()
            {
                EstimatedSampleFraction = 1;
                ExcludeFeaturesConcatTransforms = false;
                ExcludedColumnIndices = new int[] { };
                AtomicIdOffset = 0;
                Level = 0;
            }
        }

        private const int MaxRowsToRead = 1000;

        public struct SuggestedTransform : IEquatable<SuggestedTransform>
        {
            public readonly string Description;
            public readonly TransformString Transform;
            // Indicates the type of the transform. This is used by the recipe to leave/take transform.
            public readonly Type ExpertType;
            public TransformPipelineNode PipelineNode;
            // Used for grouping transforms that must occur together
            public int AtomicGroupId { get; set; }
            // Stores which columns are consumed by this transform,
            // and which are produced, at which level.
            public ColumnRoutingStructure RoutingStructure { get; set; }
            public bool AlwaysInclude { get; set; }

            public SuggestedTransform(string description,
                TransformString transform, Type expertType,
                TransformPipelineNode pipelineNode = null, int atomicGroupId = -1,
                ColumnRoutingStructure routingStructure = null, bool alwaysInclude = false)
            {
                Description = description;
                Transform = transform;
                ExpertType = expertType;
                PipelineNode = pipelineNode;
                AtomicGroupId = atomicGroupId;
                RoutingStructure = routingStructure;
                AlwaysInclude = alwaysInclude;
            }

            public bool Equals(SuggestedTransform transform)
            {
                return ExpertType == transform.ExpertType &&
                       Transform.Equals(transform.Transform) &&
                       AtomicGroupId == transform.AtomicGroupId &&
                       PipelineNode.ToString() == transform.PipelineNode.ToString() &&
                       RoutingStructure.Equals(transform.RoutingStructure);
            }

            public SuggestedTransform Clone()
            {
                return new SuggestedTransform(Description, Transform, ExpertType,
                    PipelineNode.Clone(), AtomicGroupId, RoutingStructure, AlwaysInclude);
            }

            public override string ToString() => ExpertType.Name;
        }

        public struct TransformString : IEquatable<TransformString>
        {
            public readonly string Kind;
            public readonly string Settings;

            public TransformString(string kind, string settings)
            {
                Kind = kind ?? "";
                Settings = settings ?? "";
            }

            public bool Equals(TransformString other)
            {
                return Kind == other.Kind &&
                    Settings == other.Settings;
            }

            public override string ToString()
            {
                if (Settings.Length == 0)
                    return Kind;

                StringBuilder sb = new StringBuilder();
                CmdQuoter.QuoteValue(Settings, sb, true);
                return Kind + sb.ToString();
            }
        }

        public struct InferenceResult
        {
            public readonly SuggestedTransform[] SuggestedTransforms;

            public InferenceResult(SuggestedTransform[] suggestedTransforms)
            {
                SuggestedTransforms = suggestedTransforms;
            }
        }

        public struct Column
        {
            public readonly Data.ColumnType Type;
            public readonly string Name;
            public readonly ColumnPurpose Purpose;

            public Column(Data.ColumnType type, string name, ColumnPurpose purpose)
            {
                Type = type;
                Name = name;
                Purpose = purpose;
            }
        }

        public class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private readonly ColumnPurpose _purpose;
            private readonly Lazy<Data.ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private readonly Lazy<bool> _hasMissing;

            public int ColumnId { get { return _columnId; } }
            public ColumnPurpose Purpose { get { return _purpose; } }
            public Data.ColumnType Type { get { return _type.Value; } }
            public string ColumnName { get { return _columnName.Value; } }
            public bool HasMissing { get { return _hasMissing.Value; } }

            public IntermediateColumn(IDataView data, PurposeInference.Column column)
            {
                _data = data;
                _columnId = column.ColumnIndex;
                _purpose = column.Purpose;
                _type = new Lazy<Data.ColumnType>(() => _data.Schema.GetColumnType(_columnId));
                _columnName = new Lazy<string>(() => _data.Schema.GetColumnName(_columnId));
                _hasMissing = new Lazy<bool>(() =>
                {
                    if (Type.ItemType != NumberType.R4)
                        return false;
                    return Type.IsVector ? HasMissingVector() : HasMissingOne();
                });
            }

            public string GetTempColumnName(string tag = null) => _data.Schema.GetTempColumnName(tag);

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
                        if (value.Count > 0 && value.Values.Any(Single.IsNaN))
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

        public sealed class ColumnRoutingStructure : IEquatable<ColumnRoutingStructure>
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
            public int Level { get; set; }

            public ColumnRoutingStructure(AnnotatedName[] columnsConsumed, AnnotatedName[] columnsProduced, int level = 0)
            {
                ColumnsConsumed = columnsConsumed;
                ColumnsProduced = columnsProduced;
                Level = level;
            }

            public bool Equals(ColumnRoutingStructure obj)
            {
                return obj != null &&
                       obj.Level == Level &&
                       obj.ColumnsConsumed.All(cc => ColumnsConsumed.Any(cc.Equals)) &&
                       obj.ColumnsProduced.All(cp => ColumnsProduced.Any(cp.Equals));
            }
        }

        public interface ITransformInferenceExpert
        {
            bool IncludeFeaturesOverride { get; set; }

            IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch);
        }

        public abstract class TransformInferenceExpertBase : ITransformInferenceExpert
        {
            public bool IncludeFeaturesOverride { get; set; }

            public abstract IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch);
        }

        private static IEnumerable<ITransformInferenceExpert> GetExperts(bool excludeFeaturesRename)
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

            // Check cardinality and type for numeric labels.
            yield return new Experts.LabelAdvisory();

            // For boolean columns use convert transform
            yield return new Experts.Boolean();

            // For categorical columns, use Cat transform.
            yield return new Experts.Categorical();

            // For text columns, use TextTransform.
            yield return new Experts.Text();

            // If numeric column has missing values, use Missing transform.
            yield return new Experts.NumericMissing();

            // If there's more than one feature column, concat all into Features. If it isn't called 'Features', rename it.
            if (!excludeFeaturesRename)
            {
                yield return new Experts.FeaturesColumnConcatRenameNumericOnly();
            }

            // For text columns, also use TextTransform with Unigram + trichar.
            yield return new Experts.TextUniGramTriGram();

            // For text columns, also use TextTransform with Bigram + trichar.
            yield return new Experts.TextBiGramTriGram();

            //For text columns, also use SDCA based transform.
            yield return new Experts.SdcaTransform();

            //For text columns, also use tree leaf based transform.
            yield return new Experts.NaiveBayesTransform();
        }

        public static class Experts
        {
            public sealed class AutoLabel : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var lastLabelColId = Array.FindLastIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (lastLabelColId < 0)
                        yield break;

                    var col = columns[lastLabelColId];

                    var columnArgument = new StringBuilder();
                    var columnNameQuoted = new StringBuilder();
                    columnArgument.Append("col=");
                    if (CmdQuoter.NeedsQuoting(col.ColumnName))
                    {
                        columnArgument.AppendFormat("{{name={0} src=", DefaultColumnNames.Label);
                        CmdQuoter.QuoteValue(col.ColumnName, columnArgument);
                        CmdQuoter.QuoteValue(col.ColumnName, columnNameQuoted);
                        columnArgument.Append("}");
                    }
                    else
                    {
                        columnArgument.AppendFormat("{0}:{1}", DefaultColumnNames.Label, col.ColumnName);
                        columnNameQuoted.Append(col.ColumnName);
                    }

                    if (col.Type.IsText)
                    {
                        col.GetUniqueValueCounts<ReadOnlyMemory<char>>(out var unique, out var _, out var _);
                        ch.Info("Label column '{0}' is text. Suggested auto-labeling.", col.ColumnName);

                        var args = new TransformString("AutoLabel", columnArgument.ToString());
                        string dest = DefaultColumnNames.Label;
                        string source = columnNameQuoted.ToString();

                        var epInput = new ML.Legacy.Transforms.TextToKeyConverter();
                        epInput.Column = new[]
                        {
                            new ML.Legacy.Transforms.TermTransformColumn
                            {
                                Name = dest,
                                Source = source
                            }
                        };

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = false, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );
                        yield return new SuggestedTransform("Convert text labels to keys", args, GetType(), new TransformPipelineNode(epInput), -1, routingStructure, true);

                        if (unique == 1)
                        {
                            ch.Warning("Label column '{0}' has only one value in the sample. Maybe the label column is incorrect?", col.ColumnName);
                        }
                        else if (unique > 100)
                        {
                            ch.Warning("Label column '{0}' has {1} different values in the sample. Multi-class classification might not be desirable with so many values.", col.ColumnName, unique);
                        }
                        else if (unique > 2)
                        {
                            ch.Info("Label column '{0}' has {1} different values in the sample.", col.ColumnName, unique);
                        }
                    }
                    else if (col.ColumnName != DefaultColumnNames.Label)
                    {
                        string dest = DefaultColumnNames.Label;
                        string source = columnNameQuoted.ToString();
                        var args = new TransformString("Copy", columnArgument.ToString());
                        var epInput = new ML.Legacy.Transforms.ColumnCopier
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.CopyColumnsTransformColumn
                                {
                                    Name = dest,
                                    Source = source
                                }
                            }
                        };

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        yield return new SuggestedTransform($"Rename label column to '{DefaultColumnNames.Label}'", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure, true);
                    }
                }
            }

            public sealed class GroupIdHashRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var firstGroupColId = Array.FindIndex(columns, x => x.Purpose == ColumnPurpose.Group);
                    if (firstGroupColId < 0)
                        yield break;

                    var col = columns[firstGroupColId];

                    var columnArgument = new StringBuilder();
                    var columnNameQuoted = new StringBuilder();
                    columnArgument.Append("col=");
                    if (CmdQuoter.NeedsQuoting(col.ColumnName))
                    {
                        columnArgument.AppendFormat("{name={0} src=", DefaultColumnNames.GroupId);
                        CmdQuoter.QuoteValue(col.ColumnName, columnArgument);
                        CmdQuoter.QuoteValue(col.ColumnName, columnNameQuoted);
                        columnArgument.Append("}");
                    }
                    else
                    {
                        columnArgument.AppendFormat("{0}:{1}", DefaultColumnNames.GroupId, col.ColumnName);
                        columnNameQuoted.AppendFormat("{0}", col.ColumnName);
                    }

                    if (col.Type.IsText)
                    {
                        ch.Info("Group Id column '{0}' is text. Suggested hashing.", col.ColumnName);
                        // REVIEW: we could potentially apply HashJoin to vectors of text.
                        var args = new TransformString("Hash", columnArgument.ToString());
                        string dest = DefaultColumnNames.GroupId;
                        string source = columnNameQuoted.ToString();
                        var epInput = new ML.Legacy.Transforms.CategoricalHashOneHotVectorizer
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.CategoricalHashTransformColumn
                                {
                                    Name = dest,
                                    Source = source
                                }
                            }
                        };

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = false, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        string[] outputColNames = new string[] { DefaultColumnNames.GroupId };
                        yield return new SuggestedTransform("Hash text group IDs.", args, GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }
                    else if (col.ColumnName != DefaultColumnNames.GroupId)
                    {
                        ch.Warning("Group Id column '{0}' is not text. Couldn't determine correct transformation.");
                        var args = new TransformString("Copy", columnArgument.ToString());
                        string dest = DefaultColumnNames.GroupId;
                        string source = columnNameQuoted.ToString();

                        var epInput = new ML.Legacy.Transforms.ColumnCopier
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.CopyColumnsTransformColumn
                                {
                                    Name = dest,
                                    Source = source
                                }
                            }
                        };

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        yield return new SuggestedTransform($"Rename group ID column to '{DefaultColumnNames.GroupId}'", args, GetType(), new TransformPipelineNode(epInput),
                            -1, routingStructure);
                    }
                }
            }

            public sealed class LabelAdvisory : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var firstLabelColId = Array.FindIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (firstLabelColId < 0)
                    {
                        ch.Info("Label column not present in the dataset. This is likely an unsupervised learning problem.");
                        yield break;
                    }

                    var col = columns[firstLabelColId];
                    if (col.Type.IsText)
                        yield break;

                    if (col.Type.IsKnownSizeVector && col.Type.ItemType == NumberType.R4)
                    {
                        ch.Info("Label column '{0}' has type {1}, this can be only a multi-output regression problem.", col.ColumnName, col.Type);
                        yield break;
                    }

                    if (col.Type != NumberType.R4)
                    {
                        ch.Warning("Label column '{0}' has type {1} which is not supported by any learner.", col.ColumnName, col.Type);
                        yield break;
                    }

                    int unique;
                    int singleton;
                    int total;
                    col.GetUniqueValueCounts<Single>(out unique, out singleton, out total);

                    if (unique == 1)
                    {
                        ch.Warning("Label column '{0}' has only one value in the sample. Maybe the label column is incorrect?", col.ColumnName);
                    }
                    else if (unique > 100)
                    {
                        ch.Info("Label column '{0}' has {1} different values in the sample. This is likely a regression problem.", col.ColumnName, unique);
                    }
                    else if (unique > 2)
                    {
                        ch.Info("Label column '{0}' has {1} different values in the sample. This can be treated as multi-class or regression problem.", col.ColumnName, unique);
                    }
                }
            }

            public sealed class Categorical : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    bool foundCat = false;
                    bool foundCatHash = false;
                    var colSpecCat = new StringBuilder();
                    var colSpecCatHash = new StringBuilder();
                    var catColumns = new List<ML.Legacy.Transforms.CategoricalTransformColumn>();
                    var catHashColumns = new List<ML.Legacy.Transforms.CategoricalHashTransformColumn>();
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType.IsText || column.Purpose != ColumnPurpose.CategoricalFeature)
                            continue;

                        var columnArgument = new StringBuilder();
                        var columnNameQuoted = new StringBuilder();
                        columnArgument.Append("col=");
                        if (CmdQuoter.NeedsQuoting(column.ColumnName))
                        {
                            columnArgument.Append("{name=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append(" src=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append("} ");

                            CmdQuoter.QuoteValue(column.ColumnName, columnNameQuoted);
                        }
                        else
                        {
                            columnArgument.AppendFormat("{0} ", column.ColumnName);
                            columnNameQuoted.AppendFormat("{0}", column.ColumnName);
                        }

                        if (IsDictionaryOk(column, inferenceArgs.EstimatedSampleFraction))
                        {
                            foundCat = true;
                            colSpecCat.Append(columnArgument);
                            catColumns.Add(new ML.Legacy.Transforms.CategoricalTransformColumn
                            {
                                Name = columnNameQuoted.ToString(),
                                Source = columnNameQuoted.ToString()
                            });
                        }
                        else
                        {
                            ch.Info("Categorical column '{0}' has extremely high cardinality. Suggested hash-based category encoding.", column.ColumnName);
                            foundCatHash = true;
                            colSpecCatHash.Append(columnArgument);
                            catHashColumns.Add(new ML.Legacy.Transforms.CategoricalHashTransformColumn
                            {
                                Name = columnNameQuoted.ToString(),
                                Source = columnNameQuoted.ToString()
                            });
                        }
                    }

                    if (foundCat)
                    {
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            catColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.Name }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            catColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.Name }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                        var epInput = new ML.Legacy.Transforms.CategoricalOneHotVectorizer { Column = catColumns.ToArray() };
                        featureCols.AddRange(catColumns.Select(c => c.Name));

                        ch.Info("Suggested dictionary-based category encoding for categorical columns.");
                        var args = new TransformString("Cat", colSpecCat.ToString());
                        yield return new SuggestedTransform("Convert categorical features to indicator vectors", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }

                    if (foundCatHash)
                    {
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            catHashColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.Name }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            catHashColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.Name }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                        var epInput = new ML.Legacy.Transforms.CategoricalHashOneHotVectorizer { Column = catHashColumns.ToArray() };
                        featureCols.AddRange(catColumns.Select(c => c.Name));

                        ch.Info("Suggested hash-based category encoding for categorical columns.");
                        var args = new TransformString("CatHash", colSpecCatHash.ToString());
                        yield return new SuggestedTransform("Hash categorical features and convert to indicator vectors", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }

                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, GetType(), IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }

                private bool IsDictionaryOk(IntermediateColumn column, Double dataSampleFraction)
                {
                    if (column.Type.IsVector)
                        return false;
                    Contracts.Assert(dataSampleFraction > 0);
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

            public sealed class Boolean : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var columnArgument = new StringBuilder();
                    var columnNameQuoted = new StringBuilder();
                    var epColumns = new List<ML.Legacy.Transforms.ConvertTransformColumn>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType.IsBool || column.Purpose != ColumnPurpose.NumericFeature)
                            continue;
                        columnArgument.Append("col=");
                        if (CmdQuoter.NeedsQuoting(column.ColumnName))
                        {
                            columnArgument.Append("{name=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append(" src=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append("} ");

                            CmdQuoter.QuoteValue(column.ColumnName, columnNameQuoted);
                        }
                        else
                        {
                            columnArgument.AppendFormat("{0} ", column.ColumnName);
                            columnNameQuoted.AppendFormat("{0}", column.ColumnName);
                        }

                        epColumns.Add(new ML.Legacy.Transforms.ConvertTransformColumn
                        {
                            Name = columnNameQuoted.ToString(),
                            Source = columnNameQuoted.ToString(),
                            ResultType = ML.Legacy.Data.DataKind.R4
                        });
                    }

                    if (columnArgument.Length > 0)
                    {
                        ch.Info("Suggested conversion to numeric for boolean features.");
                        var args = new TransformString("Convert", $"{columnArgument}type=R4");
                        var epInput = new ML.Legacy.Transforms.ColumnTypeConverter { Column = epColumns.ToArray(), ResultType = ML.Legacy.Data.DataKind.R4 };
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            epColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.Name }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            epColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.Name }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform("Convert boolean features to numeric", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);

                        // Concat featurized columns into existing feature column, if transformed at least one column.
                        if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        {
                            yield return InferenceHelpers.GetRemainingFeatures(epColumns.Select(c => c.Name).ToList(),
                                columns, GetType(), IncludeFeaturesOverride);
                            IncludeFeaturesOverride = true;
                        }
                    }
                }
            }

            internal static class InferenceHelpers
            {
                public static SuggestedTransform GetRemainingFeatures(List<string> newCols, IntermediateColumn[] existingColumns,
                    Type currentType, bool includeFeaturesOverride)
                {
                    // Pick up existing features columns, if they exist
                    var featuresColumnsCount = existingColumns.Count(col =>
                     (col.Purpose == ColumnPurpose.NumericFeature) &&
                     (col.ColumnName == DefaultColumnNames.Features));
                    if (includeFeaturesOverride || featuresColumnsCount > 0)
                        newCols.Insert(0, DefaultColumnNames.Features);
                    return InferenceHelpers.ConcatColumnsIntoOne(newCols, DefaultColumnNames.Features, currentType, true);
                }

                public static StringBuilder GetTextTransformArgument(string columnName, string tempColumn = null)
                {
                    var columnArgument = new StringBuilder();
                    columnArgument.Append("col=");
                    if (CmdQuoter.NeedsQuoting(columnName))
                    {
                        StringBuilder quotedColumnName = new StringBuilder();
                        CmdQuoter.QuoteValue(columnName, quotedColumnName);
                        columnArgument.AppendFormat("{name={0} src={0}}", quotedColumnName.ToString());
                    }
                    else
                        columnArgument.AppendFormat("{0} ",
                            tempColumn != null ? (tempColumn + ":" + columnName) : columnName);

                    return columnArgument;
                }

                public static StringBuilder GetTextTransformUnigramTriCharArgument(string columnName,
                    string tempColumn = null)
                {
                    StringBuilder columnArgument = GetTextTransformArgument(columnName, tempColumn);
                    columnArgument.Append(
                        " wordExtractor=NgramExtractorTransform charExtractor=NgramExtractorTransform{ngram=3}");

                    return columnArgument;
                }

                public static StringBuilder GetTextTransformBigramTriCharArgument(string columnName,
                    string tempColumn = null)
                {
                    StringBuilder columnArgument = GetTextTransformArgument(columnName, tempColumn);
                    columnArgument.Append(
                        " wordExtractor=NgramExtractorTransform{ngram=2} charExtractor=NgramExtractorTransform{ngram=3}");

                    return columnArgument;
                }

                public static SuggestedTransform ConcatColumnsIntoOne(List<string> columnNames, string concatColumnName,
                    Type transformType, bool isNumeric)
                {
                    StringBuilder columnArgument = new StringBuilder();
                    StringBuilder columnNameQuoted = new StringBuilder();
                    bool needQuoting = false;

                    columnNames.ForEach(column =>
                    {
                        columnArgument.Append("src=");
                        if (CmdQuoter.NeedsQuoting(column))
                        {
                            needQuoting = true;
                            CmdQuoter.QuoteValue(column, columnArgument);
                            CmdQuoter.QuoteValue(column, columnNameQuoted);
                            columnArgument.Append(" ");
                        }
                        else
                        {
                            columnArgument.AppendFormat("{0} ", column);
                            columnNameQuoted.AppendFormat("{0}", column);
                        }
                    });

                    string columnsToConcat = string.Join(",", columnNames);
                    string arguments = needQuoting
                        ? $"col={{name={concatColumnName} {columnArgument}}}"
                        : $"col={concatColumnName}:{columnsToConcat}";

                    var epInput = new ML.Legacy.Transforms.ColumnConcatenator
                    {
                        Column = new[]
                            {
                                new ML.Legacy.Transforms.ConcatTransformColumn
                                {
                                    Name = concatColumnName,
                                    Source = columnNames.ToArray()
                                }
                            }
                    };

                    // Not sure if resulting columns will be numeric or text, since concat can apply to either.
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        columnNames.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = isNumeric, Name = c }).ToArray();
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        new[] { new ColumnRoutingStructure.AnnotatedName { IsNumeric = isNumeric, Name = concatColumnName } };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                    return
                        new SuggestedTransform(
                            $"Concatenate {columnsToConcat} columns into column {concatColumnName}",
                            new TransformString("Concat", arguments),
                            transformType,
                            new TransformPipelineNode(epInput),
                            -1,
                            routingStructure);
                }

                public static SuggestedTransform TextTransformUnigramTriChar(string srcColumn, string dstColumn, string arg, Type transformType)
                {
                    StringBuilder columnArgument = InferenceHelpers.GetTextTransformUnigramTriCharArgument(srcColumn, dstColumn);

                    columnArgument.Append(arg);

                    var nodeInput = new ML.Legacy.Transforms.TextFeaturizer();
                    nodeInput.WordFeatureExtractor = new NGramNgramExtractor { NgramLength = 1 };
                    nodeInput.CharFeatureExtractor = new NGramNgramExtractor { NgramLength = 3 };
                    nodeInput.Column = new ML.Legacy.Transforms.TextTransformColumn
                    {
                        Name = dstColumn,
                        Source = new[] { srcColumn }
                    };

                    return TextTransform(srcColumn, dstColumn, columnArgument.ToString(), "Unigram plus Trichar",
                        transformType, new TransformPipelineNode(nodeInput));
                }

                public static SuggestedTransform TextTransformBigramTriChar(string srcColumn, string dstColumn, string arg, Type transformType)
                {
                    StringBuilder columnArgument = InferenceHelpers.GetTextTransformBigramTriCharArgument(srcColumn, dstColumn);

                    columnArgument.Append(arg);
                    var nodeInput = new ML.Legacy.Transforms.TextFeaturizer();
                    nodeInput.WordFeatureExtractor = new NGramNgramExtractor { NgramLength = 2 };
                    nodeInput.CharFeatureExtractor = new NGramNgramExtractor { NgramLength = 3 };
                    nodeInput.Column = new ML.Legacy.Transforms.TextTransformColumn
                    {
                        Name = dstColumn,
                        Source = new[] { srcColumn }
                    };

                    return TextTransform(srcColumn, dstColumn, columnArgument.ToString(), "Bigram plus Trichar",
                        transformType, new TransformPipelineNode(nodeInput));
                }

                public static SuggestedTransform TextTransform(string srcColumn, string dstColumn, string arg,
                    string outputMsg, Type transformType, TransformPipelineNode pipelineNode)
                {
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = srcColumn } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = dstColumn } };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                    return
                        new SuggestedTransform(
                            string.Format(
                                "Apply text-vectorize featurization(" + outputMsg +
                                ") for column '{0}' and output to column '{1}'",
                                srcColumn, dstColumn),
                            new TransformString("Text", arg),
                            transformType, pipelineNode, -1, routingStructure);
                }
            }

            // REVIEW: Needs to be thoroughly tested once entrypoints are supported for the various transforms.
            // Lots of transforms applied here, with many columns produced.
            public sealed class SdcaTransform : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    List<string> tempColumnList = new List<string>();

                    List<string> textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType.IsText && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName).ToList();

                    if ((textColumnNames.Count == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Count > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcatSdca");
                        yield return
                            InferenceHelpers.ConcatColumnsIntoOne(textColumnNames, concatTextColumnName, GetType(), false);
                    }
                    else
                        concatTextColumnName = textColumnNames.First();

                    //Get Unigram + Trichar for text transform on the concatenated text column.
                    string featureTextColumn = columns[0].GetTempColumnName("FeaturesText");
                    yield return InferenceHelpers.TextTransformUnigramTriChar(concatTextColumnName, featureTextColumn, " tokens=+", GetType());

                    //Get Tree Featurizer with FastTreeRegression.
                    var args = new TransformString("TreeFeaturizationTransform", "tr=FastTreeRegression feat=" + featureTextColumn);

                    // REVIEW: Once entrypoint defined for TreeFeaturizationTransform, add ep object.
                    string treeFeaturizerOutputColumnName = "Leaves";
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = featureTextColumn } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = treeFeaturizerOutputColumnName } };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                    yield return
                        new SuggestedTransform($"Apply tree featurizer transform  with fast tree regression on text features for column '{featureTextColumn}'",
                        args, GetType(), null, -1, routingStructure);

                    //Concat-Rename Leaves column generated by tree featurizer.
                    string featuresTreeFeatColumn = columns[0].GetTempColumnName("FeaturesTreeFeat");
                    var epInput = new ML.Legacy.Transforms.ColumnConcatenator
                    {
                        Column = new[]
                            {
                                new ML.Legacy.Transforms.ConcatTransformColumn
                                {
                                    Name = featuresTreeFeatColumn,
                                    Source = new [] { treeFeaturizerOutputColumnName }
                                }
                            }
                    };
                    ColumnRoutingStructure.AnnotatedName[] columnsSourceCr =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = treeFeaturizerOutputColumnName } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDestCr =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = featuresTreeFeatColumn } };
                    var routingStructureCr = new ColumnRoutingStructure(columnsSourceCr, columnsDestCr);
                    yield return
                       new SuggestedTransform("Concatenate-Rename Leaves column generated by tree featurizer to " + featuresTreeFeatColumn,
                           new TransformString("Concat", $"col={featuresTreeFeatColumn}:{treeFeaturizerOutputColumnName}"),
                           GetType(), new TransformPipelineNode(epInput), -1, routingStructureCr);

                    //Get TrainScore with KMeansPlusPlus.
                    args = new TransformString("TrainScore", "tr=KMeansPlusPlus feat=" + featureTextColumn);

                    // REVIEW: Need entrypoint for TrainScore, then add entrypoint pipeline object
                    string kMeansOutputColumnName = "Score";
                    ColumnRoutingStructure.AnnotatedName[] columnsSourceKm =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = featureTextColumn } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDestKm =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = kMeansOutputColumnName } };
                    var routingStructureKm = new ColumnRoutingStructure(columnsSourceKm, columnsDestKm);
                    yield return
                        new SuggestedTransform($"Apply train and score transform on text features for column '{featureTextColumn}'",
                            args, GetType(), null, -1, routingStructureKm);

                    //Concat-Rename Score column generated by Train Score with KMeans.
                    string featuresKMeansColumn = columns[0].GetTempColumnName("FeaturesKMeans");
                    var epInput2 = new ML.Legacy.Transforms.ColumnConcatenator
                    {
                        Column = new[]
                            {
                                new ML.Legacy.Transforms.ConcatTransformColumn
                                {
                                    Name = featuresKMeansColumn,
                                    Source = new [] { kMeansOutputColumnName }
                                }
                            }
                    };
                    ColumnRoutingStructure.AnnotatedName[] columnsSourceCc =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = kMeansOutputColumnName } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDestCc =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = featuresKMeansColumn } };
                    var routingStructureCc = new ColumnRoutingStructure(columnsSourceCc, columnsDestCc);
                    yield return
                       new SuggestedTransform("Concatenate-Rename Score column generated by Train Score with KMeans to " + featuresKMeansColumn,
                           new TransformString("Concat", $"col={featuresKMeansColumn}:{kMeansOutputColumnName}"),
                           GetType(), new TransformPipelineNode(epInput2), -1, routingStructureCc);

                    tempColumnList.Add(featureTextColumn);
                    tempColumnList.Add(featuresTreeFeatColumn);
                    tempColumnList.Add(featuresKMeansColumn);
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        tempColumnList.Add(DefaultColumnNames.Features);

                    //Concat text featurized column into feature column.
                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        yield return InferenceHelpers.ConcatColumnsIntoOne(tempColumnList, DefaultColumnNames.Features, GetType(), true);
                }
            }

            public sealed class NaiveBayesTransform : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    List<string> textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType.IsText && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName)
                            .ToList();

                    if ((textColumnNames.Count == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1) ||
                        (columns.Any(col => col.Purpose == ColumnPurpose.Label && col.Type.ItemType != NumberType.R4)))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Count > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcatNB");
                        yield return
                            InferenceHelpers.ConcatColumnsIntoOne(textColumnNames, concatTextColumnName, GetType(), false);
                    }
                    else
                        concatTextColumnName = textColumnNames.First();

                    //Get Unigram + Trichar text transform.
                    string concatTextColumnTextFeature = columns[0].GetTempColumnName("FeaturesTextNB");
                    yield return InferenceHelpers.TextTransformUnigramTriChar(concatTextColumnName, concatTextColumnTextFeature, string.Empty, GetType());

                    //Get Tree Featurizer with FastTreeRegression.
                    var args = new TransformString("TreeFeaturizationTransform", "tr=FastForestRegression{shuffleLabels+ nl=80} feat=" + concatTextColumnTextFeature);
                    string treeFeaturizerOutputColName = "Leaves";
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = concatTextColumnTextFeature} };
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = treeFeaturizerOutputColName} };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                    var epInput = new ML.Legacy.Transforms.TreeLeafFeaturizer();
                    var epFastForestRegressor = new Legacy.Trainers.FastForestRegressor
                    {
                        FeatureColumn = concatTextColumnTextFeature,
                        NumLeaves = 80,
                        ShuffleLabels = true,
                        LabelColumn = DefaultColumnNames.Label
                    };

                    yield return
                        new SuggestedTransform(
                            $"Apply tree featurizer transform with fast forest regression on '{DefaultColumnNames.Label}' column and '{concatTextColumnTextFeature}' column",
                            args, GetType(), new TransformPipelineNode(epInput, subTrainerObj: epFastForestRegressor), -1, routingStructure);

                    //Concat text featurized column into feature column.
                    List<string> featureCols = new List<string>(new[] { treeFeaturizerOutputColName });
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        featureCols.Add(DefaultColumnNames.Features);
                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        yield return InferenceHelpers.ConcatColumnsIntoOne(featureCols, DefaultColumnNames.Features, GetType(), true);
                }
            }

            public sealed class Text : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType.IsText || column.Purpose != ColumnPurpose.TextFeature)
                            continue;
                        ch.Info("Suggested text featurization for text column '{0}'.", column.ColumnName);

                        var columnDestSuffix = "_tf";
                        var columnNameSafe = column.ColumnName;
                        var quoted = false;
                        if (CmdQuoter.NeedsQuoting(columnNameSafe))
                        {
                            quoted = true;
                            var columnArgument = new StringBuilder();
                            CmdQuoter.QuoteValue(columnNameSafe, columnArgument);
                            columnNameSafe = columnArgument.ToString();
                        }

                        string columnDestRenamed = $"{columnNameSafe}{columnDestSuffix}";
                        var columnSourceDest = quoted ? $"col={{name={columnDestRenamed} src={columnNameSafe}}}" :
                            $"col={columnDestRenamed}:{columnNameSafe}";
                        var args = new TransformString("Text", columnSourceDest);

                        featureCols.Add(columnDestRenamed);
                        var epInput = new ML.Legacy.Transforms.TextFeaturizer
                        {
                            Column = new ML.Legacy.Transforms.TextTransformColumn
                            {
                                Name = columnDestRenamed,
                                Source = new[] { columnNameSafe }
                            },
                            OutputTokens = false
                        };
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = columnNameSafe} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = columnDestRenamed} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(
                            $"Apply text featurizer transform on text features for column '{column.ColumnName}'", args, typeof(Text), new TransformPipelineNode(epInput), -1, routingStructure);
                    }

                    // Concat text featurized columns into existing feature column, if transformed at least one column.
                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, GetType(), IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }
            }

            public sealed class TextUniGramTriGram : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    List<string> textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType.IsText && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName).ToList();

                    if ((textColumnNames.Count == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Count > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcat");
                        yield return
                            InferenceHelpers.ConcatColumnsIntoOne(textColumnNames, concatTextColumnName, GetType(), false);
                    }
                    else
                        concatTextColumnName = textColumnNames.First();

                    //Get Unigram + Trichar for text transform on the concatenated text column.
                    string featureTextColumn = columns[0].GetTempColumnName("FeaturesText");
                    yield return InferenceHelpers.TextTransformUnigramTriChar(concatTextColumnName, featureTextColumn, string.Empty, GetType());

                    //Concat text featurized column into feature column.
                    List<string> featureCols = new List<string>(new[] { featureTextColumn });
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        featureCols.Add(DefaultColumnNames.Features);

                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        yield return InferenceHelpers.ConcatColumnsIntoOne(featureCols, DefaultColumnNames.Features, GetType(), true);
                }
            }

            public sealed class TextBiGramTriGram : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    List<string> textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType.IsText && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName).ToList();

                    if ((textColumnNames.Count == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Count > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcat");
                        yield return
                            InferenceHelpers.ConcatColumnsIntoOne(textColumnNames, concatTextColumnName, GetType(), false);
                    }
                    else
                        concatTextColumnName = textColumnNames.First();

                    //Get Bigram + Trichar for text transform on the concatenated text column.
                    string featureTextColumn = columns[0].GetTempColumnName("FeaturesText");
                    yield return InferenceHelpers.TextTransformBigramTriChar(concatTextColumnName, featureTextColumn, string.Empty, GetType());

                    //Concat text featurized column into feature column.
                    List<string> featureCols = new List<string>(new[] { featureTextColumn });
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        featureCols.Add(DefaultColumnNames.Features);

                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        yield return InferenceHelpers.ConcatColumnsIntoOne(featureCols, DefaultColumnNames.Features, GetType(), true);
                }
            }

            public sealed class NumericMissing : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    bool found = false;
                    var columnArgument = new StringBuilder();
                    var columnNameQuoted = new StringBuilder();
                    foreach (var column in columns)
                    {
                        if (column.Type.ItemType != NumberType.R4 || column.Purpose != ColumnPurpose.NumericFeature)
                            continue;
                        if (!column.HasMissing)
                            continue;

                        ch.Info("Column '{0}' has missing values. Suggested missing indicator encoding.", column.ColumnName);
                        found = true;

                        columnArgument.Append("col=");
                        if (CmdQuoter.NeedsQuoting(column.ColumnName))
                        {
                            columnArgument.Append("{name=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append(" src=");
                            CmdQuoter.QuoteValue(column.ColumnName, columnArgument);
                            columnArgument.Append("} ");
                            CmdQuoter.QuoteValue(column.ColumnName, columnNameQuoted);
                        }
                        else
                        {
                            columnArgument.AppendFormat("{0} ", column.ColumnName);
                            columnNameQuoted.AppendFormat("{0}", column.ColumnName);
                        }
                    }
                    if (found)
                    {
                        string name = columnNameQuoted.ToString();
                        var args = new TransformString("NAHandle", columnArgument.ToString());
                        var epInput = new ML.Legacy.Transforms.MissingValueHandler
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.NAHandleTransformColumn
                                {
                                    Name = name,
                                    Source = name
                                }
                            }
                        };
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = name} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform("Replace missing features with zeroes and concatenate missing indicators", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }
                }
            }

            public class FeaturesColumnConcatRename : TransformInferenceExpertBase
            {
                public virtual bool IgnoreColumn(ColumnPurpose purpose)
                {
                    if (purpose != ColumnPurpose.TextFeature
                        && purpose != ColumnPurpose.CategoricalFeature
                        && purpose != ColumnPurpose.NumericFeature)
                        return true;
                    return false;
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    var selectedColumns = columns.Where(c => !IgnoreColumn(c.Purpose)).ToArray();
                    var colList = selectedColumns.Select(c => c.ColumnName).ToArray();
                    bool allColumnsNumeric = selectedColumns.All(c => c.Purpose == ColumnPurpose.NumericFeature && c.Type.ItemType != BoolType.Instance);
                    bool allColumnsNonNumeric = selectedColumns.All(c => c.Purpose != ColumnPurpose.NumericFeature);

                    if (colList.Length > 0)
                    {
                        var columnArgument = new StringBuilder();
                        // Check if column is named features and already numeric
                        if (colList.Length == 1 && colList[0] == DefaultColumnNames.Features && allColumnsNumeric)
                            yield break;

                        if (!allColumnsNumeric && !allColumnsNonNumeric)
                            yield break;

                        bool needQuoting = false;
                        List<string> columnListQuoted = new List<string>();

                        foreach (var column in colList)
                        {
                            var columnNameQuoted = new StringBuilder();
                            columnArgument.Append("src=");
                            if (CmdQuoter.NeedsQuoting(column))
                            {
                                needQuoting = true;
                                CmdQuoter.QuoteValue(column, columnArgument);
                                columnArgument.Append(" ");
                                CmdQuoter.QuoteValue(column, columnNameQuoted);
                            }
                            else
                            {
                                columnArgument.AppendFormat("{0} ", column);
                                columnNameQuoted.AppendFormat("{0}", column);
                            }
                            columnListQuoted.Add(columnNameQuoted.ToString());
                        }

                        string arguments;

                        if (needQuoting)
                            arguments = $"col={{name={DefaultColumnNames.Features} {columnArgument}}}";
                        else
                            arguments = $"col={DefaultColumnNames.Features}:{string.Join(",", colList)}";

                        var args = new TransformString("Concat", arguments);
                        var epInput = new ML.Legacy.Transforms.ColumnConcatenator
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.ConcatTransformColumn
                                {
                                    Name = DefaultColumnNames.Features,
                                    Source = columnListQuoted.ToArray()
                                }
                            }
                        };

                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            columnListQuoted.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = allColumnsNumeric, Name = c }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = allColumnsNumeric, Name = DefaultColumnNames.Features} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform("Concatenate feature columns into one", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }
                }
            }

            public sealed class FeaturesColumnConcatRenameIgnoreText : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.CategoricalFeature && purpose != ColumnPurpose.NumericFeature);
                }
            }

            public sealed class FeaturesColumnConcatRenameNumericOnly : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.NumericFeature);
                }
            }

            public sealed class NameColumnConcatRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs, IChannel ch)
                {
                    int count = 0;
                    bool isAllText = true;
                    var colSpec = new StringBuilder();
                    var colSpecTextOnly = new List<string>();
                    var columnListQuoted = new List<string>();

                    foreach (var column in columns)
                    {
                        var columnNameQuoted = new StringBuilder();
                        if (column.Purpose != ColumnPurpose.Name)
                            continue;
                        count++;

                        if (colSpec.Length > 0)
                            colSpec.Append(",");
                        colSpec.Append(column.ColumnName);

                        if (CmdQuoter.NeedsQuoting(column.ColumnName))
                            CmdQuoter.QuoteValue(column.ColumnName, columnNameQuoted);
                        else
                            columnNameQuoted.Append(column.ColumnName);
                        columnListQuoted.Add(columnNameQuoted.ToString());

                        if (column.Type.ItemType.IsText)
                            colSpecTextOnly.Add(column.ColumnName);
                        isAllText = isAllText && column.Type.ItemType.IsText;
                    }

                    if (count == 1 && colSpec.ToString() != DefaultColumnNames.Name)
                    {
                        var columnArgument = new StringBuilder();
                        var columnNameQuoted = new StringBuilder();
                        columnArgument.Append("col=");
                        if (CmdQuoter.NeedsQuoting(colSpec, 0))
                        {
                            columnArgument.AppendFormat("{name={0} src=", DefaultColumnNames.Name);
                            CmdQuoter.QuoteValue(colSpec.ToString(), columnArgument);
                            columnArgument.Append("}");
                            CmdQuoter.QuoteValue(colSpec.ToString(), columnNameQuoted);
                        }
                        else
                        {
                            columnArgument.AppendFormat("{0}:{1}", DefaultColumnNames.Name, colSpec);
                            columnNameQuoted.AppendFormat("{0}", colSpec);
                        }
                        var args = new TransformString("Copy", columnArgument.ToString());
                        var epInput = new ML.Legacy.Transforms.ColumnCopier
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.CopyColumnsTransformColumn
                                {
                                    Name = DefaultColumnNames.Name,
                                    Source = columnNameQuoted.ToString()
                                }
                            }
                        };
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = columnNameQuoted.ToString()} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = DefaultColumnNames.Name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform("Rename name column to 'Name'", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }
                    else if (count > 1)
                    {
                        if (!isAllText)
                            ch.Warning("Not all name columns are textual. Ignored non-textual name columns.");
                        if (string.IsNullOrWhiteSpace(colSpecTextOnly.ToString()))
                            yield break;

                        ch.Info("Suggested grouping name columns into one vector.");
                        var quoutedArgument = new StringBuilder();
                        var needQuoting = false;
                        foreach (var column in colSpecTextOnly)
                        {
                            quoutedArgument.Append(" src=");
                            if (CmdQuoter.NeedsQuoting(column))
                            {
                                needQuoting = true;
                                CmdQuoter.QuoteValue(column, quoutedArgument);
                            }
                            else
                                quoutedArgument.Append(column);
                        }
                        string arguments;
                        if (needQuoting)
                            arguments = $"col={{ name={DefaultColumnNames.Name} {quoutedArgument} }}";
                        else
                            arguments = $"col={DefaultColumnNames.Name}:{string.Join(",", colSpecTextOnly)}";
                        var args = new TransformString("Concat", arguments);
                        var epInput = new ML.Legacy.Transforms.ColumnConcatenator
                        {
                            Column = new[]
                            {
                                new ML.Legacy.Transforms.ConcatTransformColumn
                                {
                                    Name = DefaultColumnNames.Name,
                                    Source = columnListQuoted.ToArray()
                                }
                            }
                        };

                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            columnListQuoted.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = DefaultColumnNames.Name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform("Concatenate name columns into one", args,
                            GetType(), new TransformPipelineNode(epInput), -1, routingStructure);
                    }
                }
            }
        }

        /// <summary>
        /// Automatically infer transforms for the data view
        /// </summary>
        public static InferenceResult InferTransforms(IHostEnvironment env, IDataView data, PurposeInference.Column[] purposes, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("InferTransforms");
            h.CheckValue(data, nameof(data));
            h.CheckNonEmpty(purposes, nameof(purposes));
            h.CheckValue(args, nameof(args));
            h.Check(args.EstimatedSampleFraction > 0);

            data = data.Take(MaxRowsToRead);
            var cols = purposes.Where(x => !data.Schema.IsHidden(x.ColumnIndex)).Select(x => new IntermediateColumn(data, x)).ToArray();
            using (var rootCh = h.Start("InferTransforms"))
            {
                var list = new List<SuggestedTransform>();
                int atomicGroupId = args.AtomicIdOffset;
                var includeFeaturesOverride = false;
                foreach (var expert in GetExperts(args.ExcludeFeaturesConcatTransforms))
                {
                    using (var ch = h.Start(expert.GetType().ToString()))
                    {
                        expert.IncludeFeaturesOverride = includeFeaturesOverride;
                        SuggestedTransform[] suggestions = expert.Apply(cols, args, ch).ToArray();
                        includeFeaturesOverride |= expert.IncludeFeaturesOverride;

                        // Set level and group values.
                        for (int i = 0; i < suggestions.Length; i++)
                        {
                            suggestions[i].AtomicGroupId = atomicGroupId;
                            suggestions[i].RoutingStructure.Level = args.Level;
                        }

                        list.AddRange(suggestions);
                        if (suggestions.Length > 0)
                            atomicGroupId++;
                        ch.Done();
                    }
                }

                if (list.Count == 0)
                    rootCh.Info("No transforms are needed for the data.");
                rootCh.Done();
                return new InferenceResult(list.ToArray());
            }
        }

        public static SuggestedTransform[] InferTransforms(IHostEnvironment env, IDataView data, Arguments args, RoleMappedData dataRoles)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("InferTransforms");
            h.CheckValue(data, nameof(data));
            h.CheckValue(args, nameof(args));
            h.Check(args.EstimatedSampleFraction > 0);

            //available transforms in this environment
            var availableTransforms = env.ComponentCatalog.AllEntryPoints()
                .Where(x => x.InputKinds?.FirstOrDefault(i => i == typeof(CommonInputs.ITransformInput)) != null
                    && x.InputKinds?.Any(i => i == typeof(CommonInputs.ICalibratorInput)) != true);
            var dataSample = data.Take(MaxRowsToRead);

            // Infer column purposes from data sample.
            var piArgs = new PurposeInference.Arguments { MaxRowsToRead = MaxRowsToRead };
            var columnIndices = Enumerable.Range(0, dataSample.Schema.ColumnCount);
            var piResult = PurposeInference.InferPurposes(env, dataSample, columnIndices, piArgs, dataRoles);
            var purposes = piResult.Columns;

            // Infer transforms
            var inferenceResult = InferTransforms(env, data, purposes, args);

            // Keep viable transforms, where all transforms in the atomic group have pipeline nodes.
            // First clause: ensure transform t has a pipeline node (is runnable).
            // Second clause: ensure transform is in module catalog entry point list for this platform.
            // Third clause: ensure all members of t's atoimic group are also available, since it is all
            // or nothing for atomic groups (hence the name atomic -- 'uncuttable').
            return inferenceResult.SuggestedTransforms.Where(t => t.PipelineNode != null
                && availableTransforms.Any(a => a.Name.Equals(t.PipelineNode.GetEpName()))
                && !inferenceResult.SuggestedTransforms
                .Where(t2 => t2.PipelineNode == null)
                .Select(t2 => t2.AtomicGroupId)
                .Contains(t.AtomicGroupId)).ToArray();
        }

        public static SuggestedTransform[] InferConcatNumericFeatures(IHostEnvironment env, IDataView data, Arguments args, RoleMappedData dataRoles)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("InferConcatNumericFeatures");
            h.CheckValue(data, nameof(data));
            h.CheckValue(args, nameof(args));
            h.Check(args.EstimatedSampleFraction > 0);

            data = data.Take(MaxRowsToRead);

            // Infer column purposes from data sample.
            var piArgs = new PurposeInference.Arguments { MaxRowsToRead = MaxRowsToRead };
            var columnIndices = Enumerable.Range(0, data.Schema.ColumnCount);
            var piResult = PurposeInference.InferPurposes(env, data, columnIndices, piArgs, dataRoles);
            var purposes = piResult.Columns;

            var cols = purposes.Where(x => !data.Schema.IsHidden(x.ColumnIndex)
                && !args.ExcludedColumnIndices.Contains(x.ColumnIndex))
                .Select(x => new IntermediateColumn(data, x))
                .ToArray();
            using (var rootCh = h.Start("InferConcatNumericFeatures"))
            {
                var list = new List<SuggestedTransform>();
                int atomicGroupId = 0;
                var expert = new Experts.FeaturesColumnConcatRenameNumericOnly();

                using (var ch = h.Start(expert.GetType().ToString()))
                {
                    SuggestedTransform[] suggestions = expert.Apply(cols, args, ch).ToArray();
                    for (int i = 0; i < suggestions.Length; i++)
                        suggestions[i].AtomicGroupId = atomicGroupId;
                    list.AddRange(suggestions);
                    ch.Done();
                }

                if (list.Count == 0)
                    rootCh.Info("No transforms are needed for the data.");
                rootCh.Done();
                return list.ToArray();
            }
        }
    }
}
