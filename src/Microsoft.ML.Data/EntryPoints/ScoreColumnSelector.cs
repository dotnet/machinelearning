// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static partial class ScoreModel
    {
        public sealed class ScoreColumnSelectorInput : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Extra columns to write", SortOrder = 2)]
            public string[] ExtraColumns;
        }

        [TlcModule.EntryPoint(Name = "Transforms.ScoreColumnSelector", Desc = "Selects only the last score columns and the extra columns specified in the arguments.", UserName = "Choose Columns By Index")]
        public static CommonOutputs.TransformOutput SelectColumns(IHostEnvironment env, ScoreColumnSelectorInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(env, input);
            var view = input.Data;
            var maxScoreId = view.Schema.GetMaxMetadataKind(out int colMax, MetadataUtils.Kinds.ScoreColumnSetId);
            List<int> indices = new List<int>();
            for (int i = 0; i < view.Schema.ColumnCount; i++)
            {
                if (view.Schema.IsHidden(i))
                    continue;
                if (!ShouldAddColumn(view.Schema, i, input.ExtraColumns, maxScoreId))
                    continue;
                indices.Add(i);
            }
            var newView = new ChooseColumnsByIndexTransform(env, new ChooseColumnsByIndexTransform.Arguments() { Index = indices.ToArray() }, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, newView, input.Data), OutputData = newView };
        }

        private static bool ShouldAddColumn(Schema schema, int i, string[] extraColumns, uint scoreSet)
        {
            uint scoreSetId = 0;
            if (schema.TryGetMetadata(MetadataUtils.ScoreColumnSetIdType.AsPrimitive, MetadataUtils.Kinds.ScoreColumnSetId, i, ref scoreSetId)
                && scoreSetId == scoreSet)
            {
                return true;
            }
            var columnName = schema.GetColumnName(i);
            if (extraColumns != null && Array.FindIndex(extraColumns, columnName.Equals) >= 0)
                return true;
            return false;
        }

        public sealed class RenameBinaryPredictionScoreColumnsInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model used in scoring", SortOrder = 2)]
            public IPredictorModel PredictorModel;
        }

        [TlcModule.EntryPoint(Name = "Transforms.BinaryPredictionScoreColumnsRenamer", Desc = "For binary prediction, it renames the PredictedLabel and Score columns to include the name of the positive class.", UserName = "Rename Binary Prediction Score Columns")]
        public static CommonOutputs.TransformOutput RenameBinaryPredictionScoreColumns(IHostEnvironment env,
            RenameBinaryPredictionScoreColumnsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ScoreModel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            if (input.PredictorModel.Predictor.PredictionKind == PredictionKind.BinaryClassification)
            {
                ColumnType labelType;
                var labelNames = input.PredictorModel.GetLabelInfo(host, out labelType);
                if (labelNames != null && labelNames.Length == 2)
                {
                    var positiveClass = labelNames[1];

                    // Rename all the score columns.
                    int colMax;
                    var maxScoreId = input.Data.Schema.GetMaxMetadataKind(out colMax, MetadataUtils.Kinds.ScoreColumnSetId);
                    var copyCols = new List<(string Source, string Name)>();
                    for (int i = 0; i < input.Data.Schema.ColumnCount; i++)
                    {
                        if (input.Data.Schema.IsHidden(i))
                            continue;
                        if (!ShouldAddColumn(input.Data.Schema, i, null, maxScoreId))
                            continue;
                        // Do not rename the PredictedLabel column.
                        ReadOnlyMemory<char> tmp = default;
                        if (input.Data.Schema.TryGetMetadata(TextType.Instance, MetadataUtils.Kinds.ScoreValueKind, i,
                            ref tmp)
                            && ReadOnlyMemoryUtils.EqualsStr(MetadataUtils.Const.ScoreValueKind.PredictedLabel, tmp))
                        {
                            continue;
                        }
                        var source = input.Data.Schema.GetColumnName(i);
                        var name = source + "." + positiveClass;
                        copyCols.Add((source, name));
                    }

                    var copyColumn = new ColumnsCopyingTransformer(env, copyCols.ToArray()).Transform(input.Data);
                    var dropColumn = ColumnSelectingTransformer.CreateDrop(env, copyColumn, copyCols.Select(c => c.Source).ToArray());
                    return new CommonOutputs.TransformOutput { Model = new TransformModel(env, dropColumn, input.Data), OutputData = dropColumn };
                }
            }

            var newView = NopTransform.CreateIfNeeded(env, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, newView, input.Data), OutputData = newView };
        }
    }
}
