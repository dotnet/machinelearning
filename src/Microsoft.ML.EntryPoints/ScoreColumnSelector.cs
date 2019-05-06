// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.EntryPoints
{
    internal static partial class ScoreModel
    {
        public sealed class ScoreColumnSelectorInput : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Extra columns to write", SortOrder = 2)]
            public string[] ExtraColumns;
        }

        [TlcModule.EntryPoint(Name = "Transforms.ScoreColumnSelector", Desc = "Selects only the last score columns and the extra columns specified in the arguments.", UserName = "Choose Columns By Indices")]
        public static CommonOutputs.TransformOutput SelectColumns(IHostEnvironment env, ScoreColumnSelectorInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(env, input);
            var view = input.Data;
            var maxScoreId = view.Schema.GetMaxAnnotationKind(out int colMax, AnnotationUtils.Kinds.ScoreColumnSetId);
            List<int> indices = new List<int>();
            for (int i = 0; i < view.Schema.Count; i++)
            {
                if (view.Schema[i].IsHidden)
                    continue;
                if (!ShouldAddColumn(view.Schema, i, input.ExtraColumns, maxScoreId))
                    continue;
                indices.Add(i);
            }
            var newView = new ChooseColumnsByIndexTransform(env, new ChooseColumnsByIndexTransform.Options() { Indices = indices.ToArray() }, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, newView, input.Data), OutputData = newView };
        }

        private static bool ShouldAddColumn(DataViewSchema schema, int i, string[] extraColumns, uint scoreSet)
        {
            uint scoreSetId = 0;
            if (schema.TryGetAnnotation(AnnotationUtils.ScoreColumnSetIdType, AnnotationUtils.Kinds.ScoreColumnSetId, i, ref scoreSetId)
                && scoreSetId == scoreSet)
            {
                return true;
            }
            var columnName = schema[i].Name;
            if (extraColumns != null && Array.FindIndex(extraColumns, columnName.Equals) >= 0)
                return true;
            return false;
        }

        public sealed class RenameBinaryPredictionScoreColumnsInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model used in scoring", SortOrder = 2)]
            public PredictorModel PredictorModel;
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
                DataViewType labelType;
                var labelNames = input.PredictorModel.GetLabelInfo(host, out labelType);
                if (labelNames != null && labelNames.Length == 2)
                {
                    var positiveClass = labelNames[1];

                    // Rename all the score columns.
                    int colMax;
                    var maxScoreId = input.Data.Schema.GetMaxAnnotationKind(out colMax, AnnotationUtils.Kinds.ScoreColumnSetId);
                    var copyCols = new List<(string name, string source)>();
                    for (int i = 0; i < input.Data.Schema.Count; i++)
                    {
                        if (input.Data.Schema[i].IsHidden)
                            continue;
                        if (!ShouldAddColumn(input.Data.Schema, i, null, maxScoreId))
                            continue;
                        // Do not rename the PredictedLabel column.
                        ReadOnlyMemory<char> tmp = default;
                        if (input.Data.Schema.TryGetAnnotation(TextDataViewType.Instance, AnnotationUtils.Kinds.ScoreValueKind, i,
                            ref tmp)
                            && ReadOnlyMemoryUtils.EqualsStr(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, tmp))
                        {
                            continue;
                        }
                        var source = input.Data.Schema[i].Name;
                        var name = source + "." + positiveClass;
                        copyCols.Add((name, source));
                    }

                    var copyColumn = new ColumnCopyingTransformer(env, copyCols.ToArray()).Transform(input.Data);
                    var dropColumn = ColumnSelectingTransformer.CreateDrop(env, copyColumn, copyCols.Select(c => c.source).ToArray());
                    return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, dropColumn, input.Data), OutputData = dropColumn };
                }
            }

            var newView = NopTransform.CreateIfNeeded(env, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, newView, input.Data), OutputData = newView };
        }
    }
}
