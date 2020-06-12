// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(SvmLightSaver.Summary, typeof(SvmLightSaver), typeof(SvmLightSaver.Arguments), typeof(SignatureDataSaver),
    "SVM-Light Saver", SvmLightSaver.LoadName, "SvmLight", "Svm")]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The SVM-light saver is a saver class that is capable of saving the label,
    /// features, group ID and weight columns of a dataset in SVM-light format. It is a bit
    /// idiosyncratic in that unlike <see cref="TextSaver"/> and <see cref="BinarySaver"/>, there is no
    /// attempt to save all columns, just those specific columns, with other columns being dropped on
    /// the floor.
    /// </summary>
    [BestFriend]
    internal sealed class SvmLightSaver : IDataSaver
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurrenceWins, HelpText = "Write the variant of SVM-light format where feature indices start from 0, not 1", ShortName = "z")]
            public bool Zero;

            [Argument(ArgumentType.LastOccurrenceWins, HelpText = "Format output labels for a binary classification problem (-1 for negative, 1 for positive)", ShortName = "b")]
            public bool Binary;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for features", ShortName = "feat", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string FeatureColumnName = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string LabelColumnName = DefaultColumnNames.Label;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string ExampleWeightColumnName = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example groupId", ShortName = "groupId", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string RowGroupColumnName = null;
        }

        internal const string LoadName = "SvmLightSaver";
        internal const string Summary =
            "Writes Label/Features/Weight/GroupId columns into a data file in SVM-light format. " +
            "Label and Features are required, but the others are optional.";

        private readonly IHost _host;
        private readonly bool _zero;
        private readonly bool _binary;
        private readonly string _featureCol;
        private readonly string _labelCol;
        private readonly string _groupCol;
        private readonly string _weightCol;

        public SvmLightSaver(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(SvmLightSaver.LoadName);
            _host.CheckValue(args, nameof(args));

            _zero = args.Zero;
            _binary = args.Binary;
            _featureCol = args.FeatureColumnName;
            _labelCol = args.LabelColumnName;
            _groupCol = args.RowGroupColumnName;
            _weightCol = args.ExampleWeightColumnName;
        }

        public bool IsColumnSavable(DataViewType type)
        {
            // REVIEW: The SVM-light saver is a bit peculiar in that it does not
            // save all columns, just some columns, and the determination of whether it will
            // save a column or not is not dependent only on its type, but rather its name
            // and other factors. This will claim to save all columns, but it will just
            // ignore a bunch depending not on the type, but on the name.
            return true;
        }

        public void SaveData(Stream stream, IDataView data, params int[] cols)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(data, nameof(data));
            _host.CheckValueOrNull(cols);

            if (cols == null)
                cols = new int[0];

            using (var ch = _host.Start("Saving"))
            {
                var labelCol = data.Schema.GetColumnOrNull(_labelCol);
                if (!labelCol.HasValue)
                    throw ch.Except($"Column {_labelCol} not found in data");

                var featureCol = data.Schema.GetColumnOrNull(_featureCol);
                if (!featureCol.HasValue)
                    throw ch.Except($"Column {_featureCol} not found in data");

                var groupCol = !string.IsNullOrWhiteSpace(_groupCol) ? data.Schema.GetColumnOrNull(_groupCol) : default;
                if (!string.IsNullOrWhiteSpace(_groupCol) && !groupCol.HasValue)
                    throw ch.Except($"Column {_groupCol} not found in data");

                var weightCol = !string.IsNullOrWhiteSpace(_weightCol) ? data.Schema.GetColumnOrNull(_weightCol) : default;
                if (!string.IsNullOrWhiteSpace(_weightCol) && !weightCol.HasValue)
                    throw ch.Except($"Column {_weightCol} not found in data");

                foreach (var col in cols)
                {
                    _host.Check(col < data.Schema.Count);
                    var column = data.Schema[col];
                    if (column.Name != _labelCol && column.Name != _featureCol && column.Name != _groupCol && column.Name != _weightCol)
                        ch.Warning($"Column {column.Name} will not be saved. SVM-light saver saves the label column, feature column, optional group column and optional weight column.");
                }

                var columns = new List<DataViewSchema.Column>() { labelCol.Value, featureCol.Value };
                if (groupCol.HasValue)
                    columns.Add(groupCol.Value);
                if (weightCol.HasValue)
                    columns.Add(weightCol.Value);
                using (var writer = new StreamWriter(stream))
                using (var cursor = data.GetRowCursor(columns))
                {
                    // Getting the getters will fail with type errors if the types are not correct,
                    // so we rely on those messages.
                    var labelGetter = cursor.GetGetter<float>(labelCol.Value);
                    var featuresGetter = cursor.GetGetter<VBuffer<float>>(featureCol.Value);
                    var groupGetter = groupCol.HasValue ? cursor.GetGetter<ulong>(groupCol.Value) : null;
                    var weightGetter = weightCol.HasValue ? cursor.GetGetter<float>(weightCol.Value) : null;
                    VBuffer<float> features = default;
                    while (cursor.MoveNext())
                    {
                        float lab = default;
                        labelGetter(ref lab);
                        if (_binary)
                            writer.Write(float.IsNaN(lab) ? 0 : (lab > 0 ? 1 : -1));
                        else
                            writer.Write("{0:R}", lab);
                        if (groupGetter != null)
                        {
                            ulong groupId = default;
                            groupGetter(ref groupId);
                            if (groupId > 0)
                                writer.Write(" qid:{0}", groupId - 1);
                        }
                        if (weightGetter != null)
                        {
                            float weight = default;
                            weightGetter(ref weight);
                            if (weight != 1)
                                writer.Write(" cost:{0:R}", weight);
                        }

                        featuresGetter(ref features);
                        bool any = false;
                        foreach (var pair in features.Items().Where(p => p.Value != 0))
                        {
                            writer.Write(" {0}:{1}", _zero ? pair.Key : (pair.Key + 1), pair.Value);
                            any = true;
                        }
                        // If there were no non-zero items, write a dummy item. Some parsers can handle
                        // empty arrays correctly, but some assume there is at least one defined item.
                        if (!any)
                            writer.Write(" {0}:0", _zero ? 0 : 1);
                        writer.WriteLine();
                    }
                }
            }
        }
    }
}
