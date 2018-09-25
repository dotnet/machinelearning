// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(EvaluateTransform.Summary, typeof(IDataTransform), typeof(EvaluateTransform), typeof(EvaluateTransform.Arguments), typeof(SignatureDataTransform),
    "Evaluate Predictor", "Evaluate")]

[assembly: LoadableClass(EvaluateCommand.Summary, typeof(EvaluateCommand), typeof(EvaluateCommand.Arguments), typeof(SignatureCommand),
    "Evaluate Predictor", "Evaluate")]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: For simplicity (since this is currently the case),
    // we assume that all metrics are either numeric, or numeric vectors.
    /// <summary>
    /// This class contains information about an overall metric, namely its name and whether it is a vector
    /// metric or not.
    /// </summary>
    public sealed class MetricColumn
    {
        /// <summary>
        /// An enum specifying whether the metric should be maximized or minimized while sweeping. 'Info' should be
        /// used for metrics that are irrelevant to the model's quality (such as the number of positive/negative
        /// examples etc.).
        /// </summary>
        public enum Objective
        {
            Maximize,
            Minimize,
            Info,
        }

        public readonly string LoadName;
        public readonly bool IsVector;
        public readonly Objective MetricTarget;
        public readonly string Name;

        public readonly bool CanBeWeighted;
        private readonly Regex _loadNamePattern;
        private readonly string _groupName;
        private readonly string _nameFormat;

        public MetricColumn(string loadName, string name, Objective target = Objective.Maximize, bool canBeWeighted = true,
            bool isVector = false, Regex namePattern = null, string groupName = null, string nameFormat = null)
        {
            Contracts.CheckValue(loadName, nameof(loadName));
            Contracts.CheckValue(name, nameof(name));

            LoadName = loadName;
            Name = name;
            MetricTarget = target;
            CanBeWeighted = canBeWeighted;
            IsVector = isVector;
            _loadNamePattern = namePattern;
            _groupName = groupName;
            _nameFormat = nameFormat;
        }

        public string GetNameMatch(string input)
        {
            if (_loadNamePattern == null)
            {
                if (input.Equals(LoadName, StringComparison.OrdinalIgnoreCase) || (CanBeWeighted && input == "Weighted" + LoadName))
                    return Name;
                return null;
            }
            if (string.IsNullOrEmpty(_groupName) || string.IsNullOrEmpty(_nameFormat))
                return null;
            var match = _loadNamePattern.Match(input);
            if (!match.Success)
                return null;
            var s = match.Groups[_groupName];
            return string.Format(_nameFormat, s);
        }
    }

    // REVIEW: Move this interface to MLCore when IDataTransform is moved there.
    /// <summary>
    /// This is an interface for evaluation. It has two methods: <see cref="Evaluate"/> and <see cref="GetPerInstanceMetrics"/>.
    /// Both take a <see cref="RoleMappedData"/> as input. The <see cref="RoleMappedData"/> is assumed to contain all the column
    /// roles needed for evaluation, including the score column.
    /// </summary>
    public interface IEvaluator
    {
        /// <summary>
        /// Compute the aggregate metrics. Return a dictionary from the metric kind
        /// (overal/per-fold/confusion matrix/PR-curves etc.), to a data view containing the metric.
        /// </summary>
        Dictionary<string, IDataView> Evaluate(RoleMappedData data);

        /// <summary>
        /// Return an <see cref="IDataTransform"/> containing the per-instance results.
        /// </summary>
        IDataTransform GetPerInstanceMetrics(RoleMappedData data);

        /// <summary>
        /// Get all the overall metrics returned by this evaluator.
        /// </summary>
        IEnumerable<MetricColumn> GetOverallMetricColumns();
    }

    /// <summary>
    /// Signature for creating an IEvaluator.
    /// </summary>
    public delegate void SignatureEvaluator();
    public delegate void SignatureMamlEvaluator();

    public static class EvaluateTransform
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group", SortOrder = 5)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", SignatureType = typeof(SignatureMamlEvaluator))]
            public IComponentFactory<IMamlEvaluator> Evaluator;
        }

        internal const string Summary = "Runs a previously trained predictor on the data.";

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));

            using (var ch = env.Register("EvaluateTransform").Start("Create Transform"))
            {
                ch.Trace("Binding columns");
                ISchema schema = input.Schema;
                string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                    args.LabelColumn, DefaultColumnNames.Label);
                string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn),
                    args.GroupColumn, DefaultColumnNames.GroupId);
                string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                    args.WeightColumn, DefaultColumnNames.Weight);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, args.CustomColumn);

                ch.Trace("Creating evaluator");
                IMamlEvaluator eval = args.Evaluator?.CreateComponent(env) ??
                    EvaluateUtils.GetEvaluator(env, input.Schema);

                var data = new RoleMappedData(input, label, null, group, weight, null, customCols);
                return eval.GetPerInstanceMetrics(data);
            }
        }
    }

    public sealed class EvaluateCommand : DataCommand.ImplBase<EvaluateCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group", SortOrder = 5)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 6)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", SignatureType = typeof(SignatureMamlEvaluator))]
            public IComponentFactory<IMamlEvaluator> Evaluator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Results summary filename", ShortName = "sf")]
            public string SummaryFilename;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "File to save per-instance predictions and metrics to",
                ShortName = "dout")]
            public string OutputDataFile;
        }

        internal const string Summary = "Evaluates the metrics for a scored data file.";

        public EvaluateCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(EvaluateCommand))
        {
            Utils.CheckOptionalUserDirectory(Args.SummaryFilename, nameof(Args.SummaryFilename));
            Utils.CheckOptionalUserDirectory(Args.OutputDataFile, nameof(Args.OutputDataFile));
        }

        public override void Run()
        {
            using (var ch = Host.Start("Evaluate"))
            {
                RunCore(ch);
                ch.Done();
            }
        }

        private void RunCore(IChannel ch)
        {
            Host.AssertValue(ch);

            ch.Trace("Creating loader");
            IDataView view = CreateAndSaveLoader(
                (env, source) => new IO.BinaryLoader(env, new IO.BinaryLoader.Arguments(), source));

            ch.Trace("Binding columns");
            ISchema schema = view.Schema;
            string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                Args.LabelColumn, DefaultColumnNames.Label);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn),
                Args.GroupColumn, DefaultColumnNames.GroupId);
            string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                Args.WeightColumn, DefaultColumnNames.Weight);
            string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.NameColumn),
                Args.NameColumn, DefaultColumnNames.Name);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);

            ch.Trace("Creating evaluator");
            var evaluator = Args.Evaluator?.CreateComponent(Host) ??
                EvaluateUtils.GetEvaluator(Host, view.Schema);

            var data = new RoleMappedData(view, label, null, group, weight, name, customCols);
            var metrics = evaluator.Evaluate(data);
            MetricWriter.PrintWarnings(ch, metrics);
            evaluator.PrintFoldResults(ch, metrics);
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out var overall))
                throw ch.Except("No overall metrics found");
            overall = evaluator.GetOverallResults(overall);
            MetricWriter.PrintOverallMetrics(Host, ch, Args.SummaryFilename, overall, 1);
            evaluator.PrintAdditionalMetrics(ch, metrics);
            if (!string.IsNullOrWhiteSpace(Args.OutputDataFile))
            {
                var perInst = evaluator.GetPerInstanceMetrics(data);
                var perInstData = new RoleMappedData(perInst, label, null, group, weight, name, customCols);
                var idv = evaluator.GetPerInstanceDataViewToSave(perInstData);
                MetricWriter.SavePerInstance(Host, ch, Args.OutputDataFile, idv);
            }
        }
    }
}
