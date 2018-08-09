// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(TestCommand.Summary, typeof(TestCommand), typeof(TestCommand.Arguments), typeof(SignatureCommand),
    "Test Predictor", "Test")]

namespace Microsoft.ML.Runtime.Data
{
    // This command is essentially chaining together Score and Evaluate, without the need to save the intermediary scored data.
    public sealed class TestCommand : DataCommand.ImplBase<TestCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for features", ShortName = "feat", SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group", SortOrder = 5)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 6)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SortOrder = 101, SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> Scorer;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", NullName = "<Auto>", SortOrder = 102)]
            public SubComponent<IMamlEvaluator, SignatureMamlEvaluator> Evaluator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Results summary filename", ShortName = "sf")]
            public string SummaryFilename;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save per-instance predictions and metrics to",
                ShortName = "dout")]
            public string OutputDataFile;
        }

        internal const string Summary = "Scores and evaluates a data file.";

        public TestCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(TestCommand))
        {
            Host.CheckUserArg(!string.IsNullOrEmpty(Args.InputModelFile), nameof(Args.InputModelFile), "The input model file is required.");
            Utils.CheckOptionalUserDirectory(Args.SummaryFilename, nameof(Args.SummaryFilename));
            Utils.CheckOptionalUserDirectory(Args.OutputDataFile, nameof(Args.OutputDataFile));
        }

        public override void Run()
        {
            string command = "Test";
            using (var ch = Host.Start(command))
            using (var server = InitServer(ch))
            {
                var settings = CmdParser.GetSettings(ch, Args, new Arguments());
                ch.Info("maml.exe {0} {1}", command, settings);

                SendTelemetry(Host);
                using (new TimerScope(Host, ch))
                {
                    RunCore(ch);
                }

                ch.Done();
            }
        }

        private void RunCore(IChannel ch)
        {
            ch.Trace("Constructing data pipeline");
            IDataLoader loader;
            IPredictor predictor;
            RoleMappedSchema trainSchema;
            LoadModelObjects(ch, true, out predictor, true, out trainSchema, out loader);
            ch.AssertValue(predictor);
            ch.AssertValueOrNull(trainSchema);
            ch.AssertValue(loader);

            ch.Trace("Binding columns");
            ISchema schema = loader.Schema;
            string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Args.LabelColumn),
                Args.LabelColumn, DefaultColumnNames.Label);
            string features = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Args.FeatureColumn),
                Args.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Args.GroupColumn),
                Args.GroupColumn, DefaultColumnNames.GroupId);
            string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Args.WeightColumn),
                Args.WeightColumn, DefaultColumnNames.Weight);
            string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Args.NameColumn),
                Args.NameColumn, DefaultColumnNames.Name);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);

            // Score.
            ch.Trace("Scoring and evaluating");
            ch.Assert(Args.Scorer == null || Args.Scorer is ICommandLineComponentFactory, "TestCommand should only be used from the command line.");
            IDataScorerTransform scorePipe = ScoreUtils.GetScorer(Args.Scorer, predictor, loader, features, group, customCols, Host, trainSchema);

            // Evaluate.
            var evalComp = Args.Evaluator;
            if (!evalComp.IsGood())
                evalComp = EvaluateUtils.GetEvaluatorType(ch, scorePipe.Schema);
            var evaluator = evalComp.CreateInstance(Host);
            var data = new RoleMappedData(scorePipe, label, null, group, weight, name, customCols);
            var metrics = evaluator.Evaluate(data);
            MetricWriter.PrintWarnings(ch, metrics);
            evaluator.PrintFoldResults(ch, metrics);
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out var overall))
                throw ch.Except("No overall metrics found");
            overall = evaluator.GetOverallResults(overall);
            MetricWriter.PrintOverallMetrics(Host, ch, Args.SummaryFilename, overall, 1);
            evaluator.PrintAdditionalMetrics(ch, metrics);
            Dictionary<string, IDataView>[] metricValues = { metrics };
            SendTelemetryMetric(metricValues);
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