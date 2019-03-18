// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(TrainTestCommand.Summary, typeof(TrainTestCommand), typeof(TrainTestCommand.Arguments), typeof(SignatureCommand),
    "Train Test", TrainTestCommand.LoadName)]

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal sealed class TrainTestCommand : DataCommand.ImplBase<TrainTestCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The test data file", ShortName = "test", SortOrder = 1)]
            public string TestFile;

            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr", SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> Trainer;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SortOrder = 101, SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> Scorer;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", NullName = "<Auto>", SortOrder = 102, SignatureType = typeof(SignatureMamlEvaluator))]
            public IComponentFactory<IMamlEvaluator> Evaluator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Results summary filename", ShortName = "sf")]
            public string SummaryFilename;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features", ShortName = "feat", SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group", SortOrder = 5)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 6)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                Name = "CustomColumn", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumns;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize option for the feature column", ShortName = "norm")]
            public NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The validation data file", ShortName = "valid")]
            public string ValidationFile;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should cache input training data", ShortName = "cache")]
            public bool? CacheData;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>", SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save per-instance predictions and metrics to",
                ShortName = "dout")]
            public string OutputDataFile;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should load predictor from input model and use it as the initial model state", ShortName = "cont")]
            public bool ContinueTrain;
        }

        internal const string Summary = "Trains a predictor using the train file and then scores and evaluates the predictor using the test file.";
        public const string LoadName = "TrainTest";

        public TrainTestCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(TrainTestCommand))
        {
            Utils.CheckOptionalUserDirectory(args.SummaryFilename, nameof(args.SummaryFilename));
            Utils.CheckOptionalUserDirectory(args.OutputDataFile, nameof(args.OutputDataFile));
            TrainUtils.CheckTrainer(Host, args.Trainer, args.DataFile);
            if (string.IsNullOrWhiteSpace(args.TestFile))
                throw Host.ExceptUserArg(nameof(args.TestFile), "Test file must be defined.");
        }

        public override void Run()
        {
            using (var ch = Host.Start(LoadName))
            using (var server = InitServer(ch))
            {
                var settings = CmdParser.GetSettings(Host, ImplOptions, new Arguments());
                string cmd = string.Format("maml.exe {0} {1}", LoadName, settings);
                ch.Info(cmd);

                SendTelemetry(Host);

                using (new TimerScope(Host, ch))
                {
                    RunCore(ch, cmd);
                }
            }
        }

        protected override void SendTelemetryCore(IPipe<TelemetryMessage> pipe)
        {
            SendTelemetryComponent(pipe, ImplOptions.Trainer);
            base.SendTelemetryCore(pipe);
        }

        private void RunCore(IChannel ch, string cmd)
        {
            Host.AssertValue(ch);
            Host.AssertNonEmpty(cmd);

            ch.Trace("Constructing trainer");
            ITrainer trainer = ImplOptions.Trainer.CreateComponent(Host);

            IPredictor inputPredictor = null;
            if (ImplOptions.ContinueTrain && !TrainUtils.TryLoadPredictor(ch, Host, ImplOptions.InputModelFile, out inputPredictor))
                ch.Warning("No input model file specified or model file did not contain a predictor. The model state cannot be initialized.");

            ch.Trace("Constructing the training pipeline");
            IDataView trainPipe = CreateLoader();

            var schema = trainPipe.Schema;
            string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                ImplOptions.LabelColumn, DefaultColumnNames.Label);
            string features = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumn),
                ImplOptions.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn),
                ImplOptions.GroupColumn, DefaultColumnNames.GroupId);
            string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                ImplOptions.WeightColumn, DefaultColumnNames.Weight);
            string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.NameColumn),
                ImplOptions.NameColumn, DefaultColumnNames.Name);

            TrainUtils.AddNormalizerIfNeeded(Host, ch, trainer, ref trainPipe, features, ImplOptions.NormalizeFeatures);

            ch.Trace("Binding columns");
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, ImplOptions.CustomColumns);
            var data = new RoleMappedData(trainPipe, label, features, group, weight, name, customCols);

            RoleMappedData validData = null;
            if (!string.IsNullOrWhiteSpace(ImplOptions.ValidationFile))
            {
                if (!trainer.Info.SupportsValidation)
                {
                    ch.Warning("Ignoring validationFile: Trainer does not accept validation dataset.");
                }
                else
                {
                    ch.Trace("Constructing the validation pipeline");
                    IDataView validPipe = CreateRawLoader(dataFile: ImplOptions.ValidationFile);
                    validPipe = ApplyTransformUtils.ApplyAllTransformsToData(Host, trainPipe, validPipe);
                    validData = new RoleMappedData(validPipe, data.Schema.GetColumnRoleNames());
                }
            }

            // In addition to the training set, some trainers can accept two data sets, validation set and test set,
            // in training phase. The major difference between validation set and test set is that training process may
            // indirectly use validation set to improve the model but the learned model should totally independent of test set.
            // Similar to validation set, the trainer can report the scores computed using test set.
            RoleMappedData testDataUsedInTrainer = null;
            if (!string.IsNullOrWhiteSpace(ImplOptions.TestFile))
            {
                // In contrast to the if-else block for validation above, we do not throw a warning if test file is provided
                // because this is TrainTest command.
                if (trainer.Info.SupportsTest)
                {
                    ch.Trace("Constructing the test pipeline");
                    IDataView testPipeUsedInTrainer = CreateRawLoader(dataFile: ImplOptions.TestFile);
                    testPipeUsedInTrainer = ApplyTransformUtils.ApplyAllTransformsToData(Host, trainPipe, testPipeUsedInTrainer);
                    testDataUsedInTrainer = new RoleMappedData(testPipeUsedInTrainer, data.Schema.GetColumnRoleNames());
                }
            }

            var predictor = TrainUtils.Train(Host, ch, data, trainer, validData,
                ImplOptions.Calibrator, ImplOptions.MaxCalibrationExamples, ImplOptions.CacheData, inputPredictor, testDataUsedInTrainer);

            ILegacyDataLoader testPipe;
            bool hasOutfile = !string.IsNullOrEmpty(ImplOptions.OutputModelFile);
            var tempFilePath = hasOutfile ? null : Path.GetTempFileName();

            using (var file = new SimpleFileHandle(ch, hasOutfile ? ImplOptions.OutputModelFile : tempFilePath, true, !hasOutfile))
            {
                TrainUtils.SaveModel(Host, ch, file, predictor, data, cmd);
                ch.Trace("Constructing the testing pipeline");
                using (var stream = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(stream, ch))
                    testPipe = LoadLoader(rep, ImplOptions.TestFile, true);
            }

            // Score.
            ch.Trace("Scoring and evaluating");
            ch.Assert(ImplOptions.Scorer == null || ImplOptions.Scorer is ICommandLineComponentFactory, "TrainTestCommand should only be used from the command line.");
            IDataScorerTransform scorePipe = ScoreUtils.GetScorer(ImplOptions.Scorer, predictor, testPipe, features, group, customCols, Host, data.Schema);

            // Evaluate.
            var evaluator = ImplOptions.Evaluator?.CreateComponent(Host) ??
                EvaluateUtils.GetEvaluator(Host, scorePipe.Schema);
            var dataEval = new RoleMappedData(scorePipe, label, features,
                group, weight, name, customCols, opt: true);
            var metrics = evaluator.Evaluate(dataEval);
            MetricWriter.PrintWarnings(ch, metrics);
            evaluator.PrintFoldResults(ch, metrics);
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out var overall))
                throw ch.Except("No overall metrics found");
            overall = evaluator.GetOverallResults(overall);
            MetricWriter.PrintOverallMetrics(Host, ch, ImplOptions.SummaryFilename, overall, 1);
            evaluator.PrintAdditionalMetrics(ch, metrics);
            Dictionary<string, IDataView>[] metricValues = { metrics };
            SendTelemetryMetric(metricValues);
            if (!string.IsNullOrWhiteSpace(ImplOptions.OutputDataFile))
            {
                var perInst = evaluator.GetPerInstanceMetrics(dataEval);
                var perInstData = new RoleMappedData(perInst, label, null, group, weight, name, customCols);
                var idv = evaluator.GetPerInstanceDataViewToSave(perInstData);
                MetricWriter.SavePerInstance(Host, ch, ImplOptions.OutputDataFile, idv);
            }
        }
    }
}