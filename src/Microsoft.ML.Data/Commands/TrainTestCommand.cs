// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(TrainTestCommand.Summary, typeof(TrainTestCommand), typeof(TrainTestCommand.Arguments), typeof(SignatureCommand),
    "Train Test", TrainTestCommand.LoadName)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class TrainTestCommand : DataCommand.ImplBase<TrainTestCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The test data file", ShortName = "test", SortOrder = 1)]
            public string TestFile;

            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr")]
            public SubComponent<ITrainer, SignatureTrainer> Trainer = new SubComponent<ITrainer, SignatureTrainer>("AveragedPerceptron");

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SortOrder = 101)]
            public SubComponent<IDataScorerTransform, SignatureDataScorer> Scorer;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", NullName = "<Auto>", SortOrder = 102)]
            public SubComponent<IMamlEvaluator, SignatureMamlEvaluator> Evaluator;

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

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize option for the feature column", ShortName = "norm")]
            public NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The validation data file", ShortName = "valid")]
            public string ValidationFile;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should cache input training data", ShortName = "cache")]
            public bool? CacheData;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>")]
            public SubComponent<ICalibratorTrainer, SignatureCalibrator> Calibrator = new SubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

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
        private readonly ComponentCatalog.LoadableClassInfo _info;

        public TrainTestCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(TrainTestCommand))
        {
            Utils.CheckOptionalUserDirectory(args.SummaryFilename, nameof(args.SummaryFilename));
            Utils.CheckOptionalUserDirectory(args.OutputDataFile, nameof(args.OutputDataFile));
            _info = TrainUtils.CheckTrainer(Host, args.Trainer, args.DataFile);
            if (string.IsNullOrWhiteSpace(args.TestFile))
                throw Host.ExceptUserArg(nameof(args.TestFile), "Test file must be defined.");
        }

        public override void Run()
        {
            using (var ch = Host.Start(LoadName))
            using (var server = InitServer(ch))
            {
                var settings = CmdParser.GetSettings(ch, Args, new Arguments());
                string cmd = string.Format("maml.exe {0} {1}", LoadName, settings);
                ch.Info(cmd);

                SendTelemetry(Host);

                using (new TimerScope(Host, ch))
                {
                    RunCore(ch, cmd);
                }

                ch.Done();
            }
        }

        protected override void SendTelemetryCore(IPipe<TelemetryMessage> pipe)
        {
            SendTelemetryComponent(pipe, Args.Trainer);
            base.SendTelemetryCore(pipe);
        }

        private void RunCore(IChannel ch, string cmd)
        {
            Host.AssertValue(ch);
            Host.AssertNonEmpty(cmd);

            ch.Trace("Constructing trainer");
            ITrainer trainer = Args.Trainer.CreateInstance(Host);

            IPredictor inputPredictor = null;
            if (Args.ContinueTrain && !TrainUtils.TryLoadPredictor(ch, Host, Args.InputModelFile, out inputPredictor))
                ch.Warning("No input model file specified or model file did not contain a predictor. The model state cannot be initialized.");

            ch.Trace("Constructing the training pipeline");
            IDataView trainPipe = CreateLoader();

            ISchema schema = trainPipe.Schema;
            string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                Args.LabelColumn, DefaultColumnNames.Label);
            string features = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumn),
                Args.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn),
                Args.GroupColumn, DefaultColumnNames.GroupId);
            string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                Args.WeightColumn, DefaultColumnNames.Weight);
            string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.NameColumn),
                Args.NameColumn, DefaultColumnNames.Name);

            TrainUtils.AddNormalizerIfNeeded(Host, ch, trainer, ref trainPipe, features, Args.NormalizeFeatures);

            ch.Trace("Binding columns");
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);
            var data = TrainUtils.CreateExamples(trainPipe, label, features, group, weight, name, customCols);

            RoleMappedData validData = null;
            if (!string.IsNullOrWhiteSpace(Args.ValidationFile))
            {
                if (!TrainUtils.CanUseValidationData(trainer))
                {
                    ch.Warning("Ignoring validationFile: Trainer does not accept validation dataset.");
                }
                else
                {
                    ch.Trace("Constructing the validation pipeline");
                    IDataView validPipe = CreateRawLoader(dataFile: Args.ValidationFile);
                    validPipe = ApplyTransformUtils.ApplyAllTransformsToData(Host, trainPipe, validPipe);
                    validData = RoleMappedData.Create(validPipe, data.Schema.GetColumnRoleNames());
                }
            }

            var predictor = TrainUtils.Train(Host, ch, data, trainer, _info.LoadNames[0], validData,
                Args.Calibrator, Args.MaxCalibrationExamples, Args.CacheData, inputPredictor);

            IDataLoader testPipe;
            using (var file = !string.IsNullOrEmpty(Args.OutputModelFile) ?
                Host.CreateOutputFile(Args.OutputModelFile) : Host.CreateTempFile(".zip"))
            {
                TrainUtils.SaveModel(Host, ch, file, predictor, data, cmd);

                ch.Trace("Constructing the testing pipeline");
                using (var stream = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(stream, ch))
                    testPipe = LoadLoader(rep, Args.TestFile, true);
            }

            // Score.
            ch.Trace("Scoring and evaluating");
            IDataScorerTransform scorePipe = ScoreUtils.GetScorer(Args.Scorer, predictor, testPipe, features, group, customCols, Host, data.Schema);

            // Evaluate.
            var evalComp = Args.Evaluator;
            if (!evalComp.IsGood())
                evalComp = EvaluateUtils.GetEvaluatorType(ch, scorePipe.Schema);
            var evaluator = evalComp.CreateInstance(Host);
            var dataEval = TrainUtils.CreateExamplesOpt(scorePipe, label, features,
                group, weight, name, customCols);
            var metrics = evaluator.Evaluate(dataEval);
            MetricWriter.PrintWarnings(ch, metrics);
            evaluator.PrintFoldResults(ch, metrics);
            evaluator.PrintOverallResults(ch, Args.SummaryFilename, metrics);
            Dictionary<string, IDataView>[] metricValues = { metrics };
            SendTelemetryMetric(metricValues);
            if (!string.IsNullOrWhiteSpace(Args.OutputDataFile))
            {
                var perInst = evaluator.GetPerInstanceMetrics(dataEval);
                var perInstData = TrainUtils.CreateExamples(perInst, label, null, group, weight, name, customCols);
                var idv = evaluator.GetPerInstanceDataViewToSave(perInstData);
                MetricWriter.SavePerInstance(Host, ch, Args.OutputDataFile, idv);
            }
        }
    }
}