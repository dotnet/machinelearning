// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(CrossValidationCommand), typeof(CrossValidationCommand.Arguments), typeof(SignatureCommand),
    "Cross Validation", CrossValidationCommand.LoadName)]

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal sealed class CrossValidationCommand : DataCommand.ImplBase<CrossValidationCommand.Arguments>
    {
        // REVIEW: We need a way to specify different data sets, not just LabeledExamples.
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
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

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for stratification", ShortName = "strat", SortOrder = 7)]
            public string StratificationColumn;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                Name = "CustomColumn", ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumns;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of folds in k-fold cross-validation", ShortName = "k")]
            public int NumFolds = 2;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use threads", ShortName = "threads")]
            public bool UseThreads = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize option for the feature column", ShortName = "norm")]
            public NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should cache input training data", ShortName = "cache")]
            public bool? CacheData;

            [Argument(ArgumentType.Multiple, HelpText = "Transforms to apply prior to splitting the data into folds",
                Name = "PreTransform", ShortName = "prexf", SignatureType = typeof(SignatureDataTransform))]
            public KeyValuePair<string, IComponentFactory<IDataView, IDataTransform>>[] PreTransforms;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The validation data file", ShortName = "valid")]
            public string ValidationFile;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>", SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "File to save per-instance predictions and metrics to",
                ShortName = "dout")]
            public string OutputDataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Print the run/fold index in per-instance output", ShortName = "opf")]
            public bool OutputExampleFoldIndex = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should collate metrics or store them in per-folds files", ShortName = "collate")]
            public bool CollateMetrics = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should load predictor from input model and use it as the initial model state", ShortName = "cont")]
            public bool ContinueTrain;
        }

        private const string RegistrationName = nameof(CrossValidationCommand);
        public const string LoadName = "CV";

        public CrossValidationCommand(IHostEnvironment env, Arguments args)
            : base(env, args, RegistrationName)
        {
            Host.CheckUserArg(ImplOptions.NumFolds >= 2, nameof(ImplOptions.NumFolds), "Number of folds must be greater than or equal to 2.");
            TrainUtils.CheckTrainer(Host, args.Trainer, args.DataFile);
            Utils.CheckOptionalUserDirectory(ImplOptions.SummaryFilename, nameof(ImplOptions.SummaryFilename));
            Utils.CheckOptionalUserDirectory(ImplOptions.OutputDataFile, nameof(ImplOptions.OutputDataFile));
        }

        // This is for "forking" the host environment.
        private CrossValidationCommand(CrossValidationCommand impl)
            : base(impl, RegistrationName)
        {
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

            IPredictor inputPredictor = null;
            if (ImplOptions.ContinueTrain && !TrainUtils.TryLoadPredictor(ch, Host, ImplOptions.InputModelFile, out inputPredictor))
                ch.Warning("No input model file specified or model file did not contain a predictor. The model state cannot be initialized.");

            ch.Trace("Constructing data pipeline");
            ILegacyDataLoader loader = CreateRawLoader();

            // If the per-instance results are requested and there is no name column, add a GenerateNumberTransform.
            var preXf = ImplOptions.PreTransforms;
            if (!string.IsNullOrEmpty(ImplOptions.OutputDataFile))
            {
                string name = TrainUtils.MatchNameOrDefaultOrNull(ch, loader.Schema, nameof(ImplOptions.NameColumn), ImplOptions.NameColumn, DefaultColumnNames.Name);
                if (name == null)
                {
                    preXf = preXf.Concat(
                        new[]
                        {
                            new KeyValuePair<string, IComponentFactory<IDataView, IDataTransform>>(
                                "", ComponentFactoryUtils.CreateFromFunction<IDataView, IDataTransform>(
                                    (env, input) =>
                                    {
                                        var args = new GenerateNumberTransform.Options();
                                        args.Columns = new[] { new GenerateNumberTransform.Column() { Name = DefaultColumnNames.Name }, };
                                        args.UseCounter = true;
                                        return new GenerateNumberTransform(env, args, input);
                                    }))
                        }).ToArray();
                }
            }
            loader = LegacyCompositeDataLoader.Create(Host, loader, preXf);

            ch.Trace("Binding label and features columns");

            IDataView pipe = loader;
            var stratificationColumn = GetSplitColumn(ch, loader, ref pipe);
            var scorer = ImplOptions.Scorer;
            var evaluator = ImplOptions.Evaluator;

            Func<IDataView> validDataCreator = null;
            if (ImplOptions.ValidationFile != null)
            {
                validDataCreator =
                    () =>
                    {
                        // Fork the command.
                        var impl = new CrossValidationCommand(this);
                        return impl.CreateRawLoader(dataFile: ImplOptions.ValidationFile);
                    };
            }

            FoldHelper fold = new FoldHelper(Host, RegistrationName, pipe, stratificationColumn,
                ImplOptions, CreateRoleMappedData, ApplyAllTransformsToData, scorer, evaluator,
                validDataCreator, ApplyAllTransformsToData, inputPredictor, cmd, loader, !string.IsNullOrEmpty(ImplOptions.OutputDataFile));
            var tasks = fold.GetCrossValidationTasks();

            var eval = evaluator?.CreateComponent(Host) ??
                EvaluateUtils.GetEvaluator(Host, tasks[0].Result.ScoreSchema);

            // Print confusion matrix and fold results for each fold.
            for (int i = 0; i < tasks.Length; i++)
            {
                var dict = tasks[i].Result.Metrics;
                MetricWriter.PrintWarnings(ch, dict);
                eval.PrintFoldResults(ch, dict);
            }

            // Print the overall results.
            if (!TryGetOverallMetrics(tasks.Select(t => t.Result.Metrics).ToArray(), out var overallList))
                throw ch.Except("No overall metrics found");

            var overall = eval.GetOverallResults(overallList.ToArray());
            MetricWriter.PrintOverallMetrics(Host, ch, ImplOptions.SummaryFilename, overall, ImplOptions.NumFolds);
            eval.PrintAdditionalMetrics(ch, tasks.Select(t => t.Result.Metrics).ToArray());
            Dictionary<string, IDataView>[] metricValues = tasks.Select(t => t.Result.Metrics).ToArray();
            SendTelemetryMetric(metricValues);

            // Save the per-instance results.
            if (!string.IsNullOrWhiteSpace(ImplOptions.OutputDataFile))
            {
                var perInstance = EvaluateUtils.ConcatenatePerInstanceDataViews(Host, eval, ImplOptions.CollateMetrics,
                    ImplOptions.OutputExampleFoldIndex, tasks.Select(t => t.Result.PerInstanceResults).ToArray(), out var variableSizeVectorColumnNames);
                if (variableSizeVectorColumnNames.Length > 0)
                {
                    ch.Warning("Detected columns of variable length: {0}. Consider setting collateMetrics- for meaningful per-Folds results.",
                        string.Join(", ", variableSizeVectorColumnNames));
                }
                if (ImplOptions.CollateMetrics)
                {
                    ch.Assert(perInstance.Length == 1);
                    MetricWriter.SavePerInstance(Host, ch, ImplOptions.OutputDataFile, perInstance[0]);
                }
                else
                {
                    int i = 0;
                    foreach (var idv in perInstance)
                    {
                        MetricWriter.SavePerInstance(Host, ch, ConstructPerFoldName(ImplOptions.OutputDataFile, i), idv);
                        i++;
                    }
                }
            }
        }

        /// <summary>
        /// Callback from the CV method to apply the transforms from the train data to the test and/or validation data.
        /// </summary>
        private RoleMappedData ApplyAllTransformsToData(IHostEnvironment env, IChannel ch, IDataView dstData,
            RoleMappedData srcData, IDataView marker)
        {
            var pipe = ApplyTransformUtils.ApplyAllTransformsToData(env, srcData.Data, dstData, marker);
            return new RoleMappedData(pipe, srcData.Schema.GetColumnRoleNames());
        }

        /// <summary>
        /// Callback from the CV method to apply the transforms to the train data.
        /// </summary>
        private RoleMappedData CreateRoleMappedData(IHostEnvironment env, IChannel ch, IDataView data, ITrainer trainer)
        {
            foreach (var kvp in ImplOptions.Transforms)
                data = kvp.Value.CreateComponent(env, data);

            var schema = data.Schema;
            string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.LabelColumn), ImplOptions.LabelColumn, DefaultColumnNames.Label);
            string features = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.FeatureColumn), ImplOptions.FeatureColumn, DefaultColumnNames.Features);
            string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.WeightColumn), ImplOptions.WeightColumn, DefaultColumnNames.Weight);
            string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.NameColumn), ImplOptions.NameColumn, DefaultColumnNames.Name);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.GroupColumn), ImplOptions.GroupColumn, DefaultColumnNames.GroupId);

            TrainUtils.AddNormalizerIfNeeded(env, ch, trainer, ref data, features, ImplOptions.NormalizeFeatures);

            // Training pipe and examples.
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, ImplOptions.CustomColumns);

            return new RoleMappedData(data, label, features, group, weight, name, customCols);
        }

        private string GetSplitColumn(IChannel ch, IDataView input, ref IDataView output)
        {
            // The stratification column and/or group column, if they exist at all, must be present at this point.
            var schema = input.Schema;
            output = input;
            // If no stratification column was specified, but we have a group column of type Single, Double or
            // Key (contiguous) use it.
            string stratificationColumn = null;
            if (!string.IsNullOrWhiteSpace(ImplOptions.StratificationColumn))
                stratificationColumn = ImplOptions.StratificationColumn;
            else
            {
                string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(ImplOptions.GroupColumn), ImplOptions.GroupColumn, DefaultColumnNames.GroupId);
                int index;
                if (group != null && schema.TryGetColumnIndex(group, out index))
                {
                    // Check if group column key type with known cardinality.
                    var type = schema[index].Type;
                    if (type.GetKeyCount() > 0)
                        stratificationColumn = group;
                }
            }

            if (string.IsNullOrEmpty(stratificationColumn))
            {
                stratificationColumn = "StratificationColumn";
                int tmp;
                int inc = 0;
                while (input.Schema.TryGetColumnIndex(stratificationColumn, out tmp))
                    stratificationColumn = string.Format("StratificationColumn_{0:000}", ++inc);
                var keyGenArgs = new GenerateNumberTransform.Options();
                var col = new GenerateNumberTransform.Column();
                col.Name = stratificationColumn;
                keyGenArgs.Columns = new[] { col };
                output = new GenerateNumberTransform(Host, keyGenArgs, input);
            }
            else
            {
                int col;
                if (!input.Schema.TryGetColumnIndex(stratificationColumn, out col))
                    throw ch.ExceptUserArg(nameof(Arguments.StratificationColumn), "Column '{0}' does not exist", stratificationColumn);
                var type = input.Schema[col].Type;
                if (!RangeFilter.IsValidRangeFilterColumnType(ch, type))
                {
                    ch.Info("Hashing the stratification column");
                    var origStratCol = stratificationColumn;
                    int tmp;
                    int inc = 0;
                    while (input.Schema.TryGetColumnIndex(stratificationColumn, out tmp))
                        stratificationColumn = string.Format("{0}_{1:000}", origStratCol, ++inc);
                    output = new HashingEstimator(Host, origStratCol, stratificationColumn, 30).Fit(input).Transform(input);
                }
            }

            return stratificationColumn;
        }

        private bool TryGetOverallMetrics(Dictionary<string, IDataView>[] metrics, out List<IDataView> overallList)
        {
            Host.AssertNonEmpty(metrics);

            overallList = new List<IDataView>();
            for (int i = 0; i < metrics.Length; i++)
            {
                var dict = metrics[i];
                IDataView idv;
                if (!dict.TryGetValue(MetricKinds.OverallMetrics, out idv))
                    return false;
                overallList.Add(idv);
            }
            return true;
        }

        private sealed class FoldHelper
        {
            public readonly struct FoldResult
            {
                public readonly Dictionary<string, IDataView> Metrics;
                public readonly DataViewSchema ScoreSchema;
                public readonly RoleMappedData PerInstanceResults;
                public readonly RoleMappedSchema TrainSchema;

                public FoldResult(Dictionary<string, IDataView> metrics, DataViewSchema scoreSchema, RoleMappedData perInstance, RoleMappedSchema trainSchema)
                {
                    Metrics = metrics;
                    ScoreSchema = scoreSchema;
                    PerInstanceResults = perInstance;
                    TrainSchema = trainSchema;
                }
            }

            private readonly IHostEnvironment _env;
            private readonly string _registrationName;
            private readonly IDataView _inputDataView;
            private readonly string _splitColumn;
            private readonly int _numFolds;
            private readonly IComponentFactory<ITrainer> _trainer;
            private readonly IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> _scorer;
            private readonly IComponentFactory<IMamlEvaluator> _evaluator;
            private readonly IComponentFactory<ICalibratorTrainer> _calibrator;
            private readonly int _maxCalibrationExamples;
            private readonly bool _useThreads;
            private readonly bool? _cacheData;
            private readonly IPredictor _inputPredictor;
            private readonly string _cmd;
            private readonly string _outputModelFile;
            private readonly ILegacyDataLoader _loader;
            private readonly bool _savePerInstance;
            private readonly Func<IHostEnvironment, IChannel, IDataView, ITrainer, RoleMappedData> _createExamples;
            private readonly Func<IHostEnvironment, IChannel, IDataView, RoleMappedData, IDataView, RoleMappedData> _applyTransformsToTestData;
            private readonly Func<IDataView> _getValidationDataView;
            private readonly Func<IHostEnvironment, IChannel, IDataView, RoleMappedData, IDataView, RoleMappedData> _applyTransformsToValidationData;

            /// <param name="env">The environment.</param>
            /// <param name="registrationName">The registration name.</param>
            /// <param name="inputDataView">The input data view.</param>
            /// <param name="splitColumn">The column to use for splitting data into folds.</param>
            /// <param name="args">Cross validation arguments.</param>
            /// <param name="createExamples">The delegate to create RoleMappedData</param>
            /// <param name="applyTransformsToTestData">The delegate to apply the transforms from the train pipeline to the test data</param>
            /// <param name="scorer">The scorer</param>
            /// <param name="evaluator">The evaluator</param>
            /// <param name="getValidationDataView">The delegate to create validation data view</param>
            /// <param name="applyTransformsToValidationData">The delegate to apply the transforms from the train pipeline to the validation data</param>
            /// <param name="inputPredictor">The input predictor, for the continue training option</param>
            /// <param name="cmd">The command string.</param>
            /// <param name="loader">Original loader so we can construct correct pipeline for model saving.</param>
            /// <param name="savePerInstance">Whether to produce the per-instance data view.</param>
            /// <returns></returns>
            public FoldHelper(
            IHostEnvironment env,
            string registrationName,
            IDataView inputDataView,
            string splitColumn,
            Arguments args,
            Func<IHostEnvironment, IChannel, IDataView, ITrainer, RoleMappedData> createExamples,
            Func<IHostEnvironment, IChannel, IDataView, RoleMappedData, IDataView, RoleMappedData> applyTransformsToTestData,
            IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> scorer,
            IComponentFactory<IMamlEvaluator> evaluator,
            Func<IDataView> getValidationDataView = null,
            Func<IHostEnvironment, IChannel, IDataView, RoleMappedData, IDataView, RoleMappedData> applyTransformsToValidationData = null,
            IPredictor inputPredictor = null,
            string cmd = null,
            ILegacyDataLoader loader = null,
            bool savePerInstance = false)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckNonWhiteSpace(registrationName, nameof(registrationName));
                env.CheckValue(inputDataView, nameof(inputDataView));
                env.CheckValue(splitColumn, nameof(splitColumn));
                env.CheckParam(args.NumFolds > 1, nameof(args.NumFolds));
                env.CheckValue(createExamples, nameof(createExamples));
                env.CheckValue(applyTransformsToTestData, nameof(applyTransformsToTestData));
                env.CheckValue(args.Trainer, nameof(args.Trainer));
                env.CheckValueOrNull(scorer);
                env.CheckValueOrNull(evaluator);
                env.CheckValueOrNull(args.Calibrator);
                env.CheckParam(args.MaxCalibrationExamples > 0, nameof(args.MaxCalibrationExamples));
                env.CheckParam(getValidationDataView == null || applyTransformsToValidationData != null, nameof(applyTransformsToValidationData));
                env.CheckValueOrNull(inputPredictor);
                env.CheckValueOrNull(cmd);
                env.CheckValueOrNull(args.OutputModelFile);
                env.CheckValueOrNull(loader);
                _env = env;
                _registrationName = registrationName;
                _inputDataView = inputDataView;
                _splitColumn = splitColumn;
                _numFolds = args.NumFolds;
                _createExamples = createExamples;
                _applyTransformsToTestData = applyTransformsToTestData;
                _trainer = args.Trainer;
                _scorer = scorer;
                _evaluator = evaluator;
                _calibrator = args.Calibrator;
                _maxCalibrationExamples = args.MaxCalibrationExamples;
                _useThreads = args.UseThreads;
                _cacheData = args.CacheData;
                _getValidationDataView = getValidationDataView;
                _applyTransformsToValidationData = applyTransformsToValidationData;
                _inputPredictor = inputPredictor;
                _cmd = cmd;
                _outputModelFile = args.OutputModelFile;
                _loader = loader;
                _savePerInstance = savePerInstance;
            }

            private IHost GetHost()
            {
                return _env.Register(_registrationName);
            }

            /// <summary>
            /// Creates and runs tasks for each fold of cross validation. The split column is used to split the input data into folds.
            /// There are two cases:
            ///     1. The split column is R4: in this case it assumes that the values are in the interval [0,1] and will split
            ///     this interval into equal width folds. If the values are uniformly distributed it should result in balanced folds.
            ///     2. The split column is key of known cardinality: will split the whole range into equal parts to form folds. If the
            ///     keys are generated by hashing for example, it should result in balanced folds.
            /// </summary>
            /// <returns></returns>
            public Task<FoldResult>[] GetCrossValidationTasks()
            {
                var tasks = new Task<FoldResult>[_numFolds];
                for (int i = 0; i < _numFolds; i++)
                {
                    var fold = i;
                    tasks[i] = new Task<FoldResult>(() =>
                    {
                        return RunFold(fold);
                    });

                    if (_useThreads)
                        tasks[i].Start();
                    else
                        tasks[i].RunSynchronously();
                }
                Task.WaitAll(tasks);
                return tasks;
            }

            private FoldResult RunFold(int fold)
            {
                var host = GetHost();
                host.Assert(0 <= fold && fold <= _numFolds);
                // REVIEW: Make channels buffered in multi-threaded environments.
                using (var ch = host.Start($"Fold {fold}"))
                {
                    ch.Trace("Constructing trainer");
                    ITrainer trainer = _trainer.CreateComponent(host);

                    // Train pipe.
                    var trainFilter = new RangeFilter.Options();
                    trainFilter.Column = _splitColumn;
                    trainFilter.Min = (Double)fold / _numFolds;
                    trainFilter.Max = (Double)(fold + 1) / _numFolds;
                    trainFilter.Complement = true;
                    IDataView trainPipe = new RangeFilter(host, trainFilter, _inputDataView);
                    trainPipe = new OpaqueDataView(trainPipe);
                    var trainData = _createExamples(host, ch, trainPipe, trainer);

                    // Test pipe.
                    var testFilter = new RangeFilter.Options();
                    testFilter.Column = trainFilter.Column;
                    testFilter.Min = trainFilter.Min;
                    testFilter.Max = trainFilter.Max;
                    ch.Assert(!testFilter.Complement);
                    IDataView testPipe = new RangeFilter(host, testFilter, _inputDataView);
                    testPipe = new OpaqueDataView(testPipe);
                    var testData = _applyTransformsToTestData(host, ch, testPipe, trainData, trainPipe);

                    // Validation pipe and examples.
                    RoleMappedData validData = null;
                    if (_getValidationDataView != null)
                    {
                        ch.Assert(_applyTransformsToValidationData != null);
                        if (!trainer.Info.SupportsValidation)
                            ch.Warning("Trainer does not accept validation dataset.");
                        else
                        {
                            ch.Trace("Constructing the validation pipeline");
                            IDataView validLoader = _getValidationDataView();
                            var validPipe = ApplyTransformUtils.ApplyAllTransformsToData(host, _inputDataView, validLoader);
                            validPipe = new OpaqueDataView(validPipe);
                            validData = _applyTransformsToValidationData(host, ch, validPipe, trainData, trainPipe);
                        }
                    }

                    // Train.
                    var predictor = TrainUtils.Train(host, ch, trainData, trainer, validData,
                        _calibrator, _maxCalibrationExamples, _cacheData, _inputPredictor);

                    // Score.
                    ch.Trace("Scoring and evaluating");
                    ch.Assert(_scorer == null || _scorer is ICommandLineComponentFactory, "CrossValidationCommand should only be used from the command line.");
                    var bindable = ScoreUtils.GetSchemaBindableMapper(host, predictor, scorerFactorySettings: _scorer as ICommandLineComponentFactory);
                    ch.AssertValue(bindable);
                    var mapper = bindable.Bind(host, testData.Schema);
                    var scorerComp = _scorer ?? ScoreUtils.GetScorerComponent(host, mapper);
                    IDataScorerTransform scorePipe = scorerComp.CreateComponent(host, testData.Data, mapper, trainData.Schema);

                    // Save per-fold model.
                    string modelFileName = ConstructPerFoldName(_outputModelFile, fold);
                    if (modelFileName != null && _loader != null)
                    {
                        using (var file = host.CreateOutputFile(modelFileName))
                        {
                            var rmd = new RoleMappedData(
                                LegacyCompositeDataLoader.ApplyTransform(host, _loader, null, null,
                                (e, newSource) => ApplyTransformUtils.ApplyAllTransformsToData(e, trainData.Data, newSource)),
                                trainData.Schema.GetColumnRoleNames());
                            TrainUtils.SaveModel(host, ch, file, predictor, rmd, _cmd);
                        }
                    }

                    // Evaluate.
                    var eval = _evaluator?.CreateComponent(host) ??
                        EvaluateUtils.GetEvaluator(host, scorePipe.Schema);
                    // Note that this doesn't require the provided columns to exist (because of the "opt" parameter).
                    // We don't normally expect the scorer to drop columns, but if it does, we should not require
                    // all the columns in the test pipeline to still be present.
                    var dataEval = new RoleMappedData(scorePipe, testData.Schema.GetColumnRoleNames(), opt: true);

                    var dict = eval.Evaluate(dataEval);
                    RoleMappedData perInstance = null;
                    if (_savePerInstance)
                    {
                        var perInst = eval.GetPerInstanceMetrics(dataEval);
                        perInstance = new RoleMappedData(perInst, dataEval.Schema.GetColumnRoleNames(), opt: true);
                    }
                    return new FoldResult(dict, dataEval.Schema.Schema, perInstance, trainData.Schema);
                }
            }
        }
        /// <summary>
        /// Take path to expected output model file and return path to output model file for specific fold.
        /// Example: \\share\model.zip -> \\share\model.fold001.zip
        /// </summary>
        /// <param name="outputModelFile">Path to output model file</param>
        /// <param name="fold">Current fold</param>
        /// <returns>Path to output model file for specific fold</returns>
        public static string ConstructPerFoldName(string outputModelFile, int fold)
        {
            if (string.IsNullOrWhiteSpace(outputModelFile))
                return null;
            var fileName = Path.GetFileNameWithoutExtension(outputModelFile);

            return Path.Combine(Path.GetDirectoryName(outputModelFile),
             string.Format("{0}.fold{1:000}{2}", fileName, fold, Path.GetExtension(outputModelFile)));
        }
    }
}
