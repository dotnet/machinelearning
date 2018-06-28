// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(TrainCommand.Summary, typeof(TrainCommand), typeof(TrainCommand.Arguments), typeof(SignatureCommand),
    "Train Predictor", "Train")]

namespace Microsoft.ML.Runtime.Data
{
    using ColumnRole = RoleMappedSchema.ColumnRole;

    public enum NormalizeOption
    {
        No,
        Warn,
        Auto,
        Yes
    }

    public sealed class TrainCommand : DataCommand.ImplBase<TrainCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            // REVIEW: We need some better way to handle auto/none, possibly with
            // the hypothetical Maybe<string> structure.
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

            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr")]
            public SubComponent<ITrainer, SignatureTrainer> Trainer = new SubComponent<ITrainer, SignatureTrainer>("AveragedPerceptron");

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The validation data file", ShortName = "valid")]
            public string ValidationFile;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should cache input training data", ShortName = "cache")]
            public bool? CacheData;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>")]
            public SubComponent<ICalibratorTrainer, SignatureCalibrator> Calibrator = new SubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should load predictor from input model and use it as the initial model state", ShortName = "cont")]
            public bool ContinueTrain;
        }

        internal const string Summary = "Trains a predictor.";

        private readonly ComponentCatalog.LoadableClassInfo _info;
        private readonly SubComponent<ITrainer, SignatureTrainer> _trainer;

        private readonly string _labelColumn;
        private readonly string _featureColumn;
        private readonly string _groupColumn;
        private readonly string _weightColumn;
        private readonly string _nameColumn;

        public TrainCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(TrainCommand))
        {
            Host.CheckNonWhiteSpace(args.OutputModelFile, nameof(args.OutputModelFile));
            _info = TrainUtils.CheckTrainer(Host, args.Trainer, args.DataFile);
            _trainer = args.Trainer;

            _labelColumn = args.LabelColumn;
            _featureColumn = args.FeatureColumn;
            _groupColumn = args.GroupColumn;
            _weightColumn = args.WeightColumn;
            _nameColumn = args.NameColumn;
        }

        public override void Run()
        {
            string command = "Train";
            using (var ch = Host.Start(command))
            using (var server = InitServer(ch))
            {
                var settings = CmdParser.GetSettings(ch, Args, new Arguments());
                string cmd = string.Format("maml.exe {0} {1}", command, settings);
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
            SendTelemetryComponent(pipe, _trainer);
            base.SendTelemetryCore(pipe);
        }

        private void RunCore(IChannel ch, string cmd)
        {
            Host.AssertValue(ch);
            Host.AssertNonEmpty(cmd);

            ch.Trace("Constructing trainer");
            ITrainer trainer = _trainer.CreateInstance(Host);

            IPredictor inputPredictor = null;
            if (Args.ContinueTrain && !TrainUtils.TryLoadPredictor(ch, Host, Args.InputModelFile, out inputPredictor))
                ch.Warning("No input model file specified or model file did not contain a predictor. The model state cannot be initialized.");

            ch.Trace("Constructing data pipeline");
            IDataView view = CreateLoader();

            ISchema schema = view.Schema;
            var label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn), _labelColumn, DefaultColumnNames.Label);
            var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumn), _featureColumn, DefaultColumnNames.Features);
            var group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn), _groupColumn, DefaultColumnNames.GroupId);
            var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn), _weightColumn, DefaultColumnNames.Weight);
            var name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.NameColumn), _nameColumn, DefaultColumnNames.Name);

            TrainUtils.AddNormalizerIfNeeded(Host, ch, trainer, ref view, feature, Args.NormalizeFeatures);

            ch.Trace("Binding columns");

            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);
            var data = TrainUtils.CreateExamples(view, label, feature, group, weight, name, customCols);

            // REVIEW: Unify the code that creates validation examples in Train, TrainTest and CV commands.
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
                    validPipe = ApplyTransformUtils.ApplyAllTransformsToData(Host, view, validPipe);
                    validData = RoleMappedData.Create(validPipe, data.Schema.GetColumnRoleNames());
                }
            }

            var predictor = TrainUtils.Train(Host, ch, data, trainer, _info.LoadNames[0], validData,
                Args.Calibrator, Args.MaxCalibrationExamples, Args.CacheData, inputPredictor);

            using (var file = Host.CreateOutputFile(Args.OutputModelFile))
                TrainUtils.SaveModel(Host, ch, file, predictor, data, cmd);
        }
    }

    public static class TrainUtils
    {
        public static ComponentCatalog.LoadableClassInfo CheckTrainer<TSig>(IExceptionContext ectx, SubComponent<ITrainer, TSig> trainer, string dataFile)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckUserArg(trainer.IsGood(), nameof(TrainCommand.Arguments.Trainer), "A trainer is required.");

            var info = ComponentCatalog.GetLoadableClassInfo<TSig>(trainer.Kind);
            if (info == null)
                throw ectx.ExceptUserArg(nameof(TrainCommand.Arguments.Trainer), "Unknown trainer: '{0}'", trainer.Kind);
            if (!typeof(ITrainer).IsAssignableFrom(info.Type))
                throw ectx.Except("Loadable class '{0}' does not implement 'ITrainer'", info.LoadNames[0]);
            if (string.IsNullOrWhiteSpace(dataFile))
                throw ectx.ExceptUserArg(nameof(TrainCommand.Arguments.DataFile), "Data file must be defined.");
            return info;
        }

        /// <summary>
        /// If user name is null or empty, return null.
        /// Else, if the user name is found in the schema, return the user name.
        /// Else, if the user name equals the default name return null.
        /// Else, throw an error.
        /// </summary>
        public static string MatchNameOrDefaultOrNull(IExceptionContext ectx, ISchema schema, string argName, string userName, string defaultName)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckNonEmpty(argName, nameof(argName));
            ectx.CheckValueOrNull(userName);
            ectx.CheckValue(defaultName, nameof(defaultName));

            if (string.IsNullOrWhiteSpace(userName))
                return null;
            int col;
            if (schema.TryGetColumnIndex(userName, out col))
                return userName;
            if (userName == defaultName)
                return null;
#pragma warning disable TLC_ContractsNameUsesNameof
            throw ectx.ExceptUserArg(argName, $"Could not find column '{userName}'");
#pragma warning restore TLC_ContractsNameUsesNameof
        }

        public static IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, ITrainer trainer, string name,
            ICalibratorTrainerFactory calibrator, int maxCalibrationExamples)
        {
            var caliTrainer = calibrator?.CreateComponent(env);
            return TrainCore(env, ch, data, trainer, name, null, caliTrainer, maxCalibrationExamples, false);
        }

        public static IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, ITrainer trainer, string name, RoleMappedData validData,
            SubComponent<ICalibratorTrainer, SignatureCalibrator> calibrator, int maxCalibrationExamples, bool? cacheData, IPredictor inpPredictor = null)
        {
            ICalibratorTrainer caliTrainer = !calibrator.IsGood() ? null : calibrator.CreateInstance(env);
            return TrainCore(env, ch, data, trainer, name, validData, caliTrainer, maxCalibrationExamples, cacheData, inpPredictor);
        }

        private static IPredictor TrainCore(IHostEnvironment env, IChannel ch, RoleMappedData data, ITrainer trainer, string name, RoleMappedData validData,
            ICalibratorTrainer calibrator, int maxCalibrationExamples, bool? cacheData, IPredictor inpPredictor = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(data, nameof(data));
            ch.CheckValue(trainer, nameof(trainer));
            ch.CheckNonEmpty(name, nameof(name));
            ch.CheckValueOrNull(validData);
            ch.CheckValueOrNull(inpPredictor);

            var trainerRmd = trainer as ITrainer<RoleMappedData>;
            if (trainerRmd == null)
                throw ch.ExceptUserArg(nameof(TrainCommand.Arguments.Trainer), "Trainer '{0}' does not accept known training data type", name);

            Action<IChannel, ITrainer, Action<object>, object, object, object> trainCoreAction = TrainCore;
            IPredictor predictor;
            AddCacheIfWanted(env, ch, trainer, ref data, cacheData);
            ch.Trace("Training");
            if (validData != null)
                AddCacheIfWanted(env, ch, trainer, ref validData, cacheData);

            var genericExam = trainCoreAction.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(
                typeof(RoleMappedData),
                inpPredictor != null ? inpPredictor.GetType() : typeof(IPredictor));
            Action<RoleMappedData> trainExam = trainerRmd.Train;
            genericExam.Invoke(null, new object[] { ch, trainerRmd, trainExam, data, validData, inpPredictor });

            ch.Trace("Constructing predictor");
            predictor = trainerRmd.CreatePredictor();
            return CalibratorUtils.TrainCalibratorIfNeeded(env, ch, calibrator, maxCalibrationExamples, trainer, predictor, data);
        }

        public static bool CanUseValidationData(ITrainer trainer)
        {
            Contracts.CheckValue(trainer, nameof(trainer));

            if (trainer is ITrainer<RoleMappedData>)
                return trainer is IValidatingTrainer<RoleMappedData>;

            return false;
        }

        private static void TrainCore<TDataSet, TPredictor>(IChannel ch, ITrainer trainer, Action<TDataSet> train, TDataSet data, TDataSet validData = null, TPredictor predictor = null)
            where TDataSet : class
            where TPredictor : class
        {
            const string inputModelArg = nameof(TrainCommand.Arguments.InputModelFile);
            if (validData != null)
            {
                if (predictor != null)
                {
                    var incValidTrainer = trainer as IIncrementalValidatingTrainer<TDataSet, TPredictor>;
                    if (incValidTrainer != null)
                    {
                        incValidTrainer.Train(data, validData, predictor);
                        return;
                    }

                    ch.Warning("Ignoring " + inputModelArg + ": Trainer is not an incremental trainer.");
                }

                var validTrainer = trainer as IValidatingTrainer<TDataSet>;
                ch.AssertValue(validTrainer);
                validTrainer.Train(data, validData);
            }
            else
            {
                if (predictor != null)
                {
                    var incTrainer = trainer as IIncrementalTrainer<TDataSet, TPredictor>;
                    if (incTrainer != null)
                    {
                        incTrainer.Train(data, predictor);
                        return;
                    }

                    ch.Warning("Ignoring " + inputModelArg + ": Trainer is not an incremental trainer.");
                }

                train(data);
            }
        }

        public static bool TryLoadPredictor(IChannel ch, IHostEnvironment env, string inputModelFile, out IPredictor inputPredictor)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(ch);

            if (!string.IsNullOrEmpty(inputModelFile))
            {
                ch.Trace("Constructing predictor from input model");
                using (var file = env.OpenInputFile(inputModelFile))
                using (var strm = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(strm, ch))
                {
                    ch.Trace("Loading predictor");
                    return ModelLoadContext.LoadModelOrNull<IPredictor, SignatureLoadModel>(env, out inputPredictor, rep, ModelFileUtils.DirPredictor);
                }
            }

            inputPredictor = null;
            return false;
        }

        /// <summary>
        /// Save the model to the output path.
        /// The method saves the loader and the transformations of dataPipe and saves optionally predictor 
        /// and command. It also uses featureColumn, if provided, to extract feature names.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="ch">The communication channel to use.</param>
        /// <param name="output">The output file handle.</param>
        /// <param name="predictor">The predictor.</param>
        /// <param name="data">The training examples.</param>
        /// <param name="command">The command string.</param>
        public static void SaveModel(IHostEnvironment env, IChannel ch, IFileHandle output,
            IPredictor predictor, RoleMappedData data, string command = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckParam(output != null && output.CanWrite, nameof(output));
            ch.CheckValueOrNull(predictor);
            ch.CheckValue(data, nameof(data));
            ch.CheckValueOrNull(command);

            using (var stream = output.CreateWriteStream())
                SaveModel(env, ch, stream, predictor, data, command);
        }

        /// <summary>
        /// Save the model to the stream.
        /// The method saves the loader and the transformations of dataPipe and saves optionally predictor 
        /// and command. It also uses featureColumn, if provided, to extract feature names.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="ch">The communication channel to use.</param>
        /// <param name="outputStream">The output model stream.</param>
        /// <param name="predictor">The predictor.</param>
        /// <param name="data">The training examples.</param>
        /// <param name="command">The command string.</param>
        public static void SaveModel(IHostEnvironment env, IChannel ch, Stream outputStream, IPredictor predictor, RoleMappedData data, string command = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(outputStream, nameof(outputStream));
            ch.CheckValueOrNull(predictor);
            ch.CheckValue(data, nameof(data));
            ch.CheckValueOrNull(command);

            using (var ch2 = env.Start("SaveModel"))
            using (var pch = env.StartProgressChannel("Saving model"))
            {
                using (var rep = RepositoryWriter.CreateNew(outputStream, ch2))
                {
                    if (predictor != null)
                    {
                        ch2.Trace("Saving predictor");
                        ModelSaveContext.SaveModel(rep, predictor, ModelFileUtils.DirPredictor);
                    }

                    ch2.Trace("Saving loader and transformations");
                    var dataPipe = data.Data;
                    if (dataPipe is IDataLoader)
                        ModelSaveContext.SaveModel(rep, dataPipe, ModelFileUtils.DirDataLoaderModel);
                    else
                        SaveDataPipe(env, rep, dataPipe);

                    // REVIEW: Handle statistics.
                    // ModelSaveContext.SaveModel(rep, dataStats, DirDataStats);
                    if (!string.IsNullOrWhiteSpace(command))
                    {
                        using (var ent = rep.CreateEntry(ModelFileUtils.DirTrainingInfo, "Command.txt"))
                        using (var writer = Utils.OpenWriter(ent.Stream))
                            writer.WriteLine(command);
                    }
                    ModelFileUtils.SaveRoleMappings(env, ch, data.Schema, rep);

                    rep.Commit();
                }
                ch2.Done();
            }
        }

        /// <summary>
        /// Save the data pipeline defined by dataPipe. If blankLoader is true or the root IDataView is not an IDataLoader,
        /// this persists the root as a BinaryLoader having the same schema.
        /// </summary>
        public static void SaveDataPipe(IHostEnvironment env, RepositoryWriter repositoryWriter, IDataView dataPipe, bool blankLoader = false)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(repositoryWriter, nameof(repositoryWriter));
            env.CheckValue(dataPipe, nameof(dataPipe));

            IDataView pipeStart;
            var xfs = BacktrackPipe(dataPipe, out pipeStart);

            IDataLoader loader;
            Action<ModelSaveContext> saveAction;
            if (!blankLoader && (loader = pipeStart as IDataLoader) != null)
                saveAction = loader.Save;
            else
            {
                // The serialized pipe must start with a loader. If the original data view is not a loader,
                // we replace it with a binary loader with the correct schema.
                saveAction = ctx => BinaryLoader.SaveInstance(env, ctx, pipeStart.Schema);
            }

            using (var ctx = ModelFileUtils.GetDataModelSavingContext(repositoryWriter))
            {
                CompositeDataLoader.SavePipe(env, ctx, saveAction, xfs);
                ctx.Done();
            }
        }

        /// <summary>
        /// Traces back the .Source chain of the transformation pipe <paramref name="dataPipe"/> up to the moment it no longer can.
        /// Returns all the transforms of <see cref="IDataView"/> and the first data view (a non-transform). 
        /// </summary>
        /// <param name="dataPipe">The transformation pipe to traverse.</param>
        /// <param name="pipeStart">The beginning data view of the transform chain</param>
        /// <returns>The list of the transforms</returns>
        private static List<IDataTransform> BacktrackPipe(IDataView dataPipe, out IDataView pipeStart)
        {
            Contracts.AssertValue(dataPipe);

            var transforms = new List<IDataTransform>();
            while (dataPipe is IDataTransform xf)
            {
                // REVIEW: a malicious user could construct a loop in the Source chain, that would
                // cause this method to iterate forever (and throw something when the list overflows). There's 
                // no way to insulate from ALL malicious behavior.
                transforms.Add(xf);
                dataPipe = xf.Source;
                Contracts.AssertValue(dataPipe);
            }

            pipeStart = dataPipe;
            transforms.Reverse();
            return transforms;
        }

        // Returns true if a normalizer was added.
        public static bool AddNormalizerIfNeeded(IHostEnvironment env, IChannel ch, ITrainer trainer, ref IDataView view, string featureColumn, NormalizeOption autoNorm)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(trainer, nameof(trainer));
            ch.CheckValue(view, nameof(view));
            ch.CheckValueOrNull(featureColumn);
            ch.CheckUserArg(Enum.IsDefined(typeof(NormalizeOption), autoNorm), nameof(TrainCommand.Arguments.NormalizeFeatures),
                "Normalize option is invalid. Specify one of 'norm=No', 'norm=Warn', 'norm=Auto', or 'norm=Yes'.");

            if (autoNorm == NormalizeOption.No)
            {
                ch.Info("Not adding a normalizer.");
                return false;
            }

            if (string.IsNullOrEmpty(featureColumn))
                return false;

            int featCol;
            var schema = view.Schema;
            if (schema.TryGetColumnIndex(featureColumn, out featCol))
            {
                if (autoNorm != NormalizeOption.Yes)
                {
                    DvBool isNormalized = DvBool.False;
                    if (trainer.NeedNormalization() != true || schema.IsNormalized(featCol))
                    {
                        ch.Info("Not adding a normalizer.");
                        return false;
                    }
                    if (autoNorm == NormalizeOption.Warn)
                    {
                        ch.Warning("A normalizer is needed for this trainer. Either add a normalizing transform or use the 'norm=Auto', 'norm=Yes' or 'norm=No' options.");
                        return false;
                    }
                }
                ch.Info("Automatically adding a MinMax normalization transform, use 'norm=Warn' or 'norm=No' to turn this behavior off.");
                // REVIEW: This verbose constructor should be replaced with zeahmed's enhancements once #405 is committed.
                IDataView ApplyNormalizer(IHostEnvironment innerEnv, IDataView input)
                    => NormalizeTransform.Create(innerEnv, new NormalizeTransform.MinMaxArguments()
                    {
                        Column = new[] { new NormalizeTransform.AffineColumn { Source = featureColumn, Name = featureColumn } }
                    }, input);

                if (view is IDataLoader loader)
                    view = CompositeDataLoader.ApplyTransform(env, loader, tag: null, creationArgs: null, ApplyNormalizer);
                else
                    view = ApplyNormalizer(env, view);
                return true;
            }
            return false;
        }

        private static bool AddCacheIfWanted(IHostEnvironment env, IChannel ch, ITrainer trainer, ref RoleMappedData data, bool? cacheData)
        {
            Contracts.AssertValue(env, nameof(env));
            env.AssertValue(ch, nameof(ch));
            ch.AssertValue(trainer, nameof(trainer));
            ch.AssertValue(data, nameof(data));

            ITrainerEx trainerEx = trainer as ITrainerEx;
            bool shouldCache = cacheData ?? (!(data.Data is BinaryLoader) && (trainerEx == null || trainerEx.WantCaching));

            if (shouldCache)
            {
                ch.Trace("Caching");
                var prefetch = data.Schema.GetColumnRoles().Select(kc => kc.Value.Index).ToArray();
                var cacheView = new CacheDataView(env, data.Data, prefetch);
                // Because the prefetching worked, we know that these are valid columns.
                data = RoleMappedData.Create(cacheView, data.Schema.GetColumnRoleNames());
            }
            else
                ch.Trace("Not caching");
            return shouldCache;
        }

        public static IEnumerable<KeyValuePair<ColumnRole, string>> CheckAndGenerateCustomColumns(IExceptionContext ectx, KeyValuePair<string, string>[] customColumnArg)
        {
            Contracts.CheckValueOrNull(ectx);

            if (customColumnArg == null)
                return Enumerable.Empty<KeyValuePair<ColumnRole, string>>();
            foreach (var kindName in customColumnArg)
            {
                ectx.CheckUserArg(!string.IsNullOrWhiteSpace(kindName.Value), nameof(TrainCommand.Arguments.CustomColumn), "Names for columns with custom kind must not be empty");
                if (string.IsNullOrWhiteSpace(kindName.Key))
                    throw ectx.ExceptUserArg(nameof(TrainCommand.Arguments.CustomColumn), "Custom column with name '{0}' needs a kind. Use col[<Kind>]={0}", kindName.Value);
            }
            return customColumnArg.Select(kindName => new ColumnRole(kindName.Key).Bind(kindName.Value));
        }

        /// <summary>
        /// Given a schema and a bunch of column names, create the BoundSchema object. Any or all of the column
        /// names may be null or whitespace, in which case they are ignored. Any columns that are specified but not
        /// valid columns of the schema are also ignored.
        /// </summary>
        public static RoleMappedSchema CreateRoleMappedSchemaOpt(ISchema schema, string feature, string group, IEnumerable<KeyValuePair<ColumnRole, string>> custom = null)
        {
            Contracts.CheckValueOrNull(feature);
            Contracts.CheckValueOrNull(custom);

            var list = new List<KeyValuePair<ColumnRole, string>>();
            if (!string.IsNullOrWhiteSpace(feature))
                list.Add(ColumnRole.Feature.Bind(feature));
            if (!string.IsNullOrWhiteSpace(group))
                list.Add(ColumnRole.Group.Bind(group));
            if (custom != null)
                list.AddRange(custom);

            return RoleMappedSchema.CreateOpt(schema, list);
        }

        /// <summary>
        /// Given a view and a bunch of column names, create the RoleMappedData object. Any or all of the column
        /// names may be null or whitespace, in which case they are ignored. Any columns that are specified must
        /// be valid columns of the schema.
        /// </summary>
        public static RoleMappedData CreateExamples(IDataView view, string label, string feature,
            string group = null, string weight = null, string name = null,
            IEnumerable<KeyValuePair<ColumnRole, string>> custom = null)
        {
            Contracts.CheckValueOrNull(label);
            Contracts.CheckValueOrNull(feature);
            Contracts.CheckValueOrNull(group);
            Contracts.CheckValueOrNull(weight);
            Contracts.CheckValueOrNull(name);
            Contracts.CheckValueOrNull(custom);

            var list = new List<KeyValuePair<ColumnRole, string>>();
            if (!string.IsNullOrWhiteSpace(label))
                list.Add(ColumnRole.Label.Bind(label));
            if (!string.IsNullOrWhiteSpace(feature))
                list.Add(ColumnRole.Feature.Bind(feature));
            if (!string.IsNullOrWhiteSpace(group))
                list.Add(ColumnRole.Group.Bind(group));
            if (!string.IsNullOrWhiteSpace(weight))
                list.Add(ColumnRole.Weight.Bind(weight));
            if (!string.IsNullOrWhiteSpace(name))
                list.Add(ColumnRole.Name.Bind(name));
            if (custom != null)
                list.AddRange(custom);

            return RoleMappedData.Create(view, list);
        }

        /// <summary>
        /// Given a view and a bunch of column names, create the RoleMappedData object. Any or all of the column
        /// names may be null or whitespace, in which case they are ignored. Any columns that are specified but not
        /// valid columns of the schema are also ignored.
        /// </summary>
        public static RoleMappedData CreateExamplesOpt(IDataView view, string label, string feature,
            string group = null, string weight = null, string name = null,
            IEnumerable<KeyValuePair<ColumnRole, string>> custom = null)
        {
            Contracts.CheckValueOrNull(label);
            Contracts.CheckValueOrNull(feature);
            Contracts.CheckValueOrNull(group);
            Contracts.CheckValueOrNull(weight);
            Contracts.CheckValueOrNull(name);
            Contracts.CheckValueOrNull(custom);

            var list = new List<KeyValuePair<ColumnRole, string>>();
            if (!string.IsNullOrWhiteSpace(label))
                list.Add(ColumnRole.Label.Bind(label));
            if (!string.IsNullOrWhiteSpace(feature))
                list.Add(ColumnRole.Feature.Bind(feature));
            if (!string.IsNullOrWhiteSpace(group))
                list.Add(ColumnRole.Group.Bind(group));
            if (!string.IsNullOrWhiteSpace(weight))
                list.Add(ColumnRole.Weight.Bind(weight));
            if (!string.IsNullOrWhiteSpace(name))
                list.Add(ColumnRole.Name.Bind(name));
            if (custom != null)
                list.AddRange(custom);

            return RoleMappedData.CreateOpt(view, list);
        }

        private static KeyValuePair<ColumnRole, T> Pair<T>(ColumnRole kind, T value)
        {
            return new KeyValuePair<ColumnRole, T>(kind, value);
        }
    }
}
