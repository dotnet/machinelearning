// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(SchemaBindablePipelineEnsembleBase), null, typeof(SignatureLoadModel),
    SchemaBindablePipelineEnsembleBase.UserName, SchemaBindablePipelineEnsembleBase.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble
{
    /// <summary>
    /// This class represents an ensemble predictor, where each predictor has its own featurization pipeline. It is
    /// useful for the distributed training scenario, where the featurization includes trainable transforms (for example,
    /// categorical transform, or normalization).
    /// </summary>
    public abstract class SchemaBindablePipelineEnsembleBase : ICanGetTrainingLabelNames, ICanSaveModel,
        ISchemaBindableMapper, ICanSaveSummary, ICanGetSummaryInKeyValuePairs
    {
        private abstract class BoundBase : ISchemaBoundRowMapper
        {
            protected readonly SchemaBindablePipelineEnsembleBase Parent;
            private readonly HashSet<int> _inputColIndices;

            protected readonly ISchemaBoundRowMapper[] Mappers;
            protected readonly IRowToRowMapper[] BoundPipelines;
            protected readonly int[] ScoreCols;

            public ISchemaBindableMapper Bindable => Parent;
            public RoleMappedSchema InputSchema { get; }
            public ISchema OutputSchema { get; }

            public BoundBase(SchemaBindablePipelineEnsembleBase parent, RoleMappedSchema schema)
            {
                Parent = parent;
                InputSchema = schema;
                OutputSchema = new ScoreMapperSchema(Parent.ScoreType, Parent._scoreColumnKind);
                _inputColIndices = new HashSet<int>();
                for (int i = 0; i < Parent._inputCols.Length; i++)
                {
                    var name = Parent._inputCols[i];
                    if (!InputSchema.Schema.TryGetColumnIndex(name, out int col))
                        throw Parent.Host.Except("Schema does not contain required input column '{0}'", name);
                    _inputColIndices.Add(col);
                }

                Mappers = new ISchemaBoundRowMapper[Parent.PredictorModels.Length];
                BoundPipelines = new IRowToRowMapper[Parent.PredictorModels.Length];
                ScoreCols = new int[Parent.PredictorModels.Length];
                for (int i = 0; i < Mappers.Length; i++)
                {
                    // Get the RoleMappedSchema to pass to the predictor.
                    var emptyDv = new EmptyDataView(Parent.Host, schema.Schema);
                    Parent.PredictorModels[i].PrepareData(Parent.Host, emptyDv, out RoleMappedData rmd, out IPredictor predictor);

                    // Get the predictor as a bindable mapper, and bind it to the RoleMappedSchema found above.
                    var bindable = ScoreUtils.GetSchemaBindableMapper(Parent.Host, Parent.PredictorModels[i].Predictor, null);
                    Mappers[i] = bindable.Bind(Parent.Host, rmd.Schema) as ISchemaBoundRowMapper;
                    if (Mappers[i] == null)
                        throw Parent.Host.Except("Predictor {0} is not a row to row mapper", i);

                    // Make sure there is a score column, and remember its index.
                    if (!Mappers[i].OutputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out ScoreCols[i]))
                        throw Parent.Host.Except("Predictor {0} does not contain a score column", i);

                    // Get the pipeline.
                    var dv = new EmptyDataView(Parent.Host, schema.Schema);
                    var tm = new TransformModel(Parent.Host, dv, dv);
                    var pipeline = Parent.PredictorModels[i].TransformModel.Apply(Parent.Host, tm);
                    BoundPipelines[i] = pipeline.AsRowToRowMapper(Parent.Host);
                    if (BoundPipelines[i] == null)
                        throw Parent.Host.Except("Transform pipeline {0} contains transforms that do not implement IRowToRowMapper", i);
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => _inputColIndices.Contains(col);
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield break;
            }

            public IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                return new SimpleRow(OutputSchema, input, new[] { CreateScoreGetter(input, predicate, out disposer) });
            }

            public abstract Delegate CreateScoreGetter(IRow input, Func<int, bool> mapperPredicate, out Action disposer);
        }

        // A generic base class for pipeline ensembles. This class contains the combiner.
        private abstract class SchemaBindablePipelineEnsemble<T> : SchemaBindablePipelineEnsembleBase, IPredictorProducing<T>
        {
            protected sealed class Bound : BoundBase
            {
                private readonly IOutputCombiner<T> _combiner;

                public Bound(SchemaBindablePipelineEnsemble<T> parent, RoleMappedSchema schema)
                    : base(parent, schema)
                {
                    _combiner = parent.Combiner;
                }

                public override Delegate CreateScoreGetter(IRow input, Func<int, bool> mapperPredicate, out Action disposer)
                {
                    disposer = null;

                    if (!mapperPredicate(0))
                        return null;

                    var getters = new ValueGetter<T>[Mappers.Length];
                    for (int i = 0; i < Mappers.Length; i++)
                    {
                        // First get the output row from the pipelines. The input predicate of the predictor
                        // is the output predicate of the pipeline.
                        var inputPredicate = Mappers[i].GetDependencies(mapperPredicate);
                        var pipelineRow = BoundPipelines[i].GetRow(input, inputPredicate, out Action disp);
                        disposer += disp;

                        // Next we get the output row from the predictors. We activate the score column as output predicate.
                        var predictorRow = Mappers[i].GetOutputRow(pipelineRow, col => col == ScoreCols[i], out disp);
                        disposer += disp;
                        getters[i] = predictorRow.GetGetter<T>(ScoreCols[i]);
                    }

                    var comb = _combiner.GetCombiner();
                    var buffer = new T[Mappers.Length];
                    ValueGetter<T> scoreGetter =
                        (ref T dst) =>
                        {
                            for (int i = 0; i < Mappers.Length; i++)
                                getters[i](ref buffer[i]);
                            comb(ref dst, buffer, null);
                        };
                    return scoreGetter;
                }

                public ValueGetter<Single> GetLabelGetter(IRow input, int i, out Action disposer)
                {
                    Parent.Host.Assert(0 <= i && i < Mappers.Length);
                    Parent.Host.Check(Mappers[i].InputSchema.Label != null, "Mapper was not trained using a label column");

                    // The label should be in the output row of the i'th pipeline
                    var pipelineRow = BoundPipelines[i].GetRow(input, col => col == Mappers[i].InputSchema.Label.Index, out disposer);
                    return RowCursorUtils.GetLabelGetter(pipelineRow, Mappers[i].InputSchema.Label.Index);
                }

                public ValueGetter<Single> GetWeightGetter(IRow input, int i, out Action disposer)
                {
                    Parent.Host.Assert(0 <= i && i < Mappers.Length);

                    if (Mappers[i].InputSchema.Weight == null)
                    {
                        ValueGetter<Single> weight = (ref Single dst) => dst = 1;
                        disposer = null;
                        return weight;
                    }
                    // The weight should be in the output row of the i'th pipeline if it exists.
                    var inputPredicate = Mappers[i].GetDependencies(col => col == Mappers[i].InputSchema.Weight.Index);
                    var pipelineRow = BoundPipelines[i].GetRow(input, inputPredicate, out disposer);
                    return pipelineRow.GetGetter<Single>(Mappers[i].InputSchema.Weight.Index);
                }
            }

            protected readonly IOutputCombiner<T> Combiner;

            protected SchemaBindablePipelineEnsemble(IHostEnvironment env, IPredictorModel[] predictors,
                IOutputCombiner<T> combiner, string registrationName, string scoreColumnKind)
                    : base(env, predictors, registrationName, scoreColumnKind)
            {
                Combiner = combiner;
            }

            protected SchemaBindablePipelineEnsemble(IHostEnvironment env, ModelLoadContext ctx, string scoreColumnKind)
                    : base(env, ctx, scoreColumnKind)
            {
                // *** Binary format ***
                // <base>
                // The combiner

                ctx.LoadModel<IOutputCombiner<T>, SignatureLoadModel>(Host, out Combiner, "Combiner");
            }

            protected override void SaveCore(ModelSaveContext ctx)
            {
                Host.AssertValue(ctx);

                // *** Binary format ***
                // <base>
                // The combiner

                ctx.SaveModel(Combiner, "Combiner");
            }

            public override ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
            {
                return new Bound(this, schema);
            }
        }

        // This is an implementation of pipeline ensembles that combines scores of type float (regression and anomaly detection).
        private sealed class ImplOne : SchemaBindablePipelineEnsemble<Single>
        {
            protected override ColumnType ScoreType => NumberType.R4;

            public override PredictionKind PredictionKind
            {
                get
                {
                    if (_scoreColumnKind == MetadataUtils.Const.ScoreColumnKind.Regression)
                        return PredictionKind.Regression;
                    if (_scoreColumnKind == MetadataUtils.Const.ScoreColumnKind.AnomalyDetection)
                        return PredictionKind.AnomalyDetection;
                    throw Host.Except("Unknown prediction kind");
                }
            }

            public ImplOne(IHostEnvironment env, IPredictorModel[] predictors, IRegressionOutputCombiner combiner, string scoreColumnKind)
                : base(env, predictors, combiner, LoaderSignature, scoreColumnKind)
            {
            }

            public ImplOne(IHostEnvironment env, ModelLoadContext ctx, string scoreColumnKind)
                : base(env, ctx, scoreColumnKind)
            {
            }
        }

        // This is an implementation of pipeline ensemble that combines scores of type vectors of float (multi-class).
        private sealed class ImplVec : SchemaBindablePipelineEnsemble<VBuffer<Single>>
        {
            protected override ColumnType ScoreType { get { return _scoreType; } }

            public override PredictionKind PredictionKind
            {
                get
                {
                    if (_scoreColumnKind == MetadataUtils.Const.ScoreColumnKind.MultiClassClassification)
                        return PredictionKind.MultiClassClassification;
                    throw Host.Except("Unknown prediction kind");
                }
            }

            private readonly VectorType _scoreType;

            public ImplVec(IHostEnvironment env, IPredictorModel[] predictors, IMultiClassOutputCombiner combiner)
                : base(env, predictors, combiner, LoaderSignature, MetadataUtils.Const.ScoreColumnKind.MultiClassClassification)
            {
                int classCount = CheckLabelColumn(Host, predictors, false);
                _scoreType = new VectorType(NumberType.R4, classCount);
            }

            public ImplVec(IHostEnvironment env, ModelLoadContext ctx, string scoreColumnKind)
                : base(env, ctx, scoreColumnKind)
            {
                int classCount = CheckLabelColumn(Host, PredictorModels, false);
                _scoreType = new VectorType(NumberType.R4, classCount);
            }
        }

        // This is an implementation of pipeline ensembles that combines scores of type float, and also provides calibration (for binary classification).
        private sealed class ImplOneWithCalibrator : SchemaBindablePipelineEnsemble<Single>, ISelfCalibratingPredictor
        {
            protected override ColumnType ScoreType { get { return NumberType.R4; } }

            public override PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }

            public ImplOneWithCalibrator(IHostEnvironment env, IPredictorModel[] predictors, IBinaryOutputCombiner combiner)
                : base(env, predictors, combiner, LoaderSignature, MetadataUtils.Const.ScoreColumnKind.BinaryClassification)
            {
                Host.Assert(_scoreColumnKind == MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
                CheckBinaryLabel(true, Host, PredictorModels);
            }

            public ImplOneWithCalibrator(IHostEnvironment env, ModelLoadContext ctx, string scoreColumnKind)
                : base(env, ctx, scoreColumnKind)
            {
                Host.Assert(_scoreColumnKind == MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
                CheckBinaryLabel(false, Host, PredictorModels);
            }

            private static void CheckBinaryLabel(bool user, IHostEnvironment env, IPredictorModel[] predictors)
            {
                int classCount = CheckLabelColumn(env, predictors, true);
                if (classCount != 2)
                {
                    var error = string.Format("Expected label to have exactly 2 classes, instead has {0}", classCount);
                    throw user ? env.ExceptParam(nameof(predictors), error) : env.ExceptDecode(error);
                }
            }

            public IPredictor Calibrate(IChannel ch, IDataView data, ICalibratorTrainer caliTrainer, int maxRows)
            {
                Host.CheckValue(ch, nameof(ch));
                ch.CheckValue(data, nameof(data));
                ch.CheckValue(caliTrainer, nameof(caliTrainer));

                if (caliTrainer.NeedsTraining)
                {
                    var bound = new Bound(this, new RoleMappedSchema(data.Schema));
                    using (var curs = data.GetRowCursor(col => true))
                    {
                        var scoreGetter = (ValueGetter<Single>)bound.CreateScoreGetter(curs, col => true, out Action disposer);

                        // We assume that we can use the label column of the first predictor, since if the labels are not identical
                        // then the whole model is garbage anyway.
                        var labelGetter = bound.GetLabelGetter(curs, 0, out Action disp);
                        disposer += disp;
                        var weightGetter = bound.GetWeightGetter(curs, 0, out disp);
                        disposer += disp;
                        try
                        {
                            int num = 0;
                            while (curs.MoveNext())
                            {
                                Single label = 0;
                                labelGetter(ref label);
                                if (!FloatUtils.IsFinite(label))
                                    continue;
                                Single score = 0;
                                scoreGetter(ref score);
                                if (!FloatUtils.IsFinite(score))
                                    continue;
                                Single weight = 0;
                                weightGetter(ref weight);
                                if (!FloatUtils.IsFinite(weight))
                                    continue;

                                caliTrainer.ProcessTrainingExample(score, label > 0, weight);

                                if (maxRows > 0 && ++num >= maxRows)
                                    break;
                            }
                        }
                        finally
                        {
                            disposer?.Invoke();
                        }
                    }
                }

                var calibrator = caliTrainer.FinishTraining(ch);
                return CalibratorUtils.CreateCalibratedPredictor(Host, this, calibrator);
            }
        }

        private readonly string[] _inputCols;

        protected readonly IHost Host;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PIPELNEN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Save predictor models in a subdirectory
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }
        public const string UserName = "Pipeline Ensemble";
        public const string LoaderSignature = "PipelineEnsemble";

        private readonly string _scoreColumnKind;

        protected abstract ColumnType ScoreType { get; }

        public abstract PredictionKind PredictionKind { get; }

        internal IPredictorModel[] PredictorModels { get; }

        private SchemaBindablePipelineEnsembleBase(IHostEnvironment env, IPredictorModel[] predictors, string registrationName, string scoreColumnKind)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            Host.CheckNonEmpty(predictors, nameof(predictors));
            Host.CheckNonWhiteSpace(scoreColumnKind, nameof(scoreColumnKind));

            PredictorModels = predictors;
            _scoreColumnKind = scoreColumnKind;

            HashSet<string> inputCols = null;
            for (int i = 0; i < predictors.Length; i++)
            {
                var predModel = predictors[i];

                // Get the input column names.
                var inputSchema = predModel.TransformModel.InputSchema;
                if (inputCols == null)
                {
                    inputCols = new HashSet<string>();
                    for (int j = 0; j < inputSchema.ColumnCount; j++)
                    {
                        if (inputSchema.IsHidden(j))
                            continue;
                        inputCols.Add(inputSchema.GetColumnName(j));
                    }
                    _inputCols = inputCols.ToArray();
                }
                else
                {
                    int nonHiddenCols = 0;
                    for (int j = 0; j < inputSchema.ColumnCount; j++)
                    {
                        if (inputSchema.IsHidden(j))
                            continue;
                        var name = inputSchema.GetColumnName(j);
                        if (!inputCols.Contains(name))
                            throw Host.Except("Inconsistent schemas: Some schemas do not contain the column '{0}'", name);
                        nonHiddenCols++;
                    }
                    Host.Check(nonHiddenCols == _inputCols.Length,
                        "Inconsistent schemas: not all schemas have the same number of columns");
                }
            }
        }

        protected SchemaBindablePipelineEnsembleBase(IHostEnvironment env, ModelLoadContext ctx, string scoreColumnKind)
        {
            Host = env.Register(LoaderSignature);
            Host.AssertNonEmpty(scoreColumnKind);

            _scoreColumnKind = scoreColumnKind;

            // *** Binary format ***
            // int: id of _scoreColumnKind (loaded in the Create method)
            // int: number of predictors
            // The predictor models
            // int: the number of input columns
            // for each input column:
            //   int: id of the column name

            var length = ctx.Reader.ReadInt32();
            Host.CheckDecode(length > 0);
            PredictorModels = new IPredictorModel[length];
            for (int i = 0; i < PredictorModels.Length; i++)
            {
                string dir =
                    ctx.Header.ModelVerWritten == 0x00010001
                        ? "PredictorModels"
                        : Path.Combine(ctx.Directory, "PredictorModels");
                using (var ent = ctx.Repository.OpenEntry(dir, $"PredictorModel_{i:000}"))
                    PredictorModels[i] = new PredictorModel(Host, ent.Stream);
            }

            length = ctx.Reader.ReadInt32();
            Host.CheckDecode(length >= 0);
            _inputCols = new string[length];
            for (int i = 0; i < length; i++)
                _inputCols[i] = ctx.LoadNonEmptyString();
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: id of _scoreColumnKind (loaded in the Create method)
            // int: number of predictors
            // The predictor models
            // int: the number of input columns
            // for each input column:
            //   int: id of the column name

            ctx.SaveNonEmptyString(_scoreColumnKind);

            Host.AssertNonEmpty(PredictorModels);
            ctx.Writer.Write(PredictorModels.Length);

            for (int i = 0; i < PredictorModels.Length; i++)
            {
                var dir = Path.Combine(ctx.Directory, "PredictorModels");
                using (var ent = ctx.Repository.CreateEntry(dir, $"PredictorModel_{i:000}"))
                    PredictorModels[i].Save(Host, ent.Stream);
            }

            Contracts.AssertValue(_inputCols);
            ctx.Writer.Write(_inputCols.Length);
            foreach (var name in _inputCols)
                ctx.SaveNonEmptyString(name);

            SaveCore(ctx);
        }

        protected abstract void SaveCore(ModelSaveContext ctx);

        public static SchemaBindablePipelineEnsembleBase Create(IHostEnvironment env, IPredictorModel[] predictors, IOutputCombiner combiner, string scoreColumnKind)
        {
            switch (scoreColumnKind)
            {
                case MetadataUtils.Const.ScoreColumnKind.BinaryClassification:
                    var binaryCombiner = combiner as IBinaryOutputCombiner;
                    if (binaryCombiner == null)
                        throw env.Except("Combiner type incompatible with score column kind");
                    return new ImplOneWithCalibrator(env, predictors, binaryCombiner);
                case MetadataUtils.Const.ScoreColumnKind.Regression:
                case MetadataUtils.Const.ScoreColumnKind.AnomalyDetection:
                    var regressionCombiner = combiner as IRegressionOutputCombiner;
                    if (regressionCombiner == null)
                        throw env.Except("Combiner type incompatible with score column kind");
                    return new ImplOne(env, predictors, regressionCombiner, scoreColumnKind);
                case MetadataUtils.Const.ScoreColumnKind.MultiClassClassification:
                    var vectorCombiner = combiner as IMultiClassOutputCombiner;
                    if (vectorCombiner == null)
                        throw env.Except("Combiner type incompatible with score column kind");
                    return new ImplVec(env, predictors, vectorCombiner);
                default:
                    throw env.Except("Unknown score kind");
            }
        }

        public static SchemaBindablePipelineEnsembleBase Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var scoreColumnKind = ctx.LoadNonEmptyString();
            switch (scoreColumnKind)
            {
                case MetadataUtils.Const.ScoreColumnKind.BinaryClassification:
                    return new ImplOneWithCalibrator(env, ctx, scoreColumnKind);
                case MetadataUtils.Const.ScoreColumnKind.Regression:
                case MetadataUtils.Const.ScoreColumnKind.AnomalyDetection:
                    return new ImplOne(env, ctx, scoreColumnKind);
                case MetadataUtils.Const.ScoreColumnKind.MultiClassClassification:
                    return new ImplVec(env, ctx, scoreColumnKind);
                default:
                    throw env.Except("Unknown score kind");
            }
        }

        public abstract ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema);

        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            for (int i = 0; i < PredictorModels.Length; i++)
            {
                writer.WriteLine("Partition model {0} summary:", i);

                if (!(PredictorModels[i].Predictor is ICanSaveSummary summaryModel))
                {
                    writer.WriteLine("Model of type {0}", PredictorModels[i].Predictor.GetType().Name);
                    continue;
                }

                // Load the feature names for the i'th model.
                var dv = new EmptyDataView(Host, PredictorModels[i].TransformModel.InputSchema);
                PredictorModels[i].PrepareData(Host, dv, out RoleMappedData rmd, out IPredictor pred);
                summaryModel.SaveSummary(writer, rmd.Schema);
            }
        }

        // Checks that the predictors have matching label columns, and returns the number of classes in all predictors.
        protected static int CheckLabelColumn(IHostEnvironment env, IPredictorModel[] models, bool isBinary)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(models, nameof(models));

            var model = models[0];
            var edv = new EmptyDataView(env, model.TransformModel.InputSchema);
            model.PrepareData(env, edv, out RoleMappedData rmd, out IPredictor pred);
            var labelInfo = rmd.Schema.Label;
            if (labelInfo == null)
                throw env.Except("Training schema for model 0 does not have a label column");

            var labelType = rmd.Schema.Schema.GetColumnType(rmd.Schema.Label.Index);
            if (!labelType.IsKey)
                return CheckNonKeyLabelColumnCore(env, pred, models, isBinary, labelType);

            if (isBinary && labelType.KeyCount != 2)
                throw env.Except("Label is not binary");
            var schema = rmd.Schema.Schema;
            var mdType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, labelInfo.Index);
            if (mdType == null || !mdType.IsKnownSizeVector)
                throw env.Except("Label column of type key must have a vector of key values metadata");

            return Utils.MarshalInvoke(CheckKeyLabelColumnCore<int>, mdType.ItemType.RawType, env, models, labelType.AsKey, schema, labelInfo.Index, mdType);
        }

        // When the label column is not a key, we check that the number of classes is the same for all the predictors, by checking the
        // OutputType property of the IValueMapper.
        // If any of the predictors do not implement IValueMapper we throw an exception. Returns the class count.
        private static int CheckNonKeyLabelColumnCore(IHostEnvironment env, IPredictor pred, IPredictorModel[] models, bool isBinary, ColumnType labelType)
        {
            env.Assert(!labelType.IsKey);
            env.AssertNonEmpty(models);

            if (isBinary)
                return 2;

            // The label is numeric, we just have to check that the number of classes is the same.
            if (!(pred is IValueMapper vm))
                throw env.Except("Cannot determine the number of classes the predictor outputs");
            var classCount = vm.OutputType.VectorSize;

            for (int i = 1; i < models.Length; i++)
            {
                var model = models[i];
                var edv = new EmptyDataView(env, model.TransformModel.InputSchema);
                model.PrepareData(env, edv, out RoleMappedData rmd, out pred);
                vm = pred as IValueMapper;
                if (vm.OutputType.VectorSize != classCount)
                    throw env.Except("Label of model {0} has different number of classes than model 0", i);
            }
            return classCount;
        }

        // Checks that all the label columns of the model have the same key type as their label column - including the same
        // cardinality and the same key values, and returns the cardinality of the label column key.
        private static int CheckKeyLabelColumnCore<T>(IHostEnvironment env, IPredictorModel[] models, KeyType labelType, ISchema schema, int labelIndex, ColumnType keyValuesType)
            where T : IEquatable<T>
        {
            env.Assert(keyValuesType.ItemType.RawType == typeof(T));
            env.AssertNonEmpty(models);
            var labelNames = default(VBuffer<T>);
            schema.GetMetadata(MetadataUtils.Kinds.KeyValues, labelIndex, ref labelNames);
            var classCount = labelNames.Length;

            var curLabelNames = default(VBuffer<T>);
            for (int i = 1; i < models.Length; i++)
            {
                var model = models[i];
                var edv = new EmptyDataView(env, model.TransformModel.InputSchema);
                model.PrepareData(env, edv, out RoleMappedData rmd, out IPredictor pred);
                var labelInfo = rmd.Schema.Label;
                if (labelInfo == null)
                    throw env.Except("Training schema for model {0} does not have a label column", i);

                var curLabelType = rmd.Schema.Schema.GetColumnType(rmd.Schema.Label.Index);
                if (!labelType.Equals(curLabelType.AsKey))
                    throw env.Except("Label column of model {0} has different type than model 0", i);

                var mdType = rmd.Schema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, labelInfo.Index);
                if (!mdType.Equals(keyValuesType))
                    throw env.Except("Label column of model {0} has different key value type than model 0", i);
                rmd.Schema.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, labelInfo.Index, ref curLabelNames);
                if (!AreEqual(ref labelNames, ref curLabelNames))
                    throw env.Except("Label of model {0} has different values than model 0", i);
            }
            return classCount;
        }

        private static bool AreEqual<T>(ref VBuffer<T> v1, ref VBuffer<T> v2)
            where T : IEquatable<T>
        {
            if (v1.Length != v2.Length)
                return false;
            return v1.DenseValues().Zip(v2.DenseValues(), (x1, x2) => x1.Equals(x2)).All(b => b);
        }

        /// <summary>
        /// This method outputs a Key-Value Pair (kvp) per model in the ensemble.
        ///   * The key is the model number such as "Partition model 0 summary". If the model implements <see cref="ICanSaveSummary"/>
        ///     then this string is followed by the first line of the model summary (the first line contains a description specific to the
        ///     model kind, such as "Feature gains" for FastTree or "Feature weights" for linear).
        ///   * The value:
        ///       - If the model implements <see cref="ICanGetSummaryInKeyValuePairs"/> then the value is the list of Key-Value pairs
        ///         containing the detailed summary for that model (for example, linear models have a list containing kvps where the keys
        ///         are the feature names and the values are the weights. FastTree has a similar list with the feature gains as values).
        ///       - If the model does not implement <see cref="ICanGetSummaryInKeyValuePairs"/> but does implement <see cref="ICanSaveSummary"/>,
        ///         the value is a string containing the summary of that model.
        ///       - If neither of those interfaces are implemented then the value is a string containing the name of the type of model.
        /// </summary>
        /// <returns></returns>
        public IList<KeyValuePair<string, object>> GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValueOrNull(schema);

            var list = new List<KeyValuePair<string, object>>();

            var sb = new StringBuilder();
            for (int i = 0; i < PredictorModels.Length; i++)
            {
                var key = string.Format("Partition model {0} summary:", i);
                var summaryKvps = PredictorModels[i].Predictor as ICanGetSummaryInKeyValuePairs;
                var summaryModel = PredictorModels[i].Predictor as ICanSaveSummary;
                if (summaryKvps == null && summaryModel == null)
                {
                    list.Add(new KeyValuePair<string, object>(key, PredictorModels[i].Predictor.GetType().Name));
                    continue;
                }

                // Load the feature names for the i'th model.
                var dv = new EmptyDataView(Host, PredictorModels[i].TransformModel.InputSchema);
                PredictorModels[i].PrepareData(Host, dv, out RoleMappedData rmd, out IPredictor pred);

                if (summaryModel != null)
                {
                    sb.Clear();
                    using (StringWriter sw = new StringWriter(sb))
                        summaryModel.SaveSummary(sw, rmd.Schema);
                }

                if (summaryKvps != null)
                {
                    var listCur = summaryKvps.GetSummaryInKeyValuePairs(rmd.Schema);
                    if (summaryModel != null)
                    {
                        using (var reader = new StringReader(sb.ToString()))
                        {
                            string firstLine = null;
                            while (string.IsNullOrEmpty(firstLine))
                                firstLine = reader.ReadLine();
                            if (!string.IsNullOrEmpty(firstLine))
                                key += ("\r\n" + firstLine);
                        }
                    }
                    list.Add(new KeyValuePair<string, object>(key, listCur));
                }
                else
                {
                    Host.AssertValue(summaryModel);
                    list.Add(new KeyValuePair<string, object>(key, sb.ToString()));
                }

            }
            return list;
        }

        public string[] GetLabelNamesOrNull(out ColumnType labelType)
        {
            Host.AssertNonEmpty(PredictorModels);
            return PredictorModels[0].GetLabelInfo(Host, out labelType);
        }
    }
}
