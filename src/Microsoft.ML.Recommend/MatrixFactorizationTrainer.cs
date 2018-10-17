//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Core.Prediction;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Recommend;
using Microsoft.ML.Runtime.Recommend.Internal;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(MatrixFactorizationTrainer.Summary, typeof(MatrixFactorizationTrainer), typeof(MatrixFactorizationTrainer.Arguments),
    new Type[] { typeof(SignatureTrainer), typeof(SignatureMatrixRecommendingTrainer) },
    "Matrix Factorization", MatrixFactorizationTrainer.LoadNameValue, "libmf", "mf")]

namespace Microsoft.ML.Runtime.Recommend
{
    public sealed class MatrixFactorizationTrainer : TrainerBase<MatrixFactorizationPredictor>,
        IEstimator<MatrixFactorizationPredictionTransformer>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization parameter")]
            [TGUI(SuggestedSweeps = "0.01,0.05,0.1,0.5,1")]
            [TlcModule.SweepableDiscreteParam("Lambda", new object[] { 0.01f, 0.05f, 0.1f, 0.5f, 1f })]
            public Double Lambda = 0.1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Latent space dimension")]
            [TGUI(SuggestedSweeps = "8,16,64,128")]
            [TlcModule.SweepableDiscreteParam("K", new object[] { 8, 16, 64, 128 })]
            public int K = 8;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Training iterations", ShortName = "iter")]
            [TGUI(SuggestedSweeps = "10,20,40")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 10, 20, 40 })]
            public int NumIterations = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate")]
            [TGUI(SuggestedSweeps = "0.001,0.01,0.1")]
            [TlcModule.SweepableDiscreteParam("Eta", new object[] { 0.001f, 0.01f, 0.1f })]
            public Double Eta = 0.1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads", ShortName = "t")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Suppress writing additional information to output")]
            public bool Quiet;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force the matrix factorization P and Q to be non-negative", ShortName = "nn")]
            public bool NonNegative;
        };

        internal const string Summary = "From pairs of row/column indices and a value of a matrix, this trains a predictor capable of filling in unknown entries of the matrix, "
            + "utilizing a low-rank matrix factorization. This technique is often used in recommender system, where the row and column indices indicate users and items, "
            + "and the value of the matrix is some rating. ";

        private readonly Double _lambda;
        private readonly int _k;
        private readonly int _iter;
        private readonly Double _eta;
        private readonly int _threads;
        private readonly bool _quiet;
        private readonly bool _doNmf;

        public override PredictionKind PredictionKind => PredictionKind.Recommendation;
        public const string LoadNameValue = "MatrixFactorization";

        /// <summary>
        /// The row, column, and label columns that the trainer expects. This module uses tuples of (row index, column index, label value) to specify a matrix.
        /// For example, a 2-by-2 matrix
        ///   [9, 4]
        ///   [8, 7]
        /// can be encoded as tuples (0, 0, 9), (0, 1, 4), (1, 0, 8), and (1, 1, 7). It means that the row/column/label column contains [0, 0, 1, 1]/
        /// [0, 1, 0, 1]/[9, 4, 8, 7]. Note that for a given matrix, row indices are column indices are denoted by Y and X, respectively.
        /// </summary>
        public readonly SchemaShape.Column XColumn; // column indices of the training matrix
        public readonly SchemaShape.Column YColumn; // row indices of the training matrix
        public readonly SchemaShape.Column LabelColumn;

        /// <summary>
        /// The <see cref="TrainerInfo"/> contains general parameters for this trainer.
        /// </summary>
        public override TrainerInfo Info { get; }

        /// <summary>
        /// Extra information the trainer can use. For example, its validation set (if not null) can be use to evaluate the
        /// training progress made at each training iteration.
        /// </summary>
        public readonly TrainerEstimatorContext Context;

        /// <summary>
        /// Legacy constructor initializing a new instance of <see cref="MatrixFactorizationTrainer"/> through the legacy
        /// <see cref="Arguments"/> class.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="args">An instance of the legacy <see cref="Arguments"/> to apply advanced parameters to the algorithm.</param>
        public MatrixFactorizationTrainer(IHostEnvironment env, Arguments args) : base(env, LoadNameValue)
        {

            const string posError = "Parameter must be positive";
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(args.K > 0, nameof(args.K), posError);
            Host.CheckUserArg(!args.NumThreads.HasValue || args.NumThreads > 0, nameof(args.NumThreads), posError);
            Host.CheckUserArg(args.NumIterations > 0, nameof(args.NumIterations), posError);
            Host.CheckUserArg(args.Lambda > 0, nameof(args.Lambda), posError);
            Host.CheckUserArg(args.Eta > 0, nameof(args.Eta), posError);

            _lambda = args.Lambda;
            _k = args.K;
            _iter = args.NumIterations;
            _eta = args.Eta;
            _threads = args.NumThreads ?? Environment.ProcessorCount;
            _quiet = args.Quiet;
            _doNmf = args.NonNegative;

            Info = new TrainerInfo(normalization: false, caching: false);
        }

        /// <summary>
        /// Initializing a new instance of <see cref="MatrixFactorizationTrainer"/>.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="xColumnName">The name of the column hosting the matrix's column IDs.</param>
        /// <param name="yColumnName">The name of the column hosting the matrix's row IDs.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="context">The <see cref="TrainerEstimatorContext"/> for additional input data to training.</param>
        public MatrixFactorizationTrainer(IHostEnvironment env, string labelColumn, string xColumnName, string yColumnName,
            TrainerEstimatorContext context = null, Action<Arguments> advancedSettings = null)
            : base(env, LoadNameValue)
        {
            var args = new Arguments();
            advancedSettings?.Invoke(args);

            _lambda = args.Lambda;
            _k = args.K;
            _iter = args.NumIterations;
            _eta = args.Eta;
            _threads = args.NumThreads ?? Environment.ProcessorCount;
            _quiet = args.Quiet;
            _doNmf = args.NonNegative;

            Info = new TrainerInfo(normalization: false, caching: false);
            Context = context;

            LabelColumn = new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
            XColumn = new SchemaShape.Column(xColumnName, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);
            YColumn = new SchemaShape.Column(yColumnName, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);
        }

        public override MatrixFactorizationPredictor Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));

            using (var ch = Host.Start("Training"))
            {
                var pred = TrainCore(ch, context.TrainingSet, context.ValidationSet);
                return pred;
            }
        }

        private MatrixFactorizationPredictor TrainCore(IChannel ch, RoleMappedData data, RoleMappedData validData)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);
            ch.AssertValueOrNull(validData);

            ColumnInfo xColInfo;
            ColumnInfo yColInfo;
            ColumnInfo validXColInfo = null;
            ColumnInfo validYColInfo = null;

            Host.CheckValue(data.Schema.Label, nameof(data), "Input data did not have a unique label");
            RecommendUtils.CheckAndGetXYColumns(data, out xColInfo, out yColInfo, isDecode: false);
            if (data.Schema.Label.Type != NumberType.R4 && data.Schema.Label.Type != NumberType.R8)
                throw Host.Except("Column '{0}' for label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);
            MatrixFactorizationPredictor predictor;
            if (validData != null)
            {
                Host.CheckValue(validData, nameof(validData));
                Host.CheckValue(validData.Schema.Label, nameof(validData), "Input validation data did not have a unique label");
                RecommendUtils.CheckAndGetXYColumns(validData, out validXColInfo, out validYColInfo, isDecode: false);
                if (validData.Schema.Label.Type != NumberType.R4 && validData.Schema.Label.Type != NumberType.R8)
                    throw Host.Except("Column '{0}' for validation label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);

                if (!xColInfo.Type.Equals(validXColInfo.Type))
                {
                    throw Host.ExceptParam(nameof(validData), "Train and validation set X types differed, {0} vs. {1}",
                        xColInfo.Type, validXColInfo.Type);
                }
                if (!yColInfo.Type.Equals(validYColInfo.Type))
                {
                    throw Host.ExceptParam(nameof(validData), "Train and validation set Y types differed, {0} vs. {1}",
                        yColInfo.Type, validYColInfo.Type);
                }
            }

            int colCount = xColInfo.Type.KeyCount;
            int rowCount = yColInfo.Type.KeyCount;
            Host.Assert(rowCount > 0);
            Host.Assert(colCount > 0);
            // Checks for equality on the validation set ensure it is correct here.

            using (var cursor = data.Data.GetRowCursor(c => c == xColInfo.Index || c == yColInfo.Index || c == data.Schema.Label.Index))
            {
                // LibMF works only over single precision floats, but we want to be able to consume either.
                ValueGetter<Single> labGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, cursor, data.Schema.Label.Index);
                var xGetter = cursor.GetGetter<uint>(xColInfo.Index);
                var yGetter = cursor.GetGetter<uint>(yColInfo.Index);

                if (validData == null)
                {
                    // Have the trainer do its work.
                    using (var buffer = PrepareBuffer())
                    {
                        buffer.Train(ch, rowCount, colCount,
                            cursor, labGetter, yGetter, xGetter);
                        predictor = new MatrixFactorizationPredictor(Host, buffer, xColInfo.Type.AsKey, yColInfo.Type.AsKey);
                    }
                }
                else
                {
                    using (var validCursor = validData.Data.GetRowCursor(
                        c => c == validXColInfo.Index || c == validYColInfo.Index || c == validData.Schema.Label.Index))
                    {
                        ValueGetter<Single> validLabGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, validCursor, validData.Schema.Label.Index);
                        var validXGetter = validCursor.GetGetter<uint>(validXColInfo.Index);
                        var validYGetter = validCursor.GetGetter<uint>(validYColInfo.Index);

                        // Have the trainer do its work.
                        using (var buffer = PrepareBuffer())
                        {
                            buffer.TrainWithValidation(ch, rowCount, colCount,
                                cursor, labGetter, yGetter, xGetter,
                                validCursor, validLabGetter, validYGetter, validXGetter);
                            predictor = new MatrixFactorizationPredictor(Host, buffer, xColInfo.Type.AsKey, yColInfo.Type.AsKey);
                        }
                    }
                }

            }
            return predictor;
        }

        private SafeTrainingAndModelBuffer PrepareBuffer()
        {
            return new SafeTrainingAndModelBuffer(Host, _k, Math.Max(20, 2 * _threads),
                _threads, _iter, _lambda, _eta, _doNmf, _quiet, copyData: false);
        }

        public MatrixFactorizationPredictionTransformer Fit(IDataView input)
        {
            MatrixFactorizationPredictor model = null;

            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, LabelColumn.Name));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommendUtils.XKind.Value, XColumn.Name));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommendUtils.YKind.Value, YColumn.Name));

            var trainingData = new RoleMappedData(input, roles);
            var validData = Context == null ? null : new RoleMappedData(Context.ValidationSet, roles);

            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                model = TrainCore(ch, trainingData, validData);
            }

            return new MatrixFactorizationPredictionTransformer(Host, model, input.Schema, XColumn.Name, YColumn.Name);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            void CheckColumnsCompatible(SchemaShape.Column cachedColumn, string expectedColumnName)
            {
                if (!inputSchema.TryFindColumn(cachedColumn.Name, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(col), expectedColumnName, expectedColumnName);

                if (!cachedColumn.IsCompatibleWith(col))
                    throw Host.Except($"{expectedColumnName} column '{cachedColumn.Name}' is not compatible");
            }

            // In prediction phase, no label column is expected.
            if (LabelColumn != null)
                CheckColumnsCompatible(LabelColumn, LabelColumn.Name);

            // In both of training and prediction phases, we need columns of user ID and column ID.
            CheckColumnsCompatible(XColumn, XColumn.Name);
            CheckColumnsCompatible(YColumn, YColumn.Name);

            // Input columns just pass through so that output column dictionary contains all input columns.
            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);

            // Add columns produced by this estimator.
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        private SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
            };
        }
    }
}
