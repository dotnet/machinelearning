// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.Runtime.Recommender.Internal;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;

[assembly: LoadableClass(MatrixFactorizationTrainer.Summary, typeof(MatrixFactorizationTrainer), typeof(MatrixFactorizationTrainer.Arguments),
    new Type[] { typeof(SignatureTrainer), typeof(SignatureMatrixRecommendingTrainer) },
    "Matrix Factorization", MatrixFactorizationTrainer.LoadNameValue, "libmf", "mf")]

namespace Microsoft.ML.Trainers
{
    public sealed class MatrixFactorizationTrainer : TrainerBase<MatrixFactorizationPredictor>,
        IEstimator<MatrixFactorizationPredictionTransformer>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization parameter. " +
                "It's the weight of factor matrices' norms in the objective function minimized by matrix factorization's algorithm. " +
                "A small value could cause over-fitting.")]
            [TGUI(SuggestedSweeps = "0.01,0.05,0.1,0.5,1")]
            [TlcModule.SweepableDiscreteParam("Lambda", new object[] { 0.01f, 0.05f, 0.1f, 0.5f, 1f })]
            public double Lambda = 0.1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Latent space dimension (denoted by k). If the factorized matrix is m-by-n, " +
                "two factor matrices found by matrix factorization are m-by-k and k-by-n, respectively. " +
                "This value is also known as the rank of matrix factorization because k is generally much smaller than m and n.")]
            [TGUI(SuggestedSweeps = "8,16,64,128")]
            [TlcModule.SweepableDiscreteParam("K", new object[] { 8, 16, 64, 128 })]
            public int K = 8;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Training iterations; that is, the times that the training algorithm iterates through the whole training data once.", ShortName = "iter")]
            [TGUI(SuggestedSweeps = "10,20,40")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 10, 20, 40 })]
            public int NumIterations = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate. It specifies the speed of the training algorithm. " +
                "Small value may increase the number of iterations needed to achieve a reasonable result. Large value may lead to numerical difficulty such as a infinity value.")]
            [TGUI(SuggestedSweeps = "0.001,0.01,0.1")]
            [TlcModule.SweepableDiscreteParam("Eta", new object[] { 0.001f, 0.01f, 0.1f })]
            public double Eta = 0.1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads can be used in the training procedure.", ShortName = "t")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Suppress writing additional information to output.")]
            public bool Quiet;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force the factor matrices to be non-negative.", ShortName = "nn")]
            public bool NonNegative;
        };

        internal const string Summary = "From pairs of row/column indices and a value of a matrix, this trains a predictor capable of filling in unknown entries of the matrix, "
            + "using a low-rank matrix factorization. This technique is often used in recommender system, where the row and column indices indicate users and items, "
            + "and the values of the matrix are ratings. ";

        // LIBMF's parameter
        private readonly double _lambda;
        private readonly int _k;
        private readonly int _iter;
        private readonly double _eta;
        private readonly int _threads;
        private readonly bool _quiet;
        private readonly bool _doNmf;

        public override PredictionKind PredictionKind => PredictionKind.Recommendation;
        public const string LoadNameValue = "MatrixFactorization";

        /// <summary>
        /// The row index, column index, and label columns needed to specify the training matrix. This trainer uses tuples of (row index, column index, label value) to specify a matrix.
        /// For example, a 2-by-2 matrix
        ///   [9, 4]
        ///   [8, 7]
        /// can be encoded as tuples (0, 0, 9), (0, 1, 4), (1, 0, 8), and (1, 1, 7). It means that the row/column/label column contains [0, 0, 1, 1]/
        /// [0, 1, 0, 1]/[9, 4, 8, 7].
        /// </summary>

        /// <summary>
        /// The name of variable (i.e., Column in a <see cref="IDataView"/> type system) used be as matrix's column index.
        /// </summary>
        public readonly string MatrixColumnIndexName;

        /// <summary>
        /// The name of variable (i.e., column in a <see cref="IDataView"/> type system) used as matrix's row index.
        /// </summary>
        public readonly string MatrixRowIndexName;

        /// <summary>
        /// The name variable (i.e., column in a <see cref="IDataView"/> type system) used as matrix's element value.
        /// </summary>
        public readonly string LabelName;

        /// <summary>
        /// The <see cref="TrainerInfo"/> contains general parameters for this trainer.
        /// </summary>
        public override TrainerInfo Info { get; }

        /// <summary>
        /// Extra information the trainer can use. For example, its validation set (if not null) can be use to evaluate the
        /// training progress made at each training iteration.
        /// </summary>
        private readonly TrainerEstimatorContext _context;

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
        /// <param name="matrixColumnIndexColumnName">The name of the column hosting the matrix's column IDs.</param>
        /// <param name="matrixRowIndexColumnName">The name of the column hosting the matrix's row IDs.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="context">The <see cref="TrainerEstimatorContext"/> for additional input data to training.</param>
        public MatrixFactorizationTrainer(IHostEnvironment env, string labelColumn, string matrixColumnIndexColumnName, string matrixRowIndexColumnName,
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
            _context = context;

            LabelName = labelColumn;
            MatrixColumnIndexName = matrixColumnIndexColumnName;
            MatrixRowIndexName = matrixRowIndexColumnName;
        }

        /// <summary>
        /// Train a matrix factorization model based on training data, validation data, and so on in the given context.
        /// </summary>
        /// <param name="context">The information collection needed for training. <see cref="TrainContext"/> for details.</param>
        public override MatrixFactorizationPredictor Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));

            using (var ch = Host.Start("Training"))
            {
                return TrainCore(ch, context.TrainingSet, context.ValidationSet);
            }
        }

        private MatrixFactorizationPredictor TrainCore(IChannel ch, RoleMappedData data, RoleMappedData validData)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);
            ch.AssertValueOrNull(validData);

            ColumnInfo matrixColumnIndexColInfo;
            ColumnInfo matrixRowIndexColInfo;
            ColumnInfo validMatrixColumnIndexColInfo = null;
            ColumnInfo validMatrixRowIndexColInfo = null;

            ch.CheckValue(data.Schema.Label, nameof(data), "Input data did not have a unique label");
            RecommenderUtils.CheckAndGetMatrixIndexColumns(data, out matrixColumnIndexColInfo, out matrixRowIndexColInfo, isDecode: false);
            if (data.Schema.Label.Type != NumberType.R4 && data.Schema.Label.Type != NumberType.R8)
                throw ch.Except("Column '{0}' for label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);
            MatrixFactorizationPredictor predictor;
            if (validData != null)
            {
                ch.CheckValue(validData, nameof(validData));
                ch.CheckValue(validData.Schema.Label, nameof(validData), "Input validation data did not have a unique label");
                RecommenderUtils.CheckAndGetMatrixIndexColumns(validData, out validMatrixColumnIndexColInfo, out validMatrixRowIndexColInfo, isDecode: false);
                if (validData.Schema.Label.Type != NumberType.R4 && validData.Schema.Label.Type != NumberType.R8)
                    throw ch.Except("Column '{0}' for validation label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);

                if (!matrixColumnIndexColInfo.Type.Equals(validMatrixColumnIndexColInfo.Type))
                {
                    throw ch.ExceptParam(nameof(validData), "Train and validation sets' matrix-column types differed, {0} vs. {1}",
                        matrixColumnIndexColInfo.Type, validMatrixColumnIndexColInfo.Type);
                }
                if (!matrixRowIndexColInfo.Type.Equals(validMatrixRowIndexColInfo.Type))
                {
                    throw ch.ExceptParam(nameof(validData), "Train and validation sets' matrix-row types differed, {0} vs. {1}",
                        matrixRowIndexColInfo.Type, validMatrixRowIndexColInfo.Type);
                }
            }

            int colCount = matrixColumnIndexColInfo.Type.KeyCount;
            int rowCount = matrixRowIndexColInfo.Type.KeyCount;
            ch.Assert(rowCount > 0);
            ch.Assert(colCount > 0);

            // Checks for equality on the validation set ensure it is correct here.
            using (var cursor = data.Data.GetRowCursor(c => c == matrixColumnIndexColInfo.Index || c == matrixRowIndexColInfo.Index || c == data.Schema.Label.Index))
            {
                // LibMF works only over single precision floats, but we want to be able to consume either.
                var labGetter = RowCursorUtils.GetGetterAs<float>(NumberType.R4, cursor, data.Schema.Label.Index);
                var matrixColumnIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, cursor, matrixColumnIndexColInfo.Index);
                var matrixRowIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, cursor, matrixRowIndexColInfo.Index);

                if (validData == null)
                {
                    // Have the trainer do its work.
                    using (var buffer = PrepareBuffer())
                    {
                        buffer.Train(ch, rowCount, colCount, cursor, labGetter, matrixRowIndexGetter, matrixColumnIndexGetter);
                        predictor = new MatrixFactorizationPredictor(Host, buffer, matrixColumnIndexColInfo.Type.AsKey, matrixRowIndexColInfo.Type.AsKey);
                    }
                }
                else
                {
                    using (var validCursor = validData.Data.GetRowCursor(
                        c => c == validMatrixColumnIndexColInfo.Index || c == validMatrixRowIndexColInfo.Index || c == validData.Schema.Label.Index))
                    {
                        ValueGetter<float> validLabelGetter = RowCursorUtils.GetGetterAs<float>(NumberType.R4, validCursor, validData.Schema.Label.Index);
                        var validMatrixColumnIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, validCursor, validMatrixColumnIndexColInfo.Index);
                        var validMatrixRowIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, validCursor, validMatrixRowIndexColInfo.Index);

                        // Have the trainer do its work.
                        using (var buffer = PrepareBuffer())
                        {
                            buffer.TrainWithValidation(ch, rowCount, colCount,
                                cursor, labGetter, matrixRowIndexGetter, matrixColumnIndexGetter,
                                validCursor, validLabelGetter, validMatrixRowIndexGetter, validMatrixColumnIndexGetter);
                            predictor = new MatrixFactorizationPredictor(Host, buffer, matrixColumnIndexColInfo.Type.AsKey, matrixRowIndexColInfo.Type.AsKey);
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

        /// <summary>
        /// Train a matrix factorization model based on the input <see cref="IDataView"/> using the roles specified by XColumn and YColumn in <see cref="MatrixFactorizationTrainer"/>.
        /// </summary>
        /// <param name="input">The training data set.</param>
        public MatrixFactorizationPredictionTransformer Fit(IDataView input)
        {
            MatrixFactorizationPredictor model = null;

            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, LabelName));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommenderUtils.MatrixColumnIndexKind.Value, MatrixColumnIndexName));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommenderUtils.MatrixRowIndexKind.Value, MatrixRowIndexName));

            var trainingData = new RoleMappedData(input, roles);
            var validData = _context == null ? null : new RoleMappedData(_context.ValidationSet, roles);

            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                model = TrainCore(ch, trainingData, validData);
            }

            return new MatrixFactorizationPredictionTransformer(Host, model, input.Schema, MatrixColumnIndexName, MatrixRowIndexName);
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

            // Check if label column is good.
            var labelColumn = new SchemaShape.Column(LabelName, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
            CheckColumnsCompatible(labelColumn, LabelName);

            // Check if columns of matrix's row and column indexes are good. Note that column of IDataView and column of matrix are two different things.
            var matrixColumnIndexColumn = new SchemaShape.Column(MatrixColumnIndexName, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);
            var matrixRowIndexColumn = new SchemaShape.Column(MatrixRowIndexName, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);
            CheckColumnsCompatible(matrixColumnIndexColumn, MatrixColumnIndexName);
            CheckColumnsCompatible(matrixRowIndexColumn, MatrixRowIndexName);

            // Input columns just pass through so that output column dictionary contains all input columns.
            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);

            // Add columns produced by this estimator.
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        private SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelName, out var labelCol);
            Contracts.Assert(success);

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
            };
        }
    }
}
