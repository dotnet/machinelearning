//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
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
    public sealed class MatrixFactorizationTrainer : TrainerBase<MatrixFactorizationPredictor>
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
        }
        private static TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);
        public override TrainerInfo Info => _info;
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

            ColumnInfo dataY;
            ColumnInfo dataX;
            ColumnInfo validY = null;
            ColumnInfo validX = null;

            Host.CheckValue(data.Schema.Label, nameof(data), "Input data did not have a unique label");
            RecommendUtils.CheckAndGetXYColumns(data, out dataX, out dataY, isDecode: false);
            if (data.Schema.Label.Type != NumberType.R4 && data.Schema.Label.Type != NumberType.R8)
                throw Host.Except("Column '{0}' for label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);
            MatrixFactorizationPredictor predictor;
            if (validData != null)
            {
                Host.CheckValue(validData, nameof(validData));
                Host.CheckValue(validData.Schema.Label, nameof(validData), "Input validation data did not have a unique label");
                RecommendUtils.CheckAndGetXYColumns(validData, out validX, out validY, isDecode: false);
                if (validData.Schema.Label.Type != NumberType.R4 && validData.Schema.Label.Type != NumberType.R8)
                    throw Host.Except("Column '{0}' for validation label should be floating point, but is instead {1}", data.Schema.Label.Name, data.Schema.Label.Type);

                if (!dataX.Type.Equals(validX.Type))
                {
                    throw Host.ExceptParam(nameof(validData), "Train and validation set X types differed, {0} vs. {1}",
                        dataX.Type, validX.Type);
                }
                if (!dataY.Type.Equals(validY.Type))
                {
                    throw Host.ExceptParam(nameof(validData), "Train and validation set Y types differed, {0} vs. {1}",
                        dataY.Type, validY.Type);
                }
            }

            int colCount = dataX.Type.KeyCount;
            int rowCount = dataY.Type.KeyCount;
            Host.Assert(rowCount > 0);
            Host.Assert(colCount > 0);
            // Checks for equality on the validation set ensure it is correct here.

            using (var cursor = data.Data.GetRowCursor(c => c == dataY.Index || c == dataX.Index || c == data.Schema.Label.Index))
            {
                // LibMF works only over single precision floats, but we want to be able to consume either.
                ValueGetter<Single> labGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, cursor, data.Schema.Label.Index);
                var xGetter = cursor.GetGetter<uint>(dataX.Index);
                var yGetter = cursor.GetGetter<uint>(dataY.Index);

                if (validData == null)
                {
                    // Have the trainer do its work.
                    using (var buffer = PrepareBuffer())
                    {
                        buffer.Train(ch, rowCount, colCount,
                            cursor, labGetter, yGetter, xGetter);
                        predictor = new MatrixFactorizationPredictor(Host, buffer, dataX.Type.AsKey, dataY.Type.AsKey);
                    }
                }
                else
                {
                    using (var validCursor = validData.Data.GetRowCursor(c => c == validY.Index || c == validX.Index || c == validData.Schema.Label.Index))
                    {
                        ValueGetter<Single> validLabGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, validCursor, validData.Schema.Label.Index);
                        var validXGetter = validCursor.GetGetter<uint>(validX.Index);
                        var validYGetter = validCursor.GetGetter<uint>(validY.Index);

                        // Have the trainer do its work.
                        using (var buffer = PrepareBuffer())
                        {
                            buffer.TrainWithValidation(ch, rowCount, colCount,
                                cursor, labGetter, yGetter, xGetter,
                                validCursor, validLabGetter, validYGetter, validXGetter);
                            predictor = new MatrixFactorizationPredictor(Host, buffer, dataX.Type.AsKey, dataY.Type.AsKey);
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
    }
}
