// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;

namespace Microsoft.ML.StaticPipe
{
    public static class MatrixFactorizationExtensions
    {
        /// <summary>
        /// Predict matrix entry using matrix factorization
        /// </summary>
        /// <typeparam name="T">The type of physical value of matrix's row and column index. It must be an integer type such as uint.</typeparam>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="label">The label variable.</param>
        /// <param name="matrixColumnIndex">The column index of the considered matrix.</param>
        /// <param name="matrixRowIndex">The row index of the considered matrix.</param>
        /// <param name="options">Advanced algorithm settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static Scalar<float> MatrixFactorization<T>(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Key<T> matrixColumnIndex, Key<T> matrixRowIndex,
            MatrixFactorizationTrainer.Options options,
            Action<MatrixFactorizationModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(matrixColumnIndex, nameof(matrixColumnIndex));
            Contracts.CheckValue(matrixRowIndex, nameof(matrixRowIndex));
            Contracts.CheckValue(options, nameof(options));
            Contracts.CheckValueOrNull(onFit);

            var rec = new MatrixFactorizationReconciler<T>((env, labelColName, matrixColumnIndexColName, matrixRowIndexColName) =>
            {
                options.MatrixColumnIndexColumnName = matrixColumnIndexColName;
                options.MatrixRowIndexColumnName = matrixRowIndexColName;
                options.LabelColumnName = labelColName;

                var trainer = new MatrixFactorizationTrainer(env, options);

                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                else
                    return trainer;
            }, label, matrixColumnIndex, matrixRowIndex);
            return rec.Output;
        }

        private sealed class MatrixFactorizationReconciler<T> : TrainerEstimatorReconciler
        {
            // Output column name of the trained estimator.
            private static string FixedOutputName => DefaultColumnNames.Score;

            // A function used to create trainer of matrix factorization. It instantiates a trainer by indicating the
            // expected inputs and output (IDataView's) column names. That trainer has a Fit(IDataView data) for learning
            // a MatrixFactorizationPredictionTransformer from the data.
            private readonly Func<IHostEnvironment, string, string, string, IEstimator<ITransformer>> _factory;

            /// <summary>
            /// The only output produced by matrix factorization predictor
            /// </summary>
            public Scalar<float> Output { get; }

            /// <summary>
            /// The output columns.
            /// </summary>
            protected override IEnumerable<PipelineColumn> Outputs { get; }

            public MatrixFactorizationReconciler(Func<IHostEnvironment, string, string, string, IEstimator<ITransformer>> factory,
                Scalar<float> label, Key<T> matColumnIndex, Key<T> matRowIndex)
                : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(matColumnIndex, nameof(matColumnIndex)), Contracts.CheckRef(matRowIndex, nameof(matRowIndex))),
                      new string[] { FixedOutputName })
            {
                Contracts.AssertValue(factory);
                _factory = factory;

                Output = new Impl(this);
                Outputs = new PipelineColumn[] { Output };
            }

            private static PipelineColumn[] MakeInputs(Scalar<float> label, PipelineColumn matrixRowIndex, PipelineColumn matrixColumnIndex)
                => new PipelineColumn[] { label, matrixRowIndex, matrixColumnIndex };

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);

                // The first, second, third names are label, matrix's column index, and matrix's row index, respectively.
                return _factory(env, inputNames[0], inputNames[1], inputNames[2]);
            }

            private sealed class Impl : Scalar<float>
            {
                public Impl(MatrixFactorizationReconciler<T> rec) : base(rec, rec.Inputs) { }
            }
        }
    }
}
