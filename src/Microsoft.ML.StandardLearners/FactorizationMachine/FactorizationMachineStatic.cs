// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Extension methods and utilities for instantiating FFM trainer estimators inside statically typed pipelines.
    /// </summary>
    public static class FactorizationMachineStatic
    {
        /// <summary>
        /// Predict a target using a field-aware factorization machine.
        /// </summary>
        /// <param name="ctx">The binary classifier context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="learningRate">Initial learning rate.</param>
        /// <param name="numIterations">Number of training iterations.</param>
        /// <param name="numLatentDimensions">Latent space dimensions.</param>
        /// <param name="advancedSettings">A delegate to set more settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) FieldAwareFactorizationMachine(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float>[] features,
            float learningRate = 0.1f,
            int numIterations = 5,
            int numLatentDimensions = 20,
            Action<FieldAwareFactorizationMachineTrainer.Arguments> advancedSettings = null,
            Action<FieldAwareFactorizationMachinePredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckNonEmpty(features, nameof(features));

            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive");
            Contracts.CheckParam(numIterations > 0, nameof(numIterations), "Must be positive");
            Contracts.CheckParam(numLatentDimensions > 0, nameof(numLatentDimensions), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);

            var rec = new CustomReconciler((env, labelCol, featureCols) =>
            {
                var trainer = new FieldAwareFactorizationMachineTrainer(env, labelCol, featureCols, advancedSettings:
                    args =>
                    {
                        advancedSettings?.Invoke(args);
                        args.LearningRate = learningRate;
                        args.Iters = numIterations;
                        args.LatentDim = numLatentDimensions;
                    });
                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                else
                    return trainer;
            }, label, features);
            return rec.Output;
        }

        private sealed class CustomReconciler : TrainerEstimatorReconciler
        {
            private static readonly string[] _fixedOutputNames = new[] { DefaultColumnNames.Score, DefaultColumnNames.PredictedLabel };
            private readonly Func<IHostEnvironment, string, string[], IEstimator<ITransformer>> _factory;

            /// <summary>
            /// The general output for binary classifiers.
            /// </summary>
            public (Scalar<float> score, Scalar<bool> predictedLabel) Output { get; }

            /// <summary>
            /// The output columns.
            /// </summary>
            protected override IEnumerable<PipelineColumn> Outputs { get; }

            public CustomReconciler(Func<IHostEnvironment, string, string[], IEstimator<ITransformer>> factory, Scalar<bool> label, Vector<float>[] features)
                : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(features, nameof(features))), _fixedOutputNames)
            {
                Contracts.AssertValue(factory);
                _factory = factory;

                Output = (new Impl(this), new ImplBool(this));
                Outputs = new PipelineColumn[] { Output.score, Output.predictedLabel };
            }

            private static PipelineColumn[] MakeInputs(Scalar<bool> label, Vector<float>[] features)
                => new PipelineColumn[] { label }.Concat(features).ToArray();

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);
                env.Assert(Utils.Size(inputNames) == Inputs.Length);

                // First input is label, rest are features.
                return _factory(env, inputNames[0], inputNames.Skip(1).ToArray());
            }

            private sealed class Impl : Scalar<float>
            {
                public Impl(CustomReconciler rec) : base(rec, rec.Inputs) { }
            }

            private sealed class ImplBool : Scalar<bool>
            {
                public ImplBool(CustomReconciler rec) : base(rec, rec.Inputs) { }
            }
        }
    }
}
