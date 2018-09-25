// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.StaticPipe.Runtime
{
    /// <summary>
    /// General purpose reconciler for a typical case with trainers, where they accept some generally
    /// fixed number of inputs, and produce some outputs where the names of the outputs are fixed.
    /// Authors of components that want to produce columns can subclass this directly, or use one of the
    /// common nested subclasses.
    /// </summary>
    public abstract class TrainerEstimatorReconciler : EstimatorReconciler
    {
        protected readonly PipelineColumn[] Inputs;
        private readonly string[] _outputNames;

        /// <summary>
        /// The output columns. Note that subclasses should return exactly the same items each time,
        /// and the items should correspond to the output names passed into the constructor.
        /// </summary>
        protected abstract IEnumerable<PipelineColumn> Outputs { get; }

        /// <summary>
        /// Constructor for the base class.
        /// </summary>
        /// <param name="inputs">The set of inputs</param>
        /// <param name="outputNames">The names of the outputs, which we assume cannot be changed</param>
        protected TrainerEstimatorReconciler(PipelineColumn[] inputs, string[] outputNames)
        {
            Contracts.CheckValue(inputs, nameof(inputs));
            Contracts.CheckValue(outputNames, nameof(outputNames));

            Inputs = inputs;
            _outputNames = outputNames;
        }

        /// <summary>
        /// Produce the training estimator.
        /// </summary>
        /// <param name="env">The host environment to use to create the estimator.</param>
        /// <param name="inputNames">The names of the inputs, which corresponds exactly to the input columns
        /// fed into the constructor.</param>
        /// <returns>An estimator, which should produce the additional columns indicated by the output names
        /// in the constructor.</returns>
        protected abstract IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames);

        /// <summary>
        /// Produces the estimator. Note that this is made out of <see cref="ReconcileCore(IHostEnvironment, string[])"/>'s
        /// return value, plus whatever usages of <see cref="CopyColumnsEstimator"/> are necessary to avoid collisions with
        /// the output names fed to the constructor. This class provides the implementation, and subclasses should instead
        /// override <see cref="ReconcileCore(IHostEnvironment, string[])"/>.
        /// </summary>
        public sealed override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
            PipelineColumn[] toOutput,
            IReadOnlyDictionary<PipelineColumn, string> inputNames,
            IReadOnlyDictionary<PipelineColumn, string> outputNames,
            IReadOnlyCollection<string> usedNames)
        {
            Contracts.AssertValue(env);
            env.AssertValue(toOutput);
            env.AssertValue(inputNames);
            env.AssertValue(outputNames);
            env.AssertValue(usedNames);

            // The reconciler should have been called with all the input columns having names.
            env.Assert(inputNames.Keys.All(Inputs.Contains) && Inputs.All(inputNames.Keys.Contains));
            // The output name map should contain only outputs as their keys. Yet, it is possible not all
            // outputs will be required in which case these will both be subsets of those outputs indicated
            // at construction.
            env.Assert(outputNames.Keys.All(Outputs.Contains));
            env.Assert(toOutput.All(Outputs.Contains));
            env.Assert(Outputs.Count() == _outputNames.Length);

            IEstimator<ITransformer> result = null;

            // In the case where we have names used that conflict with the fixed output names, we must have some
            // renaming logic.
            var collisions = new HashSet<string>(_outputNames);
            collisions.IntersectWith(usedNames);
            var old2New = new Dictionary<string, string>();

            if (collisions.Count > 0)
            {
                // First get the old names to some temporary names.
                int tempNum = 0;
                foreach (var c in collisions)
                    old2New[c] = $"#TrainTemp{tempNum++}";
                // In the case where the input names have anything that is used, we must reconstitute the input mapping.
                if (inputNames.Values.Any(old2New.ContainsKey))
                {
                    var newInputNames = new Dictionary<PipelineColumn, string>();
                    foreach (var p in inputNames)
                        newInputNames[p.Key] = old2New.ContainsKey(p.Value) ? old2New[p.Value] : p.Value;
                    inputNames = newInputNames;
                }
                result = new CopyColumnsEstimator(env, old2New.Select(p => (p.Key, p.Value)).ToArray());
            }

            // Map the inputs to the names.
            string[] mappedInputNames = Inputs.Select(c => inputNames[c]).ToArray();
            // Finally produce the trainer.
            var trainerEst = ReconcileCore(env, mappedInputNames);
            if (result == null)
                result = trainerEst;
            else
                result = result.Append(trainerEst);

            // OK. Now handle the final renamings from the fixed names, to the desired names, in the case
            // where the output was desired, and a renaming is even necessary.
            var toRename = new List<(string source, string name)>();
            foreach ((PipelineColumn outCol, string fixedName) in Outputs.Zip(_outputNames, (c, n) => (c, n)))
            {
                if (outputNames.TryGetValue(outCol, out string desiredName))
                    toRename.Add((fixedName, desiredName));
                else
                    env.Assert(!toOutput.Contains(outCol));
            }
            // Finally if applicable handle the renaming back from the temp names to the original names.
            foreach (var p in old2New)
                toRename.Add((p.Value, p.Key));
            if (toRename.Count > 0)
                result = result.Append(new CopyColumnsEstimator(env, toRename.ToArray()));

            return result;
        }

        /// <summary>
        /// A reconciler for regression capable of handling the most common cases for regression.
        /// </summary>
        public sealed class Regression : TrainerEstimatorReconciler
        {
            /// <summary>
            /// The delegate to create the regression trainer instance.
            /// </summary>
            /// <param name="env">The environment with which to create the estimator</param>
            /// <param name="label">The label column name</param>
            /// <param name="features">The features column name</param>
            /// <param name="weights">The weights column name, or <c>null</c> if the reconciler was constructed with <c>null</c> weights</param>
            /// <returns>A estimator producing columns with the fixed name <see cref="DefaultColumnNames.Score"/>.</returns>
            public delegate IEstimator<ITransformer> EstimatorFactory(IHostEnvironment env, string label, string features, string weights);

            private readonly EstimatorFactory _estFact;

            /// <summary>
            /// The output score column for the regression. This will have this instance as its reconciler.
            /// </summary>
            public Scalar<float> Score { get; }

            protected override IEnumerable<PipelineColumn> Outputs => Enumerable.Repeat(Score, 1);

            private static readonly string[] _fixedOutputNames = new[] { DefaultColumnNames.Score };

            /// <summary>
            /// Constructs a new general regression reconciler.
            /// </summary>
            /// <param name="estimatorFactory">The delegate to create the training estimator. It is assumed that this estimator
            /// will produce a single new scalar <see cref="float"/> column named <see cref="DefaultColumnNames.Score"/>.</param>
            /// <param name="label">The input label column.</param>
            /// <param name="features">The input features column.</param>
            /// <param name="weights">The input weights column, or <c>null</c> if there are no weights.</param>
            public Regression(EstimatorFactory estimatorFactory, Scalar<float> label, Vector<float> features, Scalar<float> weights)
                    : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(features, nameof(features)), weights),
                          _fixedOutputNames)
            {
                Contracts.CheckValue(estimatorFactory, nameof(estimatorFactory));
                _estFact = estimatorFactory;
                Contracts.Assert(Inputs.Length == 2 || Inputs.Length == 3);
                Score = new Impl(this);
            }

            private static PipelineColumn[] MakeInputs(Scalar<float> label, Vector<float> features, Scalar<float> weights)
                => weights == null ? new PipelineColumn[] { label, features } : new PipelineColumn[] { label, features, weights };

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);
                env.Assert(Utils.Size(inputNames) == Inputs.Length);
                return _estFact(env, inputNames[0], inputNames[1], inputNames.Length > 2 ? inputNames[2] : null);
            }

            private sealed class Impl : Scalar<float>
            {
                public Impl(Regression rec) : base(rec, rec.Inputs) { }
            }
        }

        /// <summary>
        /// A reconciler capable of handling the most common cases for binary classification with calibrated outputs.
        /// </summary>
        public sealed class BinaryClassifier : TrainerEstimatorReconciler
        {
            /// <summary>
            /// The delegate to create the binary classifier trainer instance.
            /// </summary>
            /// <param name="env">The environment with which to create the estimator.</param>
            /// <param name="label">The label column name.</param>
            /// <param name="features">The features column name.</param>
            /// <param name="weights">The weights column name, or <c>null</c> if the reconciler was constructed with <c>null</c> weights.</param>
            /// <returns>A binary classification trainer estimator.</returns>
            public delegate IEstimator<ITransformer> EstimatorFactory(IHostEnvironment env, string label, string features, string weights);

            private readonly EstimatorFactory _estFact;
            private static readonly string[] _fixedOutputNames = new[] { DefaultColumnNames.Score, DefaultColumnNames.Probability, DefaultColumnNames.PredictedLabel };

            /// <summary>
            /// The general output for binary classifiers.
            /// </summary>
            public (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) Output { get; }

            protected override IEnumerable<PipelineColumn> Outputs => new PipelineColumn[] { Output.score, Output.probability, Output.predictedLabel };

            /// <summary>
            /// Constructs a new general regression reconciler.
            /// </summary>
            /// <param name="estimatorFactory">The delegate to create the training estimator. It is assumed that this estimator
            /// will produce a single new scalar <see cref="float"/> column named <see cref="DefaultColumnNames.Score"/>.</param>
            /// <param name="label">The input label column.</param>
            /// <param name="features">The input features column.</param>
            /// <param name="weights">The input weights column, or <c>null</c> if there are no weights.</param>
            public BinaryClassifier(EstimatorFactory estimatorFactory, Scalar<bool> label, Vector<float> features, Scalar<float> weights)
                : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(features, nameof(features)), weights),
                      _fixedOutputNames)
            {
                Contracts.CheckValue(estimatorFactory, nameof(estimatorFactory));
                _estFact = estimatorFactory;
                Contracts.Assert(Inputs.Length == 2 || Inputs.Length == 3);

                Output = (new Impl(this), new Impl(this), new ImplBool(this));
            }

            private static PipelineColumn[] MakeInputs(Scalar<bool> label, Vector<float> features, Scalar<float> weights)
                => weights == null ? new PipelineColumn[] { label, features } : new PipelineColumn[] { label, features, weights };

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);
                env.Assert(Utils.Size(inputNames) == Inputs.Length);
                return _estFact(env, inputNames[0], inputNames[1], inputNames.Length > 2 ? inputNames[2] : null);
            }

            private sealed class Impl : Scalar<float>
            {
                public Impl(BinaryClassifier rec) : base(rec, rec.Inputs) { }
            }

            private sealed class ImplBool : Scalar<bool>
            {
                public ImplBool(BinaryClassifier rec) : base(rec, rec.Inputs) { }
            }
        }

        /// <summary>
        /// A reconciler capable of handling the most common cases for binary classification that does not
        /// necessarily have calibrated outputs.
        /// </summary>
        public sealed class BinaryClassifierNoCalibration : TrainerEstimatorReconciler
        {
            /// <summary>
            /// The delegate to create the binary classifier trainer instance.
            /// </summary>
            /// <param name="env">The environment with which to create the estimator</param>
            /// <param name="label">The label column name.</param>
            /// <param name="features">The features column name.</param>
            /// <param name="weights">The weights column name, or <c>null</c> if the reconciler was constructed with <c>null</c> weights.</param>
            /// <returns>A binary classification trainer estimator.</returns>
            public delegate IEstimator<ITransformer> EstimatorFactory(IHostEnvironment env, string label, string features, string weights);

            private readonly EstimatorFactory _estFact;
            private static readonly string[] _fixedOutputNamesProb = new[] { DefaultColumnNames.Score, DefaultColumnNames.Probability, DefaultColumnNames.PredictedLabel };
            private static readonly string[] _fixedOutputNames = new[] { DefaultColumnNames.Score, DefaultColumnNames.PredictedLabel };

            /// <summary>
            /// The general output for binary classifiers.
            /// </summary>
            public (Scalar<float> score, Scalar<bool> predictedLabel) Output { get; }

            /// <summary>
            /// The output columns, which will contain at least the columns produced by <see cref="Output"/> and may contain an
            /// additional <see cref="DefaultColumnNames.Probability"/> column if at runtime we determine the predictor actually
            /// is calibrated.
            /// </summary>
            protected override IEnumerable<PipelineColumn> Outputs { get; }

            /// <summary>
            /// Constructs a new general binary classifier reconciler.
            /// </summary>
            /// <param name="estimatorFactory">The delegate to create the training estimator. It is assumed that this estimator
            /// will produce a single new scalar <see cref="float"/> column named <see cref="DefaultColumnNames.Score"/>.</param>
            /// <param name="label">The input label column.</param>
            /// <param name="features">The input features column.</param>
            /// <param name="weights">The input weights column, or <c>null</c> if there are no weights.</param>
            /// <param name="hasProbs">While this type is a compile time construct, it may be that at runtime we have determined that we will have probabilities,
            /// and so ought to do the renaming of the <see cref="DefaultColumnNames.Probability"/> column anyway if appropriate. If this is so, then this should
            /// be set to true.</param>
            public BinaryClassifierNoCalibration(EstimatorFactory estimatorFactory, Scalar<bool> label, Vector<float> features, Scalar<float> weights, bool hasProbs)
                : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(features, nameof(features)), weights),
                      hasProbs ? _fixedOutputNamesProb : _fixedOutputNames)
            {
                Contracts.CheckValue(estimatorFactory, nameof(estimatorFactory));
                _estFact = estimatorFactory;
                Contracts.Assert(Inputs.Length == 2 || Inputs.Length == 3);

                Output = (new Impl(this), new ImplBool(this));

                if (hasProbs)
                    Outputs = new PipelineColumn[] { Output.score, new Impl(this), Output.predictedLabel };
                else
                    Outputs = new PipelineColumn[] { Output.score, Output.predictedLabel };
            }

            private static PipelineColumn[] MakeInputs(Scalar<bool> label, Vector<float> features, Scalar<float> weights)
                => weights == null ? new PipelineColumn[] { label, features } : new PipelineColumn[] { label, features, weights };

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);
                env.Assert(Utils.Size(inputNames) == Inputs.Length);
                return _estFact(env, inputNames[0], inputNames[1], inputNames.Length > 2 ? inputNames[2] : null);
            }

            private sealed class Impl : Scalar<float>
            {
                public Impl(BinaryClassifierNoCalibration rec) : base(rec, rec.Inputs) { }
            }

            private sealed class ImplBool : Scalar<bool>
            {
                public ImplBool(BinaryClassifierNoCalibration rec) : base(rec, rec.Inputs) { }
            }
        }

        /// <summary>
        /// A reconciler for regression capable of handling the most common cases for regression.
        /// </summary>
        public sealed class MulticlassClassifier<TVal> : TrainerEstimatorReconciler
        {
            /// <summary>
            /// The delegate to create the multiclass classifier trainer instance.
            /// </summary>
            /// <param name="env">The environment with which to create the estimator</param>
            /// <param name="label">The label column name</param>
            /// <param name="features">The features column name</param>
            /// <param name="weights">The weights column name, or <c>null</c> if the reconciler was constructed with <c>null</c> weights</param>
            /// <returns>A estimator producing columns with the fixed name <see cref="DefaultColumnNames.Score"/> and <see cref="DefaultColumnNames.PredictedLabel"/>.</returns>
            public delegate IEstimator<ITransformer> EstimatorFactory(IHostEnvironment env, string label, string features, string weights);

            private readonly EstimatorFactory _estFact;

            /// <summary>
            /// The general output for multiclass classifiers.
            /// </summary>
            public (Vector<float> score, Key<uint, TVal> predictedLabel) Output { get; }

            protected override IEnumerable<PipelineColumn> Outputs => new PipelineColumn[] { Output.score, Output.predictedLabel };

            private static readonly string[] _fixedOutputNames = new[] { DefaultColumnNames.Score, DefaultColumnNames.PredictedLabel };

            /// <summary>
            /// Constructs a new general multiclass classifier reconciler.
            /// </summary>
            /// <param name="estimatorFactory">The delegate to create the training estimator. It is assumed that this estimator
            /// will produce a vector <see cref="float"/> column named <see cref="DefaultColumnNames.Score"/> and a scalar
            /// key column named <see cref="DefaultColumnNames.PredictedLabel"/>.</param>
            /// <param name="label">The input label column.</param>
            /// <param name="features">The input features column.</param>
            /// <param name="weights">The input weights column, or <c>null</c> if there are no weights.</param>
            public MulticlassClassifier(EstimatorFactory estimatorFactory, Key<uint, TVal> label, Vector<float> features, Scalar<float> weights)
                    : base(MakeInputs(Contracts.CheckRef(label, nameof(label)), Contracts.CheckRef(features, nameof(features)), weights),
                          _fixedOutputNames)
            {
                Contracts.CheckValue(estimatorFactory, nameof(estimatorFactory));
                _estFact = estimatorFactory;
                Contracts.Assert(Inputs.Length == 2 || Inputs.Length == 3);
                Output = (new ImplScore(this), new ImplLabel(this));
            }

            private static PipelineColumn[] MakeInputs(Key<uint, TVal> label, Vector<float> features, Scalar<float> weights)
                => weights == null ? new PipelineColumn[] { label, features } : new PipelineColumn[] { label, features, weights };

            protected override IEstimator<ITransformer> ReconcileCore(IHostEnvironment env, string[] inputNames)
            {
                Contracts.AssertValue(env);
                env.Assert(Utils.Size(inputNames) == Inputs.Length);
                return _estFact(env, inputNames[0], inputNames[1], inputNames.Length > 2 ? inputNames[2] : null);
            }

            private sealed class ImplLabel : Key<uint, TVal>
            {
                public ImplLabel(MulticlassClassifier<TVal> rec) : base(rec, rec.Inputs) { }
            }

            private sealed class ImplScore : Vector<float>
            {
                public ImplScore(MulticlassClassifier<TVal> rec) : base(rec, rec.Inputs) { }
            }
        }

    }
}
