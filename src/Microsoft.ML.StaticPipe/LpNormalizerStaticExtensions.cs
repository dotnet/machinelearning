// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed <see cref="LpNormalizingEstimator"/>.
    /// </summary>
    public static class LpNormalizerStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, LpNormalizingEstimatorBase.NormFunction norm, bool ensureZeroMean)
                : base(new Reconciler(norm, ensureZeroMean), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly LpNormalizingEstimatorBase.NormFunction _norm;
            private readonly bool _ensureZeroMean;

            public Reconciler(LpNormalizingEstimatorBase.NormFunction normKind, bool ensureZeroMean)
            {
                _norm = normKind;
                _ensureZeroMean = ensureZeroMean;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string outputColumnName, string inputColumnName)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input]));

                return new LpNormalizingEstimator(env, pairs.ToArray(), _norm, _ensureZeroMean);
            }
        }

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public static Vector<float> LpNormalize(this Vector<float> input,
            LpNormalizingEstimatorBase.NormFunction normKind = LpNormalizingEstimatorBase.Defaults.Norm,
            bool subMean = LpNormalizingEstimatorBase.Defaults.LpEnsureZeroMean) => new OutPipelineColumn(input, normKind, subMean);
    }
}
