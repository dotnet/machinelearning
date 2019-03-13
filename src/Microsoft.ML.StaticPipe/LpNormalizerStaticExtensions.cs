// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed <see cref="LpNormNormalizingEstimator"/>.
    /// </summary>
    public static class LpNormNormalizerStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, LpNormNormalizingEstimatorBase.NormFunction norm, bool ensureZeroMean)
                : base(new Reconciler(norm, ensureZeroMean), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly LpNormNormalizingEstimatorBase.NormFunction _norm;
            private readonly bool _ensureZeroMean;

            public Reconciler(LpNormNormalizingEstimatorBase.NormFunction norm, bool ensureZeroMean)
            {
                _norm = norm;
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

                return new LpNormNormalizingEstimator(env, pairs.ToArray(), _norm, _ensureZeroMean);
            }
        }

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="input">The column containing the vectors to apply the normalization to.</param>
        /// <param name="norm">Type of norm to use to normalize each sample.</param>
        /// <param name="ensureZeroMean">Subtract mean from each value before normalizing.</param>
        public static Vector<float> NormalizeLpNorm(this Vector<float> input,
            LpNormNormalizingEstimatorBase.NormFunction norm = LpNormNormalizingEstimatorBase.Defaults.Norm,
            bool ensureZeroMean = LpNormNormalizingEstimatorBase.Defaults.LpEnsureZeroMean) => new OutPipelineColumn(input, norm, ensureZeroMean);
    }
}
