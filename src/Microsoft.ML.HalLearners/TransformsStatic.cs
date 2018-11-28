// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Projections;
using System.Collections.Generic;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed Whitening estimator.
    /// </summary>
    public static class VectorWhiteningExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, WhiteningKind kind, float eps, int maxRows, int pcaNum)
                : base(new Reconciler(kind, eps, maxRows, pcaNum), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly WhiteningKind _kind;
            private readonly float _eps;
            private readonly int _maxRows;
            private readonly int _pcaNum;

            public Reconciler(WhiteningKind kind, float eps, int maxRows, int pcaNum)
            {
                _kind = kind;
                _eps = eps;
                _maxRows = maxRows;
                _pcaNum = pcaNum;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var infos = new VectorWhiteningTransformer.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; i++)
                    infos[i] = new VectorWhiteningTransformer.ColumnInfo(inputNames[((OutPipelineColumn)toOutput[i]).Input], outputNames[toOutput[i]], _kind, _eps, _maxRows, _pcaNum);

                return new VectorWhiteningEstimator(env, infos);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to which the transform will be applied.</param>
        /// <param name="eps">Whitening constant, prevents division by zero when scaling the data by inverse of eigenvalues.</param>
        /// <param name="maxRows">Maximum number of rows used to train the transform.</param>
        /// <param name="pcaNum">In case of PCA whitening, indicates the number of components to retain.</param>
        public static Vector<float> PcaWhitening(this Vector<float> input,
            float eps = VectorWhiteningTransformer.Defaults.Eps,
            int maxRows = VectorWhiteningTransformer.Defaults.MaxRows,
            int pcaNum = VectorWhiteningTransformer.Defaults.PcaNum)
            => new OutPipelineColumn(input, WhiteningKind.Pca, eps, maxRows, pcaNum);

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to which the transform will be applied.</param>
        /// <param name="eps">Whitening constant, prevents division by zero.</param>
        /// <param name="maxRows">Maximum number of rows used to train the transform.</param>
        public static Vector<float> ZcaWhitening(this Vector<float> input,
            float eps = VectorWhiteningTransformer.Defaults.Eps,
            int maxRows = VectorWhiteningTransformer.Defaults.MaxRows)
            => new OutPipelineColumn(input, WhiteningKind.Zca, eps, maxRows, VectorWhiteningTransformer.Defaults.PcaNum);
    }
}
