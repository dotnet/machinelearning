// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
    public sealed class Whitening : TrainedWrapperEstimatorBase
    {
        private readonly (string input, string output)[] _columns;
        private readonly WhiteningKind _kind;
        private readonly float _eps;
        private readonly int _maxRows;
        private readonly bool _saveInverse;
        private readonly int _pcaNum;

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public Whitening(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            WhiteningKind kind = WhiteningKind.Zca,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false,
            int pcaNum = 0)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, kind, eps, maxRows, saveInverse, pcaNum)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public Whitening(IHostEnvironment env, (string input, string output)[] columns,
            WhiteningKind kind = WhiteningKind.Zca,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false,
            int pcaNum = 0)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizer)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _kind = kind;
            _eps = eps;
            _maxRows = maxRows;
            _saveInverse = saveInverse;
            _pcaNum = pcaNum;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            var args = new WhiteningTransform.Arguments
            {
                Column = _columns.Select(x => new WhiteningTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                Kind = _kind,
                Eps = _eps,
                MaxRows = _maxRows,
                SaveInverse = _saveInverse,
                PcaNum = _pcaNum
            };

            return new TransformWrapper(Host, new WhiteningTransform(Host, args, input));
        }
    }

    /// <summary>
    /// Extensions for statically typed Whitening estimator.
    /// </summary>
    public static class WhiteningExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, WhiteningKind kind, float eps, int maxRows, bool saveInverse, int pcaNum)
                : base(new Reconciler(kind, eps, maxRows, saveInverse, pcaNum), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly WhiteningKind _kind;
            private readonly float _eps;
            private readonly int _maxRows;
            private readonly bool _saveInverse;
            private readonly int _pcaNum;

            public Reconciler(WhiteningKind kind, float eps, int maxRows, bool saveInverse, int pcaNum)
            {
                _kind = kind;
                _eps = eps;
                _maxRows = maxRows;
                _saveInverse = saveInverse;
                _pcaNum = pcaNum;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new Whitening(env, pairs.ToArray(), _kind, _eps, _maxRows, _saveInverse, _pcaNum);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public static Vector<float> PcaWhitening(this Vector<float> input,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false,
            int pcaNum = 0) => new OutPipelineColumn(input, WhiteningKind.Pca, eps, maxRows, saveInverse, pcaNum);

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public static Vector<float> ZcaWhitening(this Vector<float> input,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false,
            int pcaNum = 0) => new OutPipelineColumn(input, WhiteningKind.Zca, eps, maxRows, saveInverse, pcaNum);
    }
}
