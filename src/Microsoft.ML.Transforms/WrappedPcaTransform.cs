// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
    public sealed class PcaEstimator : TrainedWrapperEstimatorBase
    {
        private readonly PcaTransform.Arguments _args;

        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Input column to apply PCA on.</param>
        /// <param name="outputColumn">Output column. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public PcaEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int rank = PcaTransform.Defaults.Rank,
            Action<PcaTransform.Arguments> advancedSettings = null)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, rank, advancedSettings)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="PCA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the PCA on.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public PcaEstimator(IHostEnvironment env, (string input, string output)[] columns,
            int rank = PcaTransform.Defaults.Rank,
            Action<PcaTransform.Arguments> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizer)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _args = new PcaTransform.Arguments();
            _args.Column = columns.Select(x => new PcaTransform.Column { Source = x.input, Name = x.output }).ToArray();
            _args.Rank = rank;

            advancedSettings?.Invoke(_args);
        }

        public override TransformWrapper Fit(IDataView input)
        {
            return new TransformWrapper(Host, new PcaTransform(Host, _args, input));
        }
    }

    /// <summary>
    /// Extensions for statically typed <see cref="PcaEstimator"/>.
    /// </summary>
    public static class PcaEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, int rank, Action<PcaTransform.Arguments> advancedSettings)
                : base(new Reconciler(null, rank, advancedSettings), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _rank;
            private readonly Action<PcaTransform.Arguments> _advancedSettings;

            public Reconciler(PipelineColumn weightColumn, int rank, Action<PcaTransform.Arguments> advancedSettings)
            {
                _rank = rank;
                _advancedSettings = advancedSettings;
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

                return new PcaEstimator(env, pairs.ToArray(), _rank, _advancedSettings);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to apply PCA to.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static Vector<float> ToPrincipalComponents(this Vector<float> input,
            int rank = PcaTransform.Defaults.Rank,
            Action<PcaTransform.Arguments> advancedSettings = null) => new OutPipelineColumn(input, rank, advancedSettings);
    }
}
