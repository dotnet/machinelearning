// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Projections;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.Transforms.Projections.LpNormNormalizerTransform;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
    public sealed class LpNormalizer : TrivialWrapperEstimator
    {
        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public LpNormalizer(IHostEnvironment env, string inputColumn, string outputColumn = null, NormalizerKind normKind = NormalizerKind.L2Norm, bool subMean = false)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, normKind, subMean)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public LpNormalizer(IHostEnvironment env, (string input, string output)[] columns, NormalizerKind normKind = NormalizerKind.L2Norm, bool subMean = false)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LpNormalizer)), MakeTransformer(env, columns, normKind, subMean))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, NormalizerKind normKind, bool subMean)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            var args = new LpNormNormalizerTransform.Arguments
            {
                Column = columns.Select(x => new LpNormNormalizerTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                SubMean = subMean,
                NormKind = normKind
            };

            // Create a valid instance of data.
            var schema = new Schema(columns.Select(x => new Schema.Column(x.input, new VectorType(NumberType.R4), null)));
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new LpNormNormalizerTransform(env, args, emptyData));
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
    public sealed class GlobalContrastNormalizer : TrivialWrapperEstimator
    {
        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public GlobalContrastNormalizer(IHostEnvironment env, string inputColumn, string outputColumn = null, bool subMean = true, bool useStdDev = false, float scale = 1)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, subMean, useStdDev , scale)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public GlobalContrastNormalizer(IHostEnvironment env, (string input, string output)[] columns, bool subMean = true, bool useStdDev = false, float scale = 1)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(GlobalContrastNormalizer)), MakeTransformer(env, columns, subMean, useStdDev, scale))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, bool subMean, bool useStdDev, float scale)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            var args = new LpNormNormalizerTransform.GcnArguments
            {
                Column = columns.Select(x => new LpNormNormalizerTransform.GcnColumn { Source = x.input, Name = x.output }).ToArray(),
                SubMean = subMean,
                UseStdDev = useStdDev,
                Scale = scale
            };

            // Create a valid instance of data.
            var schema = new Schema(columns.Select(x => new Schema.Column(x.input, new VectorType(NumberType.R4), null)));
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new LpNormNormalizerTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Extensions for statically typed LpNormalizer estimator.
    /// </summary>
    public static class LpNormNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, NormalizerKind normKind, bool subMean)
                : base(new Reconciler(normKind, subMean), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly NormalizerKind _normKind;
            private readonly bool _subMean;

            public Reconciler(NormalizerKind normKind, bool subMean)
            {
                _normKind = normKind;
                _subMean = subMean;
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

                return new LpNormalizer(env, pairs.ToArray(), _normKind, _subMean);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public static Vector<float> LpNormalize(this Vector<float> input, NormalizerKind normKind = NormalizerKind.L2Norm, bool subMean = false) => new OutPipelineColumn(input, normKind, subMean);
    }

    /// <summary>
    /// Extensions for statically typed GcNormalizer estimator.
    /// </summary>
    public static class GcNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, bool subMean, bool useStdDev, float scale)
                : base(new Reconciler(subMean, useStdDev, scale), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly bool _subMean;
            private readonly bool _useStdDev;
            private readonly float _scale;

            public Reconciler(bool subMean, bool useStdDev, float scale)
            {
                _subMean = subMean;
                _useStdDev = useStdDev;
                _scale = scale;
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

                return new GlobalContrastNormalizer(env, pairs.ToArray(), _subMean, _useStdDev, _scale);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public static Vector<float> GlobalContrastNormalize(this Vector<float> input,
            bool subMean = true,
            bool useStdDev = false,
            float scale = 1) => new OutPipelineColumn(input, subMean, useStdDev, scale);
    }
}
