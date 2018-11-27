// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.FeatureSelection;
using Microsoft.ML.Transforms.Projections;
using System.Collections.Generic;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed <see cref="LpNormalizingEstimator"/>.
    /// </summary>
    public static class LpNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, LpNormalizingEstimatorBase.NormalizerKind normKind, bool subMean)
                : base(new Reconciler(normKind, subMean), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly LpNormalizingEstimatorBase.NormalizerKind _normKind;
            private readonly bool _subMean;

            public Reconciler(LpNormalizingEstimatorBase.NormalizerKind normKind, bool subMean)
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

                return new LpNormalizingEstimator(env, pairs.ToArray(), _normKind, _subMean);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LpNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        public static Vector<float> LpNormalize(this Vector<float> input,
            LpNormalizingEstimatorBase.NormalizerKind normKind = LpNormalizingEstimatorBase.Defaults.NormKind,
            bool subMean = LpNormalizingEstimatorBase.Defaults.LpSubstractMean) => new OutPipelineColumn(input, normKind, subMean);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="GlobalContrastNormalizingEstimator"/>.
    /// </summary>
    public static class GlobalContrastNormalizerExtensions
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

                return new GlobalContrastNormalizingEstimator(env, pairs.ToArray(), _subMean, _useStdDev, _scale);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="input">The column to apply to.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        public static Vector<float> GlobalContrastNormalize(this Vector<float> input,
            bool subMean = LpNormalizingEstimatorBase.Defaults.GcnSubstractMean,
            bool useStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev,
            float scale = LpNormalizingEstimatorBase.Defaults.Scale) => new OutPipelineColumn(input, subMean, useStdDev, scale);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="MutualInformationFeatureSelectorExtensions"/>.
    /// </summary>
    public static class MutualInformationFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumn<T> : Vector<T>
        {
            public readonly Vector<T> Input;
            public readonly PipelineColumn LabelColumn;

            public OutPipelineColumn(Vector<T> input, Scalar<float> labelColumn, int slotsInOutput, int numBins)
                : base(new Reconciler<T>(labelColumn, slotsInOutput, numBins), input, labelColumn)
            {
                Input = input;
                LabelColumn = labelColumn;
            }

            public OutPipelineColumn(Vector<T> input, Scalar<bool> labelColumn, int slotsInOutput, int numBins)
               : base(new Reconciler<T>(labelColumn, slotsInOutput, numBins), input, labelColumn)
            {
                Input = input;
                LabelColumn = labelColumn;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly PipelineColumn _labelColumn;
            private readonly int _slotsInOutput;
            private readonly int _numBins;

            public Reconciler(PipelineColumn labelColumn, int slotsInOutput, int numBins)
            {
                _labelColumn = labelColumn;
                _slotsInOutput = slotsInOutput;
                _numBins = numBins;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn<T>)outCol).Input], outputNames[outCol]));

                return new MutualInformationFeatureSelectingEstimator(env, inputNames[_labelColumn], _slotsInOutput, _numBins, pairs.ToArray());
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<float> SelectFeaturesBasedOnMutualInformation(
            this Vector<float> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<float> SelectFeaturesBasedOnMutualInformation(
            this Vector<float> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<double> SelectFeaturesBasedOnMutualInformation(
            this Vector<double> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<double> SelectFeaturesBasedOnMutualInformation(
            this Vector<double> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<bool> SelectFeaturesBasedOnMutualInformation(
            this Vector<bool> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<bool>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<bool> SelectFeaturesBasedOnMutualInformation(
            this Vector<bool> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectingEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectingEstimator.Defaults.NumBins) => new OutPipelineColumn<bool>(input, labelColumn, slotsInOutput, numBins);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="CountFeatureSelectorExtensions"/>.
    /// </summary>
    public static class CountFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumn<T> : Vector<T>
        {
            public readonly Vector<T> Input;

            public OutPipelineColumn(Vector<T> input, long count)
                : base(new Reconciler<T>(count), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly long _count;

            public Reconciler(long count)
            {
                _count = count;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var infos = new CountFeatureSelectingEstimator.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; i++)
                    infos[i] = new CountFeatureSelectingEstimator.ColumnInfo(inputNames[((OutPipelineColumn<T>)toOutput[i]).Input], outputNames[toOutput[i]], _count);

                return new CountFeatureSelectingEstimator(env, infos);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<float> SelectFeaturesBasedOnCount(this Vector<float> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<float>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<double> SelectFeaturesBasedOnCount(this Vector<double> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<double>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">Name of the input column.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FeatureSelectionTransform.cs?range=1-5,9-120)]
        /// ]]>
        /// </format>
        /// </example>
        public static Vector<string> SelectFeaturesBasedOnCount(this Vector<string> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<string>(input, count);
    }
}
