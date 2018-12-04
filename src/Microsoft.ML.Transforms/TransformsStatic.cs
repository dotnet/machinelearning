// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.FeatureSelection;
using Microsoft.ML.Transforms.Projections;
using System;
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

    public static class CategoricalStaticExtensions
    {
        public enum OneHotVectorOutputKind : byte
        {
            /// <summary>
            /// Output is a bag (multi-set) vector
            /// </summary>
            Bag = 1,

            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        public enum OneHotScalarOutputKind : byte
        {
            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        private const KeyValueOrder DefSort = (KeyValueOrder)ValueToKeyMappingEstimator.Defaults.Sort;
        private const int DefMax = ValueToKeyMappingEstimator.Defaults.MaxNumTerms;
        private const OneHotVectorOutputKind DefOut = (OneHotVectorOutputKind)OneHotEncodingEstimator.Defaults.OutKind;

        private readonly struct Config
        {
            public readonly KeyValueOrder Order;
            public readonly int Max;
            public readonly OneHotVectorOutputKind OutputKind;
            public readonly Action<ValueToKeyMappingTransformer.TermMap> OnFit;

            public Config(OneHotVectorOutputKind outputKind, KeyValueOrder order, int max, Action<ValueToKeyMappingTransformer.TermMap> onFit)
            {
                OutputKind = outputKind;
                Order = order;
                Max = max;
                OnFit = onFit;
            }
        }

        private static Action<ValueToKeyMappingTransformer.TermMap> Wrap<T>(ToKeyFitResult<T>.OnFit onFit)
        {
            if (onFit == null)
                return null;
            // The type T asociated with the delegate will be the actual value type once #863 goes in.
            // However, until such time as #863 goes in, it would be too awkward to attempt to extract the metadata.
            // For now construct the useless object then pass it into the delegate.
            return map => onFit(new ToKeyFitResult<T>(map));
        }

        private interface ICategoricalCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new OneHotEncodingEstimator.ColumnInfo[toOutput.Length];
                Action<ValueToKeyMappingTransformer> onFit = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ICategoricalCol)toOutput[i];
                    infos[i] = new OneHotEncodingEstimator.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], (OneHotEncodingTransformer.OutputKind)tcol.Config.OutputKind,
                        tcol.Config.Max, (ValueToKeyMappingTransformer.SortOrder)tcol.Config.Order);
                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetTermMap(ii));
                    }
                }
                var est = new OneHotEncodingEstimator(env, infos);
                if (onFit != null)
                    est.WrapTermWithDelegate(onFit);
                return est;
            }
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="order">How the Id for each value would be assigined: by occurrence or by value.</param>
        /// <param name="maxItems">Maximum number of ids to keep during data scanning.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> OneHotEncoding(this Scalar<string> input, OneHotScalarOutputKind outputKind = (OneHotScalarOutputKind)DefOut, KeyValueOrder order = DefSort,
            int maxItems = DefMax, ToKeyFitResult<ReadOnlyMemory<char>>.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplScalar<string>(input, new Config((OneHotVectorOutputKind)outputKind, order, maxItems, Wrap(onFit)));
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: Multiarray, array or binary encoded data.</param>
        /// <param name="order">How the Id for each value would be assigined: by occurrence or by value.</param>
        /// <param name="maxItems">Maximum number of ids to keep during data scanning.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> OneHotEncoding(this Vector<string> input, OneHotVectorOutputKind outputKind = DefOut, KeyValueOrder order = DefSort, int maxItems = DefMax,
            ToKeyFitResult<ReadOnlyMemory<char>>.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(outputKind, order, maxItems, Wrap(onFit)));
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToBinaryVectorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutVectorColumn<TKey, TValue> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<Key<TKey, TValue>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }

            public OutVectorColumn(Key<TKey, TValue> input)
              : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TKey, TValue> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public OutVarVectorColumn(VarVector<Key<TKey, TValue>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TKey> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<Key<TKey>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }

            public OutVectorColumn(Key<TKey> input)
              : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TKey> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public OutVarVectorColumn(VarVector<Key<TKey>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new KeyToBinaryVectorMappingTransformer.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new KeyToBinaryVectorMappingTransformer.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]]);
                }
                return new KeyToBinaryVectorMappingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey, TValue>(this Key<TKey, TValue> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static VarVector<float> ToBinaryVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey>(this Key<TKey> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static VarVector<float> ToBinaryVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey>(input);
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToVectorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
            bool Bag { get; }
        }

        private sealed class OutVectorColumn<TKey, TValue> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVectorColumn(Key<TKey, TValue> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }

            public OutVectorColumn(Vector<Key<TKey, TValue>> input, bool bag)
                : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = bag;
            }

            public OutVectorColumn(VarVector<Key<TKey, TValue>> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = true;
            }
        }

        private sealed class OutVarVectorColumn<TKey, TValue> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVarVectorColumn(VarVector<Key<TKey, TValue>> input)
            : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }
        }

        private sealed class OutVectorColumn<TKey> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVectorColumn(Key<TKey> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }

            public OutVectorColumn(Vector<Key<TKey>> input, bool bag)
                : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = bag;
            }

            public OutVectorColumn(VarVector<Key<TKey>> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = true;
            }
        }

        private sealed class OutVarVectorColumn<TKey> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVarVectorColumn(VarVector<Key<TKey>> input)
            : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new KeyToVectorMappingTransformer.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new KeyToVectorMappingTransformer.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]], col.Bag);
                }
                return new KeyToVectorMappingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey, TValue>(this Key<TKey, TValue> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input, false);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static VarVector<float> ToVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input, true);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey>(this Key<TKey> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input, false);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static VarVector<float> ToVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input, true);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class NAReplacerExtensions
    {
        private readonly struct Config
        {
            public readonly bool ImputeBySlot;
            public readonly MissingValueReplacingTransformer.ColumnInfo.ReplacementMode ReplacementMode;

            public Config(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode,
                bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
            {
                ImputeBySlot = imputeBySlot;
                ReplacementMode = replacementMode;
            }
        }

        private interface IColInput
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class OutScalar<TValue> : Scalar<TValue>, IColInput
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }

            public OutScalar(Scalar<TValue> input, Config config)
              : base(Reconciler.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class OutVectorColumn<TValue> : Vector<TValue>, IColInput
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }

            public OutVectorColumn(Vector<TValue> input, Config config)
              : base(Reconciler.Inst, input)
            {
                Input = input;
                Config = config;
            }

        }

        private sealed class OutVarVectorColumn<TValue> : VarVector<TValue>, IColInput
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }

            public OutVarVectorColumn(VarVector<TValue> input, Config config)
                : base(Reconciler.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new MissingValueReplacingTransformer.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new MissingValueReplacingTransformer.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]], col.Config.ReplacementMode, col.Config.ImputeBySlot);
                }
                return new MissingValueReplacingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static Scalar<float> ReplaceNaNValues(this Scalar<float> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<float>(input, new Config(replacementMode, false));
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static Scalar<double> ReplaceNaNValues(this Scalar<double> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<double>(input, new Config(replacementMode, false));
        }
        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        /// <param name="imputeBySlot">If true, per-slot imputation of replacement is performed.
        /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
        /// where imputation is always for the entire column.</param>
        public static Vector<float> ReplaceNaNValues(this Vector<float> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode, bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<float>(input, new Config(replacementMode, imputeBySlot));
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        /// <param name="imputeBySlot">If true, per-slot imputation of replacement is performed.
        /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
        /// where imputation is always for the entire column.</param>
        public static Vector<double> ReplaceNaNValues(this Vector<double> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode, bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<double>(input, new Config(replacementMode, imputeBySlot));
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static VarVector<float> ReplaceNaNValues(this VarVector<float> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<float>(input, new Config(replacementMode, false));
        }
        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static VarVector<double> ReplaceNaNValues(this VarVector<double> input, MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<double>(input, new Config(replacementMode, false));
        }
    }
}
