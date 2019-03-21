// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.Transforms.Text.TextFeaturizingEstimator;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed <see cref="GlobalContrastNormalizingEstimator"/>.
    /// </summary>
    public static class GlobalContrastNormalizerStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, bool ensureZeroMean, bool ensureUnitStandardDeviation, float scale)
                : base(new Reconciler(ensureZeroMean, ensureUnitStandardDeviation, scale), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly bool _ensureZeroMean;
            private readonly bool _ensureUnitStandardDeviation;
            private readonly float _scale;

            public Reconciler(bool ensureZeroMean, bool ensureUnitStandardDeviation, float scale)
            {
                _ensureZeroMean = ensureZeroMean;
                _ensureUnitStandardDeviation = ensureUnitStandardDeviation;
                _scale = scale;
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

                return new GlobalContrastNormalizingEstimator(env, pairs.ToArray(), _ensureZeroMean, _ensureUnitStandardDeviation, _scale);
            }
        }

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="GcNormalize"]/*'/>
        /// <param name="input">The column containing the vectors to apply the normalization to.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
        /// <param name="scale">Scale features by this value.</param>
        public static Vector<float> NormalizeGlobalContrast(this Vector<float> input,
            bool ensureZeroMean = LpNormNormalizingEstimatorBase.Defaults.GcnEnsureZeroMean,
            bool ensureUnitStandardDeviation = LpNormNormalizingEstimatorBase.Defaults.EnsureUnitStdDev,
            float scale = LpNormNormalizingEstimatorBase.Defaults.Scale) => new OutPipelineColumn(input, ensureZeroMean, ensureUnitStandardDeviation, scale);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="MutualInformationFeatureSelectorStaticExtensions"/>.
    /// </summary>
    public static class MutualInformationFeatureSelectorStaticExtensions
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
                var pairs = new List<(string outputColumnName, string inputColumnName)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn<T>)outCol).Input]));

                return new MutualInformationFeatureSelectingEstimator(env, inputNames[_labelColumn], _slotsInOutput, _numBins, pairs.ToArray());
            }
        }

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
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
    /// Extensions for statically typed <see cref="CountFeatureSelectorStaticExtensions"/>.
    /// </summary>
    public static class CountFeatureSelectorStaticExtensions
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

                var infos = new CountFeatureSelectingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; i++)
                    infos[i] = new CountFeatureSelectingEstimator.ColumnOptions(outputNames[toOutput[i]], inputNames[((OutPipelineColumn<T>)toOutput[i]).Input], _count);

                return new CountFeatureSelectingEstimator(env, infos);
            }
        }

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
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

        /// <include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
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

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToBinaryVectorStaticExtensions
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
                var infos = new (string outputColumnName, string inputColumnName)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = (outputNames[toOutput[i]], inputNames[col.Input]);
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
    public static class KeyToVectorStaticExtensions
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
                var infos = new KeyToVectorMappingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new KeyToVectorMappingEstimator.ColumnOptions(outputNames[toOutput[i]], inputNames[col.Input], col.Bag);
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
    public static class NAReplacerStaticExtensions
    {
        private readonly struct Config
        {
            public readonly bool ImputeBySlot;
            public readonly MissingValueReplacingEstimator.ReplacementMode ReplacementMode;

            public Config(MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode,
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
                var infos = new MissingValueReplacingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new MissingValueReplacingEstimator.ColumnOptions(outputNames[toOutput[i]], inputNames[col.Input], col.Config.ReplacementMode, col.Config.ImputeBySlot);
                }
                return new MissingValueReplacingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static Scalar<float> ReplaceNaNValues(this Scalar<float> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<float>(input, new Config(replacementMode, false));
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static Scalar<double> ReplaceNaNValues(this Scalar<double> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode)
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
        public static Vector<float> ReplaceNaNValues(this Vector<float> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode, bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
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
        public static Vector<double> ReplaceNaNValues(this Vector<double> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode, bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<double>(input, new Config(replacementMode, imputeBySlot));
        }

        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static VarVector<float> ReplaceNaNValues(this VarVector<float> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<float>(input, new Config(replacementMode, false));
        }
        /// <summary>
        /// Scan through all rows and replace NaN values according to replacement strategy.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="replacementMode">How NaN should be replaced</param>
        public static VarVector<double> ReplaceNaNValues(this VarVector<double> input, MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<double>(input, new Config(replacementMode, false));
        }
    }

    public static partial class ConvertStaticExtensions
    {

        private interface IConvertCol
        {
            PipelineColumn Input { get; }
            InternalDataKind Kind { get; }
        }

        private sealed class ImplScalar<T> : Scalar<float>, IConvertCol
        {
            public PipelineColumn Input { get; }
            public InternalDataKind Kind { get; }
            public ImplScalar(PipelineColumn input, InternalDataKind kind) : base(Rec.Inst, input)
            {
                Input = input;
                Kind = kind;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, IConvertCol
        {
            public PipelineColumn Input { get; }
            public InternalDataKind Kind { get; }
            public ImplVector(PipelineColumn input, InternalDataKind kind) : base(Rec.Inst, input)
            {
                Input = input;
                Kind = kind;
            }
        }

        private sealed class ImplVarVector<T> : VarVector<float>, IConvertCol
        {
            public PipelineColumn Input { get; }
            public InternalDataKind Kind { get; }
            public ImplVarVector(PipelineColumn input, InternalDataKind kind) : base(Rec.Inst, input)
            {
                Input = input;
                Kind = kind;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new TypeConvertingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (IConvertCol)toOutput[i];
                    infos[i] = new TypeConvertingEstimator.ColumnOptions(outputNames[toOutput[i]], tcol.Kind.ToDataKind(), inputNames[tcol.Input]);
                }
                return new TypeConvertingEstimator(env, infos);
            }
        }
    }

    public static partial class TermStaticExtensions
    {
        // I am not certain I see a good way to cover the distinct types beyond complete enumeration.
        // Raw generics would allow illegal possible inputs, for example, Scalar<Bitmap>. So, this is a partial
        // class, and all the public facing extension methods for each possible type are in a T4 generated result.

        private const KeyOrdinality DefSort = (KeyOrdinality)ValueToKeyMappingEstimator.Defaults.Ordinality;
        private const int DefMax = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys;

        private readonly struct Config
        {
            public readonly KeyOrdinality Order;
            public readonly int Max;
            public readonly Action<ValueToKeyMappingTransformer.TermMap> OnFit;

            public Config(KeyOrdinality order, int max, Action<ValueToKeyMappingTransformer.TermMap> onFit)
            {
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

        private interface ITermCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Key<uint, T>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<Key<uint, T>>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVarVector<T> : VarVector<Key<uint, T>>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVarVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new ValueToKeyMappingEstimator.ColumnOptions[toOutput.Length];
                Action<ValueToKeyMappingTransformer> onFit = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ITermCol)toOutput[i];
                    infos[i] = new ValueToKeyMappingEstimator.ColumnOptions(outputNames[toOutput[i]], inputNames[tcol.Input],
                        tcol.Config.Max, (ValueToKeyMappingEstimator.KeyOrdinality)tcol.Config.Order);
                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetTermMap(ii));
                    }
                }
                var est = new ValueToKeyMappingEstimator(env, infos);
                if (onFit == null)
                    return est;
                return est.WithOnFitDelegate(onFit);
            }
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToValueStaticExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutKeyColumn<TOuterKey, TInnerKey> : Key<TInnerKey>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutKeyColumn(Key<TOuterKey, Key<TInnerKey>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutScalarColumn<TKey, TValue> : Scalar<TValue>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutScalarColumn(Key<TKey, TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TKey, TValue> : Vector<TValue>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<Key<TKey, TValue>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TKey, TValue> : VarVector<TValue>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVarVectorColumn(VarVector<Key<TKey, TValue>> input)
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
                var cols = new (string outputColumnName, string inputColumnName)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (IColInput)toOutput[i];
                    cols[i] = (outputNames[toOutput[i]], inputNames[outCol.Input]);
                }
                return new KeyToValueMappingEstimator(env, cols);
            }
        }

        /// <summary>
        /// Convert a key column to a column containing the corresponding value.
        /// </summary>
        public static Key<TInnerKey> ToValue<TOuterKey, TInnerKey>(this Key<TOuterKey, Key<TInnerKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutKeyColumn<TOuterKey, TInnerKey>(input);
        }

        /// <summary>
        /// Convert a key column to a column containing the corresponding value.
        /// </summary>
        public static Scalar<TValue> ToValue<TKey, TValue>(this Key<TKey, TValue> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalarColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Convert a key column to a column containing the corresponding value.
        /// </summary>
        public static Vector<TValue> ToValue<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Convert a key column to a column containing the corresponding value.
        /// </summary>
        public static VarVector<TValue> ToValue<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey, TValue>(input);
        }
    }

    /// <summary>
    /// The extension methods and implementation support for concatenating columns together.
    /// </summary>
    public static class ConcatStaticExtensions
    {
        /// <summary>
        /// Given a scalar vector, produce a vector of length one.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The scalar column.</param>
        /// <returns>The vector column, whose single item has the same value as the input.</returns>
        public static Vector<T> AsVector<T>(this Scalar<T> me)
            => new Impl<T>(Join(me, (PipelineColumn[])null));

        /// <summary>
        /// Given a bunch of normalized vectors, concatenate them together into a normalized vector.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static NormVector<T> ConcatWith<T>(this NormVector<T> me, params NormVector<T>[] others)
            => new ImplNorm<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns, concatenate them together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static Vector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVector<T>[] others)
            => new Impl<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns, concatenate them together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static Vector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVector<T>[] others)
            => new Impl<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this VarVector<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        private interface IContainsColumn
        {
            PipelineColumn WrappedColumn { get; }
        }

        /// <summary>
        /// A wrapping object for the implicit conversions in <see cref="ConcatWith{T}(Scalar{T}, ScalarOrVector{T}[])"/>
        /// and other related methods.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        public sealed class ScalarOrVector<T> : ScalarOrVectorOrVarVector<T>
        {
            private ScalarOrVector(PipelineColumn col) : base(col) { }
            public static implicit operator ScalarOrVector<T>(Scalar<T> c) => new ScalarOrVector<T>(c);
            public static implicit operator ScalarOrVector<T>(Vector<T> c) => new ScalarOrVector<T>(c);
            public static implicit operator ScalarOrVector<T>(NormVector<T> c) => new ScalarOrVector<T>(c);
        }

        /// <summary>
        /// A wrapping object for the implicit conversions in <see cref="ConcatWith{T}(Scalar{T}, ScalarOrVectorOrVarVector{T}[])"/>
        /// and other related methods.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        public class ScalarOrVectorOrVarVector<T> : IContainsColumn
        {
            public PipelineColumn WrappedColumn { get; }

            private protected ScalarOrVectorOrVarVector(PipelineColumn col)
            {
                Contracts.CheckValue(col, nameof(col));
                WrappedColumn = col;
            }

            public static implicit operator ScalarOrVectorOrVarVector<T>(VarVector<T> c)
               => new ScalarOrVectorOrVarVector<T>(c);
        }

        #region Implementation support
        private sealed class Rec : EstimatorReconciler
        {
            /// <summary>
            /// For the moment the concat estimator can only do one at a time, so I want to apply these operations
            /// one at a time, which means a separate reconciler. Otherwise there may be problems with name overwriting.
            /// If that is ever adjusted, then we can make a slightly more efficient reconciler, though this is probably
            /// not that important of a consideration from a runtime perspective.
            /// </summary>
            public static Rec Inst => new Rec();

            private Rec() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                // For the moment, the concat estimator can only do one concatenation at a time.
                // So we will chain the estimators.
                Contracts.AssertNonEmpty(toOutput);
                IEstimator<ITransformer> est = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var ccol = (IConcatCol)toOutput[i];
                    string[] inputs = ccol.Sources.Select(s => inputNames[s]).ToArray();
                    var localEst = new ColumnConcatenatingEstimator(env, outputNames[toOutput[i]], inputs);
                    if (i == 0)
                        est = localEst;
                    else
                        est = est.Append(localEst);
                }
                return est;
            }
        }

        private static PipelineColumn[] Join(PipelineColumn col, IContainsColumn[] cols)
        {
            if (Utils.Size(cols) == 0)
                return new[] { col };
            var retVal = new PipelineColumn[cols.Length + 1];
            retVal[0] = col;
            for (int i = 0; i < cols.Length; ++i)
                retVal[i + 1] = cols[i].WrappedColumn;
            return retVal;
        }

        private static PipelineColumn[] Join(PipelineColumn col, PipelineColumn[] cols)
        {
            if (Utils.Size(cols) == 0)
                return new[] { col };
            var retVal = new PipelineColumn[cols.Length + 1];
            retVal[0] = col;
            Array.Copy(cols, 0, retVal, 1, cols.Length);
            return retVal;
        }

        private interface IConcatCol
        {
            PipelineColumn[] Sources { get; }
        }

        private sealed class Impl<T> : Vector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public Impl(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }

        private sealed class ImplVar<T> : VarVector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public ImplVar(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }

        private sealed class ImplNorm<T> : NormVector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public ImplNorm(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }
        #endregion
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class NAIndicatorStaticExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutScalar<TValue> : Scalar<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutScalar(Scalar<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TValue> : Vector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TValue> : VarVector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVarVectorColumn(VarVector<TValue> input)
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
                var columnPairs = new (string outputColumnName, string inputColumnName)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    columnPairs[i] = (outputNames[toOutput[i]], inputNames[col.Input]);
                }
                return new MissingValueIndicatorEstimator(env, columnPairs);
            }
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static Scalar<bool> IsMissingValue(this Scalar<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static Scalar<bool> IsMissingValue(this Scalar<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<double>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static Vector<bool> IsMissingValue(this Vector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static Vector<bool> IsMissingValue(this Vector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<double>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static VarVector<bool> IsMissingValue(this VarVector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating whether input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating whether input column entries were missing.</returns>
        public static VarVector<bool> IsMissingValue(this VarVector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<double>(input);
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class TextFeaturizerStaticExtensions
    {
        internal sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Scalar<string>[] Inputs;

            public OutPipelineColumn(IEnumerable<Scalar<string>> inputs, Options options)
                : base(new Reconciler(options), inputs.ToArray())
            {
                Inputs = inputs.ToArray();
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly Options _settings;

            public Reconciler(Options options)
            {
                _settings = options;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutPipelineColumn)toOutput[0];
                var inputs = outCol.Inputs.Select(x => inputNames[x]);
                return new TextFeaturizingEstimator(env, outputNames[outCol], inputs, _settings);
            }
        }
        /// <summary>
        /// Accept text data and converts it to array which represent combinations of ngram/skip-gram token counts.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <param name="otherInputs">Additional data.</param>
        /// <param name="options">Advanced transform settings.</param>
        /// <returns></returns>
        public static Vector<float> FeaturizeText(this Scalar<string> input, Scalar<string>[] otherInputs = null, TextFeaturizingEstimator.Options options = null)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValueOrNull(otherInputs);
            otherInputs = otherInputs ?? new Scalar<string>[0];
            return new OutPipelineColumn(new[] { input }.Concat(otherInputs), options);
        }
    }

    public static class ApproximatedKernelMappingStaticExtenensions
    {
        private readonly struct Config
        {
            public readonly int Rank;
            public readonly bool UseCosAndSinBases;
            public readonly int? Seed;
            public readonly KernelBase Generator;

            public Config(int rank, bool useCosAndSinBases, KernelBase generator, int? seed = null)
            {
                Rank = rank;
                UseCosAndSinBases = useCosAndSinBases;
                Generator = generator;
                Seed = seed;
            }
        }
        private interface IColInput
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplVector<T> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Reconciler.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static readonly Reconciler Inst = new Reconciler();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new ApproximatedKernelMappingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (IColInput)toOutput[i];
                    infos[i] = new ApproximatedKernelMappingEstimator.ColumnOptions(outputNames[toOutput[i]], tcol.Config.Rank, tcol.Config.UseCosAndSinBases, inputNames[tcol.Input], tcol.Config.Generator, tcol.Config.Seed);
                }
                return new ApproximatedKernelMappingEstimator(env, infos);
            }
        }

        /// <summary>
        /// It maps input to a random low-dimensional feature space. It is useful when data has non-linear features, since the transform
        /// is designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user
        /// speciﬁed shift-invariant kernel. With this transform, we are able to use linear methods (which are scalable) to approximate more complex kernel SVM models.
        /// </summary>
        /// <param name="input">The column to apply Random Fourier transfomration.</param>
        /// <param name="rank">The number of random Fourier features to create.</param>
        /// <param name="useCosAndSinBases">If <see langword="true"/>, use both of cos and sin basis functions to create two features for every random Fourier frequency.
        /// Otherwise, only cos bases would be used.</param>
        /// <param name="generator">Which kernel to use. (if it is null, <see cref="GaussianKernel"/> is used.)</param>
        /// <param name="seed">The seed of the random number generator for generating the new features. If not specified global random would be used.</param>
        public static Vector<float> ApproximatedKernelMap(this Vector<float> input,
            int rank = ApproximatedKernelMappingEstimator.Defaults.Rank, bool useCosAndSinBases = ApproximatedKernelMappingEstimator.Defaults.UseCosAndSinBases,
            KernelBase generator = null, int? seed = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(rank, useCosAndSinBases, generator, seed));
        }
    }

    public static class PcaStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, string weightColumn, int rank,
                                     int overSampling, bool ensureZeroMean, int? seed = null)
                : base(new Reconciler(weightColumn, rank, overSampling, ensureZeroMean, seed), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly PrincipalComponentAnalyzer.ColumnOptions _colInfo;

            public Reconciler(string weightColumn, int rank, int overSampling, bool ensureZeroMean, int? seed = null)
            {
                _colInfo = new PrincipalComponentAnalyzer.ColumnOptions(
                    null, null, weightColumn, rank, overSampling, ensureZeroMean, seed);
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutPipelineColumn)toOutput[0];
                var inputColName = inputNames[outCol.Input];
                var outputColName = outputNames[outCol];
                return new PrincipalComponentAnalyzer(env, outputColName, inputColName,
                                         _colInfo.WeightColumn, _colInfo.Rank, _colInfo.Oversampling,
                                         _colInfo.EnsureZeroMean, _colInfo.Seed);
            }
        }

        /// <summary>
        /// Replaces the input vector with its projection to the principal component subspace,
        /// which can significantly reduce size of vector.
        /// </summary>
        /// <include file='../Microsoft.ML.PCA/doc.xml' path='doc/members/member[@name="PCA"]/*'/>
        /// <param name="input">The column to apply PCA to.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="ensureZeroMean">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation</param>
        /// <returns>Vector containing the principal components.</returns>
        public static Vector<float> ProjectToPrincipalComponents(this Vector<float> input,
            string weightColumn = PrincipalComponentAnalyzer.Defaults.WeightColumn,
            int rank = PrincipalComponentAnalyzer.Defaults.Rank,
            int overSampling = PrincipalComponentAnalyzer.Defaults.Oversampling,
            bool ensureZeroMean = PrincipalComponentAnalyzer.Defaults.EnsureZeroMean,
            int? seed = null) => new OutPipelineColumn(input, weightColumn, rank, overSampling, ensureZeroMean, seed);
    }
}
