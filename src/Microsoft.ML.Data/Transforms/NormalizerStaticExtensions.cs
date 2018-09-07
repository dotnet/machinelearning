// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Extension methods for static pipelines for normalization of data.
    /// </summary>
    public static class NormalizerStaticExtensions
    {
        private const long MaxTrain = Normalizer.Defaults.MaxTrainingExamples;
        private const bool FZ = Normalizer.Defaults.FixZero;

        /// <summary>
        /// Learns an affine function based on the minimum and maximum, so that all values between the minimum and
        /// maximum observed during fitting fall into the range of -1 to 1.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>false</c>, then the observed minimum and maximum during fitting
        /// will map to -1 and 1 respectively, exactly. If however set to <c>true</c>, then 0 will always map to 0.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate that can be called whenever the function is fit, with the learned slopes
        /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.
        /// Note that if values are later transformed that are lower than the minimum, or higher than the maximum,
        /// observed during fitting, that the output values may be outside the range of -1 to 1.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<float> Normalize(
            this Vector<float> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
        }

        /// <summary>
        /// Learns an affine function based on the minimum and maximum, so that all values between the minimum and
        /// maximum observed during fitting fall into the range of -1 to 1.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>false</c>, then the observed minimum and maximum during fitting
        /// will map to -1 and 1 respectively, exactly. If however set to <c>true</c>, then 0 will always map to 0.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
        /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.
        /// Note that if values are later transformed that are lower than the minimum, or higher than the maximum,
        /// observed during fitting, that the output values may be outside the range of -1 to 1.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<double> Normalize(
            this Vector<double> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
        }

        private static NormVector<T> NormalizeByMinMaxCore<T>(Vector<T> input, bool fixZero, long maxTrainingExamples,
            OnFitAffine<ImmutableArray<T>> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) => new Normalizer.MinMaxColumn(src, name, maxTrainingExamples, fixZero), AffineMapper(onFit));
        }

        // We have a slightly different breaking up of categories of normalizers versus the dynamic API. Both the mean-var and
        // CDF normalizers are initialized in the same way because they gather exactly the same statistics, but from the point of
        // view of the static API what is more important is the type of mapping that winds up being computed.

        /// <summary>
        /// Learns an affine function based on the observed mean and standard deviation. This is less susceptible
        /// to outliers as compared to <see cref="Normalize(Vector{float}, bool, long, OnFitAffine{ImmutableArray{float}})"/>.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>true</c> then the offset will always be considered zero.</param>
        /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
        /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
        /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<float> NormalizeByMeanVar(
            this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
        }

        /// <summary>
        /// Learns an affine function based on the observed mean and standard deviation. This is less susceptible
        /// to outliers as compared to <see cref="Normalize(Vector{double}, bool, long, OnFitAffine{ImmutableArray{double}})"/>.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>true</c> then the offset will always be considered zero.</param>
        /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
        /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
        /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<double> NormalizeByMeanVar(
            this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
        }

        /// <summary>
        /// Learns a function based on the cumulative density function of a normal distribution parameterized by
        /// a mean and variance as observed during fitting.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>false</c>, then the learned distributional parameters will be
        /// adjusted in such a way as to ensure that the input 0 maps to the output 0.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
        /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned mean and standard
        /// deviation for all slots.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<float> NormalizeByCumulativeDistribution(
            this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitCumulativeDistribution<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
        }

        /// <summary>
        /// Learns a function based on the cumulative density function of a normal distribution parameterized by
        /// a mean and variance as observed during fitting.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="fixZero">If set to <c>false</c>, then the learned distributional parameters will be
        /// adjusted in such a way as to ensure that the input 0 maps to the output 0.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
        /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned mean and standard
        /// deviation for all slots.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<double> NormalizeByCumulativeDistribution(
            this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitCumulativeDistribution<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
        }

        private static NormVector<T> NormalizeByMVCdfCore<T>(Vector<T> input, bool fixZero, bool useLog, bool useCdf, long maxTrainingExamples, Action<IColumnFunction> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) =>
            {
                if (useLog)
                    return new Normalizer.LogMeanVarColumn(src, name, maxTrainingExamples, useCdf);
                return new Normalizer.MeanVarColumn(src, name, maxTrainingExamples, fixZero, useCdf);
            }, onFit);
        }

        /// <summary>
        /// Learns a function based on a discretization of the input values. The observed values for each slot are
        /// analyzed, and the range of numbers is partitioned into monotonically increasing bins. An attempt is made
        /// to make these bins equal in population, but under some circumstances this may be impossible (e.g., a slot
        /// with a very dominant mode). The way the mapping works is, if there are <c>N</c> bins in a slot, and a value
        /// falls in the range of bin <c>n</c> (indexed from 0), the output value is <c>n / (N - 1)</c>, and then possibly
        /// subtracting off the binned value for what 0 would have been if <paramref name="fixZero"/> is true.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="maxBins">The maximum number of discretization points to learn per slot.</param>
        /// <param name="fixZero">Normally the output is in the range of 0 to 1, but if set to <c>true</c>, then what
        /// would have been the output for a zero input is subtracted off the value.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the bin upper bounds for each slot.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<float> NormalizeByBinning(
            this Vector<float> input, int maxBins = Normalizer.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitBinned<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
        }

        /// <summary>
        /// Learns a function based on a discretization of the input values. The observed values for each slot are
        /// analyzed, and the range of numbers is partitioned into monotonically increasing bins. An attempt is made
        /// to make these bins equal in population, but under some circumstances this may be impossible (e.g., a slot
        /// with a very dominant mode). The way the mapping works is, if there are <c>N</c> bins in a slot, and a value
        /// falls in the range of bin <c>n</c> (indexed from 0), the output value is <c>n / (N - 1)</c>, and then possibly
        /// subtracting off the binned value for what 0 would have been if <paramref name="fixZero"/> is true.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <param name="maxBins">The maximum number of discretization points to learn per slot.</param>
        /// <param name="fixZero">Normally the output is in the range of 0 to 1, but if set to <c>true</c>, then what
        /// would have been the output for a zero input is subtracted off the value.
        /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
        /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
        /// <param name="onFit">A delegate called whenever the estimator is fit, with the bin upper bounds for each slot.</param>
        /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
        /// vector values.</remarks>
        /// <returns>The normalized column.</returns>
        public static NormVector<double> NormalizeByBinning(
            this Vector<double> input, int maxBins = Normalizer.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitBinned<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
        }

        private static NormVector<T> NormalizeByBinningCore<T>(Vector<T> input, int numBins, bool fixZero, long maxTrainingExamples,
            OnFitBinned<ImmutableArray<T>> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(numBins > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) => new Normalizer.BinningColumn(src, name, maxTrainingExamples, fixZero, numBins), BinMapper(onFit));
        }

        /// <summary>
        /// For user provided delegates to receive information when an affine normalizer is fitted.
        /// The function of the normalizer transformer is <c>(input - offset) * scale</c>.
        /// </summary>
        /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
        /// <param name="scale">The scales. In the scalar case, this is a single value. In the vector case this is of length equal
        /// to the number of slots.</param>
        /// <param name="offset">The offsets. In the scalar case, this is a single value. In the vector case this is of length equal
        /// to the number of slots, or of length zero if all the offsets are zero.</param>
        public delegate void OnFitAffine<TData>(TData scale, TData offset);

        /// <summary>
        /// For user provided delegates to receive information when a cumulative distribution function normalizer is fitted.
        /// </summary>
        /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
        /// <param name="mean">The mean value. In the scalar case, this is a single value. In the vector case this is of length equal
        /// to the number of slots.</param>
        /// <param name="standardDeviation">The standard deviation. In the scalar case, this is a single value. In the vector case
        /// this is of length equal to the number of slots.</param>
        public delegate void OnFitCumulativeDistribution<TData>(TData mean, TData standardDeviation);

        /// <summary>
        /// For user provided delegates to receive information when a binning normalizer is fitted.
        /// The function fo the normalizer transformer is, given a value, find its index in the upper bounds, then divide that value
        /// by the number of upper bounds minus 1, so as to scale the index between 0 and 1. Then, if zero had been fixed, subtract
        /// off the value that would have been computed by the above procedure for the value zero.
        /// </summary>
        /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
        /// <param name="upperBounds">For a scalar column a single sequence of the bin upper bounds. For a vector, the same, but
        /// for all slots.</param>
        public delegate void OnFitBinned<TData>(ImmutableArray<TData> upperBounds);

        #region Implementation support
        private delegate Normalizer.ColumnBase CreateNormCol(string input, string name);

        private sealed class Rec : EstimatorReconciler
        {
            // All settings are self contained in the columns.
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var cols = new Normalizer.ColumnBase[toOutput.Length];
                List<(int idx, Action<IColumnFunction> onFit)> onFits = null;

                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (INormColCreator)toOutput[i];
                    cols[i] = col.CreateNormCol(inputNames[col.Input], outputNames[toOutput[i]]);
                    if (col.OnFit != null)
                        Utils.Add(ref onFits, (i, col.OnFit));
                }
                var norm = new Normalizer(env, cols);
                if (Utils.Size(onFits) == 0)
                    return norm;
                return norm.WithOnFitDelegate(normTrans =>
                {
                    Contracts.Assert(normTrans.ColumnFunctions.Count == toOutput.Length);
                    foreach ((int idx, Action<IColumnFunction> onFit) in onFits)
                        onFit(normTrans.ColumnFunctions[idx]);
                });
            }
        }

        private static Action<IColumnFunction> AffineMapper<TData>(OnFitAffine<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizerTransformer.IAffineData<TData>)col;
                onFit(aCol.Scale, aCol.Offset);
            };
        }

        private static Action<IColumnFunction> CdfMapper<TData>(OnFitCumulativeDistribution<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizerTransformer.ICdfData<TData>)col;
                onFit(aCol.Mean, aCol.Stddev);
            };
        }

        private static Action<IColumnFunction> BinMapper<TData>(OnFitBinned<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizerTransformer.IBinData<TData>)col;
                onFit(aCol.UpperBounds);
            };
        }

        private interface INormColCreator
        {
            CreateNormCol CreateNormCol { get; }
            PipelineColumn Input { get; }
            Action<IColumnFunction> OnFit { get; }
        }

        private sealed class Impl<T> : NormVector<T>, INormColCreator
        {
            public PipelineColumn Input { get; }
            public CreateNormCol CreateNormCol { get; }
            public Action<IColumnFunction> OnFit { get; }

            public Impl(Vector<T> input, CreateNormCol del, Action<IColumnFunction> onFitDel)
                : base(Rec.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(del);
                Contracts.AssertValueOrNull(onFitDel);
                Input = input;
                CreateNormCol = del;
                OnFit = onFitDel;
            }
        }
        #endregion
    }
}
