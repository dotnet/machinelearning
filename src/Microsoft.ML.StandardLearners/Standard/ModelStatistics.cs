// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(LinearModelStatistics), null, typeof(SignatureLoadModel),
    "Linear Model Statistics",
    LinearModelStatistics.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Represents a coefficient statistics object.
    /// </summary>
    public readonly struct CoefficientStatistics
    {
        public readonly string Name;
        public readonly float Estimate;
        public readonly float StandardError;
        public readonly float ZScore;
        public readonly float PValue;

        internal CoefficientStatistics(string name, float estimate, float stdError, float zScore, float pValue)
        {
            Contracts.AssertNonEmpty(name);
            Name = name;
            Estimate = estimate;
            StandardError = stdError;
            ZScore = zScore;
            PValue = pValue;
        }
    }

    // REVIEW: Make this class a loadable class and implement ICanSaveModel.
    // REVIEW: Reconcile with the stats in OLS learner.
    /// <summary>
    /// The statistics for linear predictor.
    /// </summary>
    public sealed class LinearModelStatistics : ICanSaveModel
    {
        internal const string LoaderSignature = "LinearModelStats";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LMODSTAT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LinearModelStatistics).Assembly.FullName);
        }

        private readonly IHostEnvironment _env;

        // Total count of training examples used to train the model.
        private readonly long _trainingExampleCount;

        // The deviance of this model.
        private readonly float _deviance;

        // The deviance of the null hypothesis.
        private readonly float _nullDeviance;

        // Total count of parameters.
        private readonly int _paramCount;

        // The standard errors of coefficients, including the bias.
        // The standard error of bias is placed at index zero.
        // It could be null when there are too many non-zero weights so that
        // the memory is insufficient to hold the Hessian matrix necessary for the computation
        // of the variance-covariance matrix.
        private readonly VBuffer<float>? _coeffStdError;

        public long TrainingExampleCount => _trainingExampleCount;

        public float Deviance => _deviance;

        public float NullDeviance => _nullDeviance;

        public int ParametersCount => _paramCount;

        internal LinearModelStatistics(IHostEnvironment env, long trainingExampleCount, int paramCount, float deviance, float nullDeviance)
        {
            Contracts.AssertValue(env);
            env.Assert(trainingExampleCount > 0);
            env.Assert(paramCount > 0);
            _env = env;
            _paramCount = paramCount;
            _trainingExampleCount = trainingExampleCount;
            _deviance = deviance;
            _nullDeviance = nullDeviance;
        }

        internal LinearModelStatistics(IHostEnvironment env, long trainingExampleCount, int paramCount, float deviance, float nullDeviance, in VBuffer<float> coeffStdError)
            : this(env, trainingExampleCount, paramCount, deviance, nullDeviance)
        {
            _env.Assert(coeffStdError.GetValues().Length == _paramCount);
            _coeffStdError = coeffStdError;
        }

        internal LinearModelStatistics(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _env = env;
            _env.AssertValue(ctx);

            // *** Binary Format ***
            // int: count of parameters
            // long: count of training examples
            // float: deviance
            // float: null deviance
            // bool: whether standard error is included
            // (Conditional) float[_paramCount]: values of std errors of coefficients
            // (Conditional) int: length of std errors of coefficients
            // (Conditional) int[_paramCount]: indices of std errors of coefficients

            _paramCount = ctx.Reader.ReadInt32();
            _env.CheckDecode(_paramCount > 0);

            _trainingExampleCount = ctx.Reader.ReadInt64();
            _env.CheckDecode(_trainingExampleCount > 0);

            _deviance = ctx.Reader.ReadFloat();
            _nullDeviance = ctx.Reader.ReadFloat();

            var hasStdErrors = ctx.Reader.ReadBoolean();
            if (!hasStdErrors)
            {
                _env.Assert(_coeffStdError == null);
                return;
            }

            float[] stdErrorValues = ctx.Reader.ReadFloatArray(_paramCount);
            int length = ctx.Reader.ReadInt32();
            _env.CheckDecode(length >= _paramCount);
            if (length == _paramCount)
            {
                _coeffStdError = new VBuffer<float>(length, stdErrorValues);
                return;
            }

            _env.Assert(length > _paramCount);
            int[] stdErrorIndices = ctx.Reader.ReadIntArray(_paramCount);
            _coeffStdError = new VBuffer<float>(length, _paramCount, stdErrorValues, stdErrorIndices);
        }

        internal static LinearModelStatistics Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LinearModelStatistics(env, ctx);
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(ctx, nameof(ctx));
            SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary Format ***
            // int: count of parameters
            // long: count of training examples
            // float: deviance
            // float: null deviance
            // bool: whether standard error is included
            // (Conditional) float[_paramCount]: values of std errors of coefficients
            // (Conditional) int: length of std errors of coefficients
            // (Conditional) int[_paramCount]: indices of std errors of coefficients

            _env.Assert(_paramCount > 0);
            ctx.Writer.Write(_paramCount);

            _env.Assert(_trainingExampleCount > 0);
            ctx.Writer.Write(_trainingExampleCount);

            ctx.Writer.Write(_deviance);
            ctx.Writer.Write(_nullDeviance);

            bool hasStdErrors = _coeffStdError.HasValue;
            ctx.Writer.Write(hasStdErrors);
            if (!hasStdErrors)
                return;

            var coeffStdErrorValues = _coeffStdError.Value.GetValues();
            _env.Assert(coeffStdErrorValues.Length == _paramCount);
            ctx.Writer.WriteSinglesNoCount(coeffStdErrorValues);
            ctx.Writer.Write(_coeffStdError.Value.Length);
            if (_coeffStdError.Value.IsDense)
                return;

            ctx.Writer.WriteIntsNoCount(_coeffStdError.Value.GetIndices());
        }

        /// <summary>
        /// Computes the standart deviation, Z-Score and p-Value.
        /// </summary>
        public static bool TryGetBiasStatistics(LinearModelStatistics stats, float bias, out float stdError, out float zScore, out float pValue)
        {
            if (!stats._coeffStdError.HasValue)
            {
                stdError = 0;
                zScore = 0;
                pValue = 0;
                return false;
            }

            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);
            stdError = stats._coeffStdError.Value.GetValues()[0];
            Contracts.Assert(stdError == stats._coeffStdError.Value.GetItemOrDefault(0));
            zScore = bias / stdError;
            pValue = 1.0f - (float)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));
            return true;
        }

        private static void GetUnorderedCoefficientStatistics(LinearModelStatistics stats, in VBuffer<float> weights, in VBuffer<ReadOnlyMemory<char>> names,
            ref VBuffer<float> estimate, ref VBuffer<float> stdErr, ref VBuffer<float> zScore, ref VBuffer<float> pValue, out ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames)
        {
            if (!stats._coeffStdError.HasValue)
            {
                getSlotNames = null;
                return;
            }

            Contracts.Assert(stats._coeffStdError.Value.Length == weights.Length + 1);

            var statisticsCount = stats.ParametersCount - 1;

            var estimateEditor = VBufferEditor.Create(ref estimate, statisticsCount);
            var stdErrorEditor = VBufferEditor.Create(ref stdErr, statisticsCount);
            var zScoreEditor = VBufferEditor.Create(ref zScore, statisticsCount);
            var pValueEditor = VBufferEditor.Create(ref pValue, statisticsCount);

            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            bool denseStdError = stats._coeffStdError.Value.IsDense;
            ReadOnlySpan<int> stdErrorIndices = stats._coeffStdError.Value.GetIndices();
            ReadOnlySpan<float> coeffStdErrorValues = stats._coeffStdError.Value.GetValues();
            for (int i = 1; i < stats.ParametersCount; i++)
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                Contracts.Assert(0 <= wi && wi < weights.Length);
                var weight = estimateEditor.Values[i - 1] = weights.GetItemOrDefault(wi);
                var stdError = stdErrorEditor.Values[wi] = coeffStdErrorValues[i];
                zScoreEditor.Values[i - 1] = weight / stdError;
                pValueEditor.Values[i - 1] = 1 - (float)ProbabilityFunctions.Erf(Math.Abs(zScoreEditor.Values[i - 1] / sqrt2));
            }

            estimate = estimateEditor.Commit();
            stdErr = stdErrorEditor.Commit();
            zScore = zScoreEditor.Commit();
            pValue = pValueEditor.Commit();

            var slotNames = names;
            getSlotNames =
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, statisticsCount);
                    ReadOnlySpan<int> stdErrorIndices2 = stats._coeffStdError.Value.GetIndices();
                    for (int i = 1; i <= statisticsCount; i++)
                    {
                        int wi = denseStdError ? i - 1 : stdErrorIndices2[i] - 1;
                        editor.Values[i - 1] = slotNames.GetItemOrDefault(wi);
                    }
                    dst = editor.Commit();
                };
        }

        private List<CoefficientStatistics> GetUnorderedCoefficientStatistics(LinearBinaryModelParameters parent, DataViewSchema.Column featureColumn)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(parent, nameof(parent));

            if (!_coeffStdError.HasValue)
                return new List<CoefficientStatistics>();

            var weights = parent.Weights as IReadOnlyList<float>;
            _env.Assert(_paramCount == 1 || weights != null);
            _env.Assert(_coeffStdError.Value.Length == weights.Count + 1);

            var names = default(VBuffer<ReadOnlyMemory<char>>);

            featureColumn.Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref names);
            _env.Assert(names.Length > 0, "FeatureColumn has no metadata.");

            ReadOnlySpan<float> stdErrorValues = _coeffStdError.Value.GetValues();
            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            List<CoefficientStatistics> result = new List<CoefficientStatistics>(_paramCount - 1);
            bool denseStdError = _coeffStdError.Value.IsDense;
            ReadOnlySpan<int> stdErrorIndices = _coeffStdError.Value.GetIndices();
            float[] zScores = new float[_paramCount - 1];
            for (int i = 1; i < _paramCount; i++)
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                _env.Assert(0 <= wi && wi < weights.Count);
                var name = names.GetItemOrDefault(wi).ToString();
                if (string.IsNullOrEmpty(name))
                    name = $"f{wi}";
                var weight = weights[wi];
                var stdError = stdErrorValues[i];
                var zScore = zScores[i - 1] = weight / stdError;
                var pValue = 1 - (float)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));
                result.Add(new CoefficientStatistics(name, weight, stdError, zScore, pValue));
            }
            return result;
        }

        /// <summary>
        /// Gets the coefficient statistics as an object.
        /// </summary>
        public CoefficientStatistics[] GetCoefficientStatistics(LinearBinaryModelParameters parent, DataViewSchema.Column featureColumn, int paramCountCap)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(parent, nameof(parent));
            _env.CheckParam(paramCountCap >= 0, nameof(paramCountCap));

            if (paramCountCap > _paramCount)
                paramCountCap = _paramCount;

            float stdError;
            float zScore;
            float pValue;
            var bias = parent.Bias;
            if (!TryGetBiasStatistics(parent.Statistics, bias, out stdError, out zScore, out pValue))
                return null;

            var order = GetUnorderedCoefficientStatistics(parent, featureColumn).OrderByDescending(stat => stat.ZScore).Take(paramCountCap - 1);
            return order.Prepend(new[] { new CoefficientStatistics("(Bias)", bias, stdError, zScore, pValue) }).ToArray();
        }

        internal void SaveText(TextWriter writer, LinearBinaryModelParameters parent, DataViewSchema.Column featureColumn, int paramCountCap)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(writer, nameof(writer));
            _env.AssertValueOrNull(parent);
            writer.WriteLine();
            writer.WriteLine("*** MODEL STATISTICS SUMMARY ***   ");
            writer.WriteLine("Count of training examples:\t{0}", _trainingExampleCount);
            writer.WriteLine("Residual Deviance:         \t{0}", _deviance);
            writer.WriteLine("Null Deviance:             \t{0}", _nullDeviance);
            writer.WriteLine("AIC:                       \t{0}", 2 * _paramCount + _deviance);

            if (parent == null)
                return;

            var coeffStats = GetCoefficientStatistics(parent, featureColumn, paramCountCap);
            if (coeffStats == null)
                return;

            writer.WriteLine();
            writer.WriteLine("Coefficients statistics:");
            writer.WriteLine("Coefficient    \tEstimate\tStd. Error\tz value  \tPr(>|z|)");

            foreach (var coeffStat in coeffStats)
            {
                writer.WriteLine("{0,-15}\t{1,-10:G7}\t{2,-10:G7}\t{3,-10:G7}\t{4}",
                            coeffStat.Name,
                            coeffStat.Estimate,
                            coeffStat.StandardError,
                            coeffStat.ZScore,
                            DecorateProbabilityString(coeffStat.PValue));
            }

            writer.WriteLine("---");
            writer.WriteLine("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
        }

        /// <summary>
        /// Support method for linear models and <see cref="ICanGetSummaryInKeyValuePairs"/>.
        /// </summary>
        internal void SaveSummaryInKeyValuePairs(LinearBinaryModelParameters parent,
            DataViewSchema.Column featureColumn, int paramCountCap, List<KeyValuePair<string, object>> resultCollection)
        {
            Contracts.AssertValue(_env);
            _env.AssertValue(resultCollection);

            resultCollection.Add(new KeyValuePair<string, object>("Count of training examples", _trainingExampleCount));
            resultCollection.Add(new KeyValuePair<string, object>("Residual Deviance", _deviance));
            resultCollection.Add(new KeyValuePair<string, object>("Null Deviance", _nullDeviance));
            resultCollection.Add(new KeyValuePair<string, object>("AIC", 2 * _paramCount + _deviance));

            if (parent == null)
                return;

            var coeffStats = GetCoefficientStatistics(parent, featureColumn, paramCountCap);
            if (coeffStats == null)
                return;

            foreach (var coeffStat in coeffStats)
            {
                resultCollection.Add(new KeyValuePair<string, object>(
                    coeffStat.Name,
                    new float[] { coeffStat.Estimate, coeffStat.StandardError, coeffStat.ZScore, coeffStat.PValue }));
            }
        }

        internal DataViewSchema.Metadata MakeStatisticsMetadata(LinearBinaryModelParameters parent, RoleMappedSchema schema, in VBuffer<ReadOnlyMemory<char>> names)
        {
            _env.AssertValueOrNull(parent);
            _env.AssertValue(schema);

            var builder = new DataViewSchema.Metadata.Builder();

            builder.AddPrimitiveValue("Count of training examples", NumberDataViewType.Int64, _trainingExampleCount);
            builder.AddPrimitiveValue("Residual Deviance", NumberDataViewType.Single, _deviance);
            builder.AddPrimitiveValue("Null Deviance", NumberDataViewType.Single, _nullDeviance);
            builder.AddPrimitiveValue("AIC", NumberDataViewType.Single, 2 * _paramCount + _deviance);

            if (parent == null)
                return builder.ToMetadata();

            if (!TryGetBiasStatistics(parent.Statistics, parent.Bias, out float biasStdErr, out float biasZScore, out float biasPValue))
                return builder.ToMetadata();

            var biasEstimate = parent.Bias;
            builder.AddPrimitiveValue("BiasEstimate", NumberDataViewType.Single, biasEstimate);
            builder.AddPrimitiveValue("BiasStandardError", NumberDataViewType.Single, biasStdErr);
            builder.AddPrimitiveValue("BiasZScore", NumberDataViewType.Single, biasZScore);
            builder.AddPrimitiveValue("BiasPValue", NumberDataViewType.Single, biasPValue);

            var weights = default(VBuffer<float>);
            parent.GetFeatureWeights(ref weights);
            var estimate = default(VBuffer<float>);
            var stdErr = default(VBuffer<float>);
            var zScore = default(VBuffer<float>);
            var pValue = default(VBuffer<float>);
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames;
            GetUnorderedCoefficientStatistics(parent.Statistics, in weights, in names, ref estimate, ref stdErr, ref zScore, ref pValue, out getSlotNames);

            var subMetaBuilder = new DataViewSchema.Metadata.Builder();
            subMetaBuilder.AddSlotNames(stdErr.Length, getSlotNames);
            var subMeta = subMetaBuilder.ToMetadata();
            var colType = new VectorType(NumberDataViewType.Single, stdErr.Length);

            builder.Add("Estimate", colType, (ref VBuffer<float> dst) => estimate.CopyTo(ref dst), subMeta);
            builder.Add("StandardError", colType, (ref VBuffer<float> dst) => stdErr.CopyTo(ref dst), subMeta);
            builder.Add("ZScore", colType, (ref VBuffer<float> dst) => zScore.CopyTo(ref dst), subMeta);
            builder.Add("PValue", colType, (ref VBuffer<float> dst) => pValue.CopyTo(ref dst), subMeta);

            return builder.ToMetadata();
        }

        private string DecorateProbabilityString(float probZ)
        {
            Contracts.AssertValue(_env);
            _env.Assert(0 <= probZ && probZ <= 1);
            if (probZ < 0.001)
                return string.Format("{0} ***", probZ);
            if (probZ < 0.01)
                return string.Format("{0} **", probZ);
            if (probZ < 0.05)
                return string.Format("{0} *", probZ);
            if (probZ < 0.1)
                return string.Format("{0} .", probZ);

            return probZ.ToString();
        }
    }
}
