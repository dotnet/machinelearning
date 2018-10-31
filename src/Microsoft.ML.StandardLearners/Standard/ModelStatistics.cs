// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(LinearModelStatistics), null, typeof(SignatureLoadModel),
    "Linear Model Statistics",
    LinearModelStatistics.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    /// <summary>
    /// Represents a coefficient statistics object.
    /// </summary>
    public struct CoefficientStatistics
    {
        public readonly string Name;
        public readonly Single Estimate;
        public readonly Single StandardError;
        public readonly Single ZScore;
        public readonly Single PValue;

        public CoefficientStatistics(string name, Single estimate, Single stdError, Single zScore, Single pValue)
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
        public const string LoaderSignature = "LinearModelStats";

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
        private readonly Single _deviance;

        // The deviance of the null hypothesis.
        private readonly Single _nullDeviance;

        // Total count of parameters.
        private readonly int _paramCount;

        // The standard errors of coefficients, including the bias.
        // The standard error of bias is placed at index zero.
        // It could be null when there are too many non-zero weights so that
        // the memory is insufficient to hold the Hessian matrix necessary for the computation
        // of the variance-covariance matrix.
        private readonly VBuffer<Single>? _coeffStdError;

        public long TrainingExampleCount { get { return _trainingExampleCount; } }

        public Single Deviance { get { return _deviance; } }

        public Single NullDeviance { get { return _nullDeviance; } }

        public int ParametersCount { get { return _paramCount; } }

        internal LinearModelStatistics(IHostEnvironment env, long trainingExampleCount, int paramCount, Single deviance, Single nullDeviance)
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

        internal LinearModelStatistics(IHostEnvironment env, long trainingExampleCount, int paramCount, Single deviance, Single nullDeviance, in VBuffer<Single> coeffStdError)
            : this(env, trainingExampleCount, paramCount, deviance, nullDeviance)
        {
            _env.Assert(coeffStdError.Count == _paramCount);
            _coeffStdError = coeffStdError;
        }

        public LinearModelStatistics(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _env = env;
            _env.AssertValue(ctx);

            // *** Binary Format ***
            // int: count of parameters
            // long: count of training examples
            // Single: deviance
            // Single: null deviance
            // bool: whether standard error is included
            // (Conditional) Single[_paramCount]: values of std errors of coefficients
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

            Single[] stdErrorValues = ctx.Reader.ReadFloatArray(_paramCount);
            int length = ctx.Reader.ReadInt32();
            _env.CheckDecode(length >= _paramCount);
            if (length == _paramCount)
            {
                _coeffStdError = new VBuffer<Single>(length, stdErrorValues);
                return;
            }

            _env.Assert(length > _paramCount);
            int[] stdErrorIndices = ctx.Reader.ReadIntArray(_paramCount);
            _coeffStdError = new VBuffer<Single>(length, _paramCount, stdErrorValues, stdErrorIndices);
        }

        public static LinearModelStatistics Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LinearModelStatistics(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
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
            // Single: deviance
            // Single: null deviance
            // bool: whether standard error is included
            // (Conditional) Single[_paramCount]: values of std errors of coefficients
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

            _env.Assert(_coeffStdError.Value.Count == _paramCount);
            ctx.Writer.WriteFloatsNoCount(_coeffStdError.Value.Values, _paramCount);
            ctx.Writer.Write(_coeffStdError.Value.Length);
            if (_coeffStdError.Value.IsDense)
                return;

            ctx.Writer.WriteIntsNoCount(_coeffStdError.Value.Indices, _paramCount);
        }

        public static bool TryGetBiasStatistics(LinearModelStatistics stats, Single bias, out Single stdError, out Single zScore, out Single pValue)
        {
            if (!stats._coeffStdError.HasValue)
            {
                stdError = 0;
                zScore = 0;
                pValue = 0;
                return false;
            }

            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);
            stdError = stats._coeffStdError.Value.Values[0];
            Contracts.Assert(stdError == stats._coeffStdError.Value.GetItemOrDefault(0));
            zScore = bias / stdError;
            pValue = 1 - (Single)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));
            return true;
        }

        private static void GetUnorderedCoefficientStatistics(LinearModelStatistics stats, in VBuffer<Single> weights, in VBuffer<ReadOnlyMemory<char>> names,
            ref VBuffer<Single> estimate, ref VBuffer<Single> stdErr, ref VBuffer<Single> zScore, ref VBuffer<Single> pValue, out ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames)
        {
            if (!stats._coeffStdError.HasValue)
            {
                getSlotNames = null;
                return;
            }

            Contracts.Assert(stats._coeffStdError.Value.Length == weights.Length + 1);

            var estimateValues = estimate.Values;
            if (Utils.Size(estimateValues) < stats.ParametersCount - 1)
                estimateValues = new Single[stats.ParametersCount - 1];
            var stdErrorValues = stdErr.Values;
            if (Utils.Size(stdErrorValues) < stats.ParametersCount - 1)
                stdErrorValues = new Single[stats.ParametersCount - 1];
            var zScoreValues = zScore.Values;
            if (Utils.Size(zScoreValues) < stats.ParametersCount - 1)
                zScoreValues = new Single[stats.ParametersCount - 1];
            var pValueValues = pValue.Values;
            if (Utils.Size(pValueValues) < stats.ParametersCount - 1)
                pValueValues = new Single[stats.ParametersCount - 1];

            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            bool denseStdError = stats._coeffStdError.Value.IsDense;
            int[] stdErrorIndices = stats._coeffStdError.Value.Indices;
            for (int i = 1; i < stats.ParametersCount; i++)
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                Contracts.Assert(0 <= wi && wi < weights.Length);
                var weight = estimateValues[i - 1] = weights.GetItemOrDefault(wi);
                var stdError = stdErrorValues[wi] = stats._coeffStdError.Value.Values[i];
                zScoreValues[i - 1] = weight / stdError;
                pValueValues[i - 1] = 1 - (Single)ProbabilityFunctions.Erf(Math.Abs(zScoreValues[i - 1] / sqrt2));
            }

            estimate = new VBuffer<Single>(stats.ParametersCount - 1, estimateValues, estimate.Indices);
            stdErr = new VBuffer<Single>(stats.ParametersCount - 1, stdErrorValues, stdErr.Indices);
            zScore = new VBuffer<Single>(stats.ParametersCount - 1, zScoreValues, zScore.Indices);
            pValue = new VBuffer<Single>(stats.ParametersCount - 1, pValueValues, pValue.Indices);

            var slotNames = names;
            getSlotNames =
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < stats.ParametersCount - 1)
                        values = new ReadOnlyMemory<char>[stats.ParametersCount - 1];
                    for (int i = 1; i < stats.ParametersCount; i++)
                    {
                        int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                        values[i - 1] = slotNames.GetItemOrDefault(wi);
                    }
                    dst = new VBuffer<ReadOnlyMemory<char>>(stats.ParametersCount - 1, values, dst.Indices);
                };
        }

        private IEnumerable<CoefficientStatistics> GetUnorderedCoefficientStatistics(LinearBinaryPredictor parent, RoleMappedSchema schema)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(parent, nameof(parent));

            if (!_coeffStdError.HasValue)
                yield break;

            var weights = parent.Weights2 as IReadOnlyList<Single>;
            _env.Assert(_paramCount == 1 || weights != null);
            _env.Assert(_coeffStdError.Value.Length == weights.Count + 1);

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, weights.Count, ref names);

            Single[] stdErrorValues = _coeffStdError.Value.Values;
            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            bool denseStdError = _coeffStdError.Value.IsDense;
            int[] stdErrorIndices = _coeffStdError.Value.Indices;
            Single[] zScores = new Single[_paramCount - 1];
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
                var pValue = 1 - (Single)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));
                yield return new CoefficientStatistics(name, weight, stdError, zScore, pValue);
            }
        }

        /// <summary>
        /// Gets the coefficient statistics as an object.
        /// </summary>
        public CoefficientStatistics[] GetCoefficientStatistics(LinearBinaryPredictor parent, RoleMappedSchema schema, int paramCountCap)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(parent, nameof(parent));
            _env.CheckValue(schema, nameof(schema));
            _env.CheckParam(paramCountCap >= 0, nameof(paramCountCap));

            if (paramCountCap > _paramCount)
                paramCountCap = _paramCount;

            Single stdError;
            Single zScore;
            Single pValue;
            var bias = parent.Bias;
            if (!TryGetBiasStatistics(parent.Statistics, bias, out stdError, out zScore, out pValue))
                return null;

            var order = GetUnorderedCoefficientStatistics(parent, schema).OrderByDescending(stat => stat.ZScore).Take(paramCountCap - 1);
            return order.Prepend(new[] { new CoefficientStatistics("(Bias)", bias, stdError, zScore, pValue) }).ToArray();
        }

        public void SaveText(TextWriter writer, LinearBinaryPredictor parent, RoleMappedSchema schema, int paramCountCap)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(writer, nameof(writer));
            _env.AssertValueOrNull(parent);
            _env.AssertValueOrNull(schema);
            writer.WriteLine();
            writer.WriteLine("*** MODEL STATISTICS SUMMARY ***   ");
            writer.WriteLine("Count of training examples:\t{0}", _trainingExampleCount);
            writer.WriteLine("Residual Deviance:         \t{0}", _deviance);
            writer.WriteLine("Null Deviance:             \t{0}", _nullDeviance);
            writer.WriteLine("AIC:                       \t{0}", 2 * _paramCount + _deviance);

            if (parent == null)
                return;

            var coeffStats = GetCoefficientStatistics(parent, schema, paramCountCap);
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

        public void SaveSummaryInKeyValuePairs(LinearBinaryPredictor parent,
            RoleMappedSchema schema, int paramCountCap, List<KeyValuePair<string, object>> resultCollection)
        {
            Contracts.AssertValue(_env);
            _env.AssertValue(resultCollection);

            resultCollection.Add(new KeyValuePair<string, object>("Count of training examples", _trainingExampleCount));
            resultCollection.Add(new KeyValuePair<string, object>("Residual Deviance", _deviance));
            resultCollection.Add(new KeyValuePair<string, object>("Null Deviance", _nullDeviance));
            resultCollection.Add(new KeyValuePair<string, object>("AIC", 2 * _paramCount + _deviance));

            if (parent == null)
                return;

            var coeffStats = GetCoefficientStatistics(parent, schema, paramCountCap);
            if (coeffStats == null)
                return;

            foreach (var coeffStat in coeffStats)
            {
                resultCollection.Add(new KeyValuePair<string, object>(
                    coeffStat.Name,
                    new Single[] { coeffStat.Estimate, coeffStat.StandardError, coeffStat.ZScore, coeffStat.PValue }));
            }
        }

        public void AddStatsColumns(List<IColumn> list, LinearBinaryPredictor parent, RoleMappedSchema schema, in VBuffer<ReadOnlyMemory<char>> names)
        {
            _env.AssertValue(list);
            _env.AssertValueOrNull(parent);
            _env.AssertValue(schema);

            long count = _trainingExampleCount;
            list.Add(RowColumnUtils.GetColumn("Count of training examples", NumberType.I8, ref count));
            var dev = _deviance;
            list.Add(RowColumnUtils.GetColumn("Residual Deviance", NumberType.R4, ref dev));
            var nullDev = _nullDeviance;
            list.Add(RowColumnUtils.GetColumn("Null Deviance", NumberType.R4, ref nullDev));
            var aic = 2 * _paramCount + _deviance;
            list.Add(RowColumnUtils.GetColumn("AIC", NumberType.R4, ref aic));

            if (parent == null)
                return;

            Single biasStdErr;
            Single biasZScore;
            Single biasPValue;
            if (!TryGetBiasStatistics(parent.Statistics, parent.Bias, out biasStdErr, out biasZScore, out biasPValue))
                return;

            var biasEstimate = parent.Bias;
            list.Add(RowColumnUtils.GetColumn("BiasEstimate", NumberType.R4, ref biasEstimate));
            list.Add(RowColumnUtils.GetColumn("BiasStandardError", NumberType.R4, ref biasStdErr));
            list.Add(RowColumnUtils.GetColumn("BiasZScore", NumberType.R4, ref biasZScore));
            list.Add(RowColumnUtils.GetColumn("BiasPValue", NumberType.R4, ref biasPValue));

            var weights = default(VBuffer<Single>);
            parent.GetFeatureWeights(ref weights);
            var estimate = default(VBuffer<Single>);
            var stdErr = default(VBuffer<Single>);
            var zScore = default(VBuffer<Single>);
            var pValue = default(VBuffer<Single>);
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames;
            GetUnorderedCoefficientStatistics(parent.Statistics, in weights, in names, ref estimate, ref stdErr, ref zScore, ref pValue, out getSlotNames);

            var slotNamesCol = RowColumnUtils.GetColumn(MetadataUtils.Kinds.SlotNames,
                new VectorType(TextType.Instance, stdErr.Length), getSlotNames);
            var slotNamesRow = RowColumnUtils.GetRow(null, slotNamesCol);
            var colType = new VectorType(NumberType.R4, stdErr.Length);

            list.Add(RowColumnUtils.GetColumn("Estimate", colType, ref estimate, slotNamesRow));
            list.Add(RowColumnUtils.GetColumn("StandardError", colType, ref stdErr, slotNamesRow));
            list.Add(RowColumnUtils.GetColumn("ZScore", colType, ref zScore, slotNamesRow));
            list.Add(RowColumnUtils.GetColumn("PValue", colType, ref pValue, slotNamesRow));
        }

        private string DecorateProbabilityString(Single probZ)
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
