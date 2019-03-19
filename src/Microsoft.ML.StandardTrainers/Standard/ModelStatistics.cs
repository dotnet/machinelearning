// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(ModelStatisticsBase), typeof(LinearModelParameterStatistics), null, typeof(SignatureLoadModel),
    "Linear Model Statistics",
    LinearModelParameterStatistics.LoaderSignature)]

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(ModelStatisticsBase), typeof(ModelStatisticsBase), null, typeof(SignatureLoadModel),
    "Model Statistics",
   ModelStatisticsBase.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Represents a coefficient statistics object containing statistics about the calculated model parameters.
    /// </summary>
    public sealed class CoefficientStatistics
    {
        /// <summary>
        /// The model parameter (bias of weight) for which the statistics are generated.
        /// </summary>
        public readonly float Estimate;

        /// <summary>
        /// The <a href="https://en.wikipedia.org/wiki/Standard_error">standard deviation</a> of the estimate of this model parameter (bias of weight).
        /// </summary>
        public readonly float StandardError;

        /// <summary>
        /// The <a href="https://en.wikipedia.org/wiki/Standard_score">standard score</a> of the estimate of this model parameter (bias of weight).
        /// Quantifies by how much the estimate is above or below the mean.
        /// </summary>
        public readonly float ZScore;

        /// <summary>
        /// The <a href="https://en.wikipedia.org/wiki/P-value">probability value</a> of the estimate of this model parameter (bias of weight).
        /// </summary>
        public readonly float PValue;

        /// <summary>
        /// The index of the feature, in the Features vector, to which this model parameter (bias of weight) corresponds to.
        /// </summary>
        public readonly int Index;

        internal CoefficientStatistics(int featureIndex, float estimate, float stdError, float zScore, float pValue)
        {
            Index = featureIndex;
            Estimate = estimate;
            StandardError = stdError;
            ZScore = zScore;
            PValue = pValue;
        }
    }

    // REVIEW: Reconcile with the stats in OLS learner.
    /// <summary>
    /// The statistics for linear predictor.
    /// </summary>
    public class ModelStatisticsBase : ICanSaveModel
    {
        private protected IHostEnvironment Env;

        // Total count of training examples used to train the model.
        public readonly long TrainingExampleCount;

        // The deviance of this model.
        public readonly float Deviance;

        // The deviance of the null hypothesis.
        public readonly float NullDeviance;

        // Total count of parameters.
        public readonly int ParametersCount;

        internal const string LoaderSignature = "ModelStats";

        internal ModelStatisticsBase(IHostEnvironment env, long trainingExampleCount, int paramCount, float deviance, float nullDeviance)
        {
            Contracts.CheckValue(env, nameof(env));
            Env = env;

            Env.Assert(trainingExampleCount > 0);
            Env.Assert(paramCount > 0);

            ParametersCount = paramCount;
            TrainingExampleCount = trainingExampleCount;
            Deviance = deviance;
            NullDeviance = nullDeviance;
        }

        internal ModelStatisticsBase(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            Env = env;
            Env.AssertValue(ctx);

            // *** Binary Format ***
            // int: count of parameters
            // long: count of training examples
            // float: deviance
            // float: null deviance

            ParametersCount = ctx.Reader.ReadInt32();
            Env.CheckDecode(ParametersCount > 0);

            TrainingExampleCount = ctx.Reader.ReadInt64();
            Env.CheckDecode(TrainingExampleCount > 0);

            Deviance = ctx.Reader.ReadFloat();
            NullDeviance = ctx.Reader.ReadFloat();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(Env);
            Env.CheckValue(ctx, nameof(ctx));
            SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private protected virtual void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary Format ***
            // int: count of parameters
            // long: count of training examples
            // float: deviance
            // float: null deviance

            Env.Assert(ParametersCount > 0);
            ctx.Writer.Write(ParametersCount);

            Env.Assert(TrainingExampleCount > 0);
            ctx.Writer.Write(TrainingExampleCount);

            ctx.Writer.Write(Deviance);
            ctx.Writer.Write(NullDeviance);
        }

        internal virtual void SaveText(TextWriter writer, DataViewSchema.Column featureColumn, int paramCountCap)
        {
            Contracts.AssertValue(Env);
            Env.CheckValue(writer, nameof(writer));

            writer.WriteLine();
            writer.WriteLine("*** MODEL STATISTICS SUMMARY ***   ");
            writer.WriteLine("Count of training examples:\t{0}", TrainingExampleCount);
            writer.WriteLine("Residual Deviance:         \t{0}", Deviance);
            writer.WriteLine("Null Deviance:             \t{0}", NullDeviance);
            writer.WriteLine("AIC:                       \t{0}", 2 * ParametersCount + Deviance);
        }

        /// <summary>
        /// Support method for linear models and <see cref="ICanGetSummaryInKeyValuePairs"/>.
        /// </summary>
        internal virtual void SaveSummaryInKeyValuePairs(DataViewSchema.Column featureColumn, int paramCountCap, List<KeyValuePair<string, object>> resultCollection)
        {
            Contracts.AssertValue(Env);
            Env.AssertValue(resultCollection);

            resultCollection.Add(new KeyValuePair<string, object>("Count of training examples", TrainingExampleCount));
            resultCollection.Add(new KeyValuePair<string, object>("Residual Deviance", Deviance));
            resultCollection.Add(new KeyValuePair<string, object>("Null Deviance", NullDeviance));
            resultCollection.Add(new KeyValuePair<string, object>("AIC", 2 * ParametersCount + Deviance));
        }

        internal virtual DataViewSchema.Annotations MakeStatisticsMetadata(RoleMappedSchema schema, in VBuffer<ReadOnlyMemory<char>> names)
        {
            var builder = new DataViewSchema.Annotations.Builder();

            builder.AddPrimitiveValue("Count of training examples", NumberDataViewType.Int64, TrainingExampleCount);
            builder.AddPrimitiveValue("Residual Deviance", NumberDataViewType.Single, Deviance);
            builder.AddPrimitiveValue("Null Deviance", NumberDataViewType.Single, NullDeviance);
            builder.AddPrimitiveValue("AIC", NumberDataViewType.Single, 2 * ParametersCount + Deviance);

            return builder.ToAnnotations();
        }

        private protected virtual VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MOD STAT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ModelStatisticsBase).Assembly.FullName);
        }
    }

    // REVIEW: Reconcile with the stats in OLS learner.
    /// <summary>
    /// The statistics for linear predictor.
    /// </summary>
    public sealed class LinearModelParameterStatistics : ModelStatisticsBase
    {
        internal new const string LoaderSignature = "LinearModelStats";

        private const int CoeffStatsRefactorVersion = 0x00010002;

        private protected override VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LMODSTAT",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Refactored the stats for the parameters in the base class.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LinearModelParameterStatistics).Assembly.FullName);
        }

        // The standard errors of coefficients, including the bias.
        // The standard error of bias is placed at index zero.
        // It could be null when there are too many non-zero weights so that
        // the memory is insufficient to hold the Hessian matrix necessary for the computation
        // of the variance-covariance matrix.
        private readonly VBuffer<float> _coeffStdError;

        /// <summary>
        /// The weights of the LinearModelParams trained.
        /// </summary>
        private readonly VBuffer<float> _weights;

        /// <summary>
        /// The bias of the LinearModelParams trained.
        /// </summary>
        private readonly float _bias;

        internal LinearModelParameterStatistics(IHostEnvironment env, long trainingExampleCount, int paramCount, float deviance, float nullDeviance,
            in VBuffer<float> coeffStdError, VBuffer<float> weights, float bias)
            : base(env, trainingExampleCount, paramCount, deviance, nullDeviance)
        {
            Env.Assert(trainingExampleCount > 0);
            Env.Assert(paramCount > 0);
            Env.Assert(coeffStdError.Length > 0, nameof(coeffStdError));
            Env.Assert(weights.Length > 0, nameof(weights));

            _coeffStdError = coeffStdError;
            _weights = weights;
            _bias = bias;
        }

        private LinearModelParameterStatistics(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            // *** Binary format ***
            // <base>
            // bool: whether standard error is included
            // float[_paramCount]: values of std errors of coefficients
            // int: length of std errors of coefficients
            // (Conditional)int[_paramCount]: indices of std errors of coefficients

            //backwards compatibility
            if (ctx.Header.ModelVerWritten < CoeffStatsRefactorVersion)
            {
                if (!ctx.Reader.ReadBoolean()) // this was used in the old model to denote whether there were stdErrorValues or not.
                    return;
            }

            float[] stdErrorValues = ctx.Reader.ReadFloatArray(ParametersCount);
            int length = ctx.Reader.ReadInt32();
            env.CheckDecode(length >= ParametersCount);

            if (length == ParametersCount)
            {
                _coeffStdError = new VBuffer<float>(length, stdErrorValues);
            }
            else
            {
                env.Assert(length > ParametersCount);
                int[] stdErrorIndices = ctx.Reader.ReadIntArray(ParametersCount);
                _coeffStdError = new VBuffer<float>(length, ParametersCount, stdErrorValues, stdErrorIndices);
            }

            //read the bias
            _bias = ctx.Reader.ReadFloat();

            //read the weights
            bool isWeightsDense = ctx.Reader.ReadBoolByte();
            var weightsLength = ctx.Reader.ReadInt32();
            var weightsValues = ctx.Reader.ReadFloatArray(weightsLength);

            if (isWeightsDense)
            {
                _weights = new VBuffer<float>(weightsLength, weightsValues);
            }
            else
            {
                int[] weightsIndices = ctx.Reader.ReadIntArray(weightsLength);
                _weights = new VBuffer<float>(weightsLength, weightsLength, stdErrorValues, weightsIndices);
            }
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary Format ***
            // <base>
            // (Conditional) float[_paramCount]: values of std errors of coefficients
            // (Conditional) int: length of std errors of coefficients
            // (Conditional) int[_paramCount]: indices of std errors of coefficients

            base.SaveCore(ctx);

            var coeffStdErrorValues = _coeffStdError.GetValues();
            Env.Assert(coeffStdErrorValues.Length == ParametersCount);
            ctx.Writer.WriteSinglesNoCount(coeffStdErrorValues);
            ctx.Writer.Write(_coeffStdError.Length);
            if (!_coeffStdError.IsDense)
                ctx.Writer.WriteIntsNoCount(_coeffStdError.GetIndices());

            //save the bias
            ctx.Writer.Write(_bias);

            //save the weights
            ctx.Writer.WriteBoolByte(_weights.IsDense);
            ctx.Writer.Write(_weights.Length);
            ctx.Writer.WriteSinglesNoCount(_weights.GetValues());
            if (!_weights.IsDense)
                ctx.Writer.WriteIntsNoCount(_coeffStdError.GetIndices());
        }

        /// <summary>
        /// Computes the standart deviation, Z-Score and p-Value for the value being passed as the bias.
        /// </summary>
        public CoefficientStatistics GetBiasStatisticsForValue(float bias)
        {
            const double sqrt2 = 1.41421356237; // Math.Sqrt(2);
            var stdError = _coeffStdError.GetValues()[0];
            Contracts.Assert(stdError == _coeffStdError.GetItemOrDefault(0));
            var zScore = bias / stdError;
            var pValue = 1.0f - (float)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));

            //int feature index, float estimate, float stdError, float zScore, float pValue
            return new CoefficientStatistics(0, bias, stdError, zScore, pValue);
        }

        /// <summary>
        /// Computes the standart deviation, Z-Score and p-Value for the calculated bias.
        /// </summary>
        public CoefficientStatistics GetBiasStatistics() => GetBiasStatisticsForValue(_bias);

        private void GetUnorderedCoefficientStatistics(in VBuffer<ReadOnlyMemory<char>> names,
            ref VBuffer<float> estimate, ref VBuffer<float> stdErr, ref VBuffer<float> zScore, ref VBuffer<float> pValue, out ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames)
        {
            Contracts.Assert(_coeffStdError.Length == _weights.Length + 1);

            var statisticsCount = ParametersCount - 1;

            var estimateEditor = VBufferEditor.Create(ref estimate, statisticsCount);
            var stdErrorEditor = VBufferEditor.Create(ref stdErr, statisticsCount);
            var zScoreEditor = VBufferEditor.Create(ref zScore, statisticsCount);
            var pValueEditor = VBufferEditor.Create(ref pValue, statisticsCount);

            const double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            bool denseStdError = _coeffStdError.IsDense;
            ReadOnlySpan<int> stdErrorIndices = _coeffStdError.GetIndices();
            ReadOnlySpan<float> coeffStdErrorValues = _coeffStdError.GetValues();
            for (int i = 1; i < ParametersCount; i++)
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                Contracts.Assert(0 <= wi && wi < _weights.Length);
                var weight = estimateEditor.Values[i - 1] = _weights.GetItemOrDefault(wi);
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
                    ReadOnlySpan<int> stdErrorIndices2 = _coeffStdError.GetIndices();
                    for (int i = 1; i <= statisticsCount; i++)
                    {
                        int wi = denseStdError ? i - 1 : stdErrorIndices2[i] - 1;
                        editor.Values[i - 1] = slotNames.GetItemOrDefault(wi);
                    }
                    dst = editor.Commit();
                };
        }

        private List<CoefficientStatistics> GetUnorderedCoefficientStatistics()
        {
            Contracts.AssertValue(Env);

            Env.Assert(_coeffStdError.Length == _weights.Length + 1);

            ReadOnlySpan<float> stdErrorValues = _coeffStdError.GetValues();
            const Double sqrt2 = 1.41421356237; // Math.Sqrt(2);

            List<CoefficientStatistics> result = new List<CoefficientStatistics>(ParametersCount - 1);
            bool denseStdError = _coeffStdError.IsDense;
            ReadOnlySpan<int> stdErrorIndices = _coeffStdError.GetIndices();
            float[] zScores = new float[ParametersCount - 1];
            for (int i = 1; i < ParametersCount; i++) //skip the bias term
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                Env.Assert(0 <= wi && wi < _weights.Length);
                var weight = _weights.GetItemOrDefault(wi);
                var stdError = stdErrorValues[i];
                var zScore = zScores[i - 1] = weight / stdError;
                var pValue = 1 - (float)ProbabilityFunctions.Erf(Math.Abs(zScore / sqrt2));
                result.Add(new CoefficientStatistics(wi, weight, stdError, zScore, pValue));
            }
            return result;
        }

        private string[] GetFeatureNames(DataViewSchema.Column featureColumn)
        {
            var names = default(VBuffer<ReadOnlyMemory<char>>);

            featureColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref names);
            Env.Assert(names.Length > 0, "FeatureColumnName has no metadata.");

            bool denseStdError = _coeffStdError.IsDense;
            ReadOnlySpan<int> stdErrorIndices = _coeffStdError.GetIndices();

            var featureNames = new List<string>();

            for (int i = 1; i < ParametersCount; i++) //skip the bias term
            {
                int wi = denseStdError ? i - 1 : stdErrorIndices[i] - 1;
                Env.Assert(0 <= wi && wi < _weights.Length);
                var name = names.GetItemOrDefault(wi).ToString();
                if (string.IsNullOrEmpty(name))
                    name = $"f{wi}";

                featureNames.Add(name);
            }

            return featureNames.ToArray();

        }

        /// <summary>
        /// Gets the coefficient statistics as an object.
        /// </summary>
        public CoefficientStatistics[] GetWeightsCoefficientStatistics(int paramCountCap)
        {
            Env.CheckParam(paramCountCap >= 0, nameof(paramCountCap));

            if (paramCountCap > ParametersCount)
                paramCountCap = ParametersCount;

            var order = GetUnorderedCoefficientStatistics().OrderByDescending(stat => stat.ZScore).Take(paramCountCap - 1);
            return order.ToArray();
        }

        /// <summary>
        /// Saves the statistics in Text format.
        /// </summary>
        internal override void SaveText(TextWriter writer, DataViewSchema.Column featureColumn, int paramCountCap)
        {
            base.SaveText(writer, featureColumn, paramCountCap);

            var biasStats = GetBiasStatistics();
            var coeffStats = GetWeightsCoefficientStatistics(paramCountCap);
            if (coeffStats == null)
                return;

            var featureNames = GetFeatureNames(featureColumn);
            Env.Assert(featureNames.Length >= 1);

            writer.WriteLine();
            writer.WriteLine("Coefficients statistics:");
            writer.WriteLine("Coefficient    \tEstimate\tStd. Error\tz value  \tPr(>|z|)");

            Func<float, string> decorateProbabilityString = (float probZ) =>
            {
                Contracts.AssertValue(Env);
                Env.Assert(0 <= probZ && probZ <= 1);
                if (probZ < 0.001)
                    return string.Format("{0} ***", probZ);
                if (probZ < 0.01)
                    return string.Format("{0} **", probZ);
                if (probZ < 0.05)
                    return string.Format("{0} *", probZ);
                if (probZ < 0.1)
                    return string.Format("{0} .", probZ);

                return probZ.ToString();
            };

            writer.WriteLine("(Bias)\t{0,-10:G7}\t{1,-10:G7}\t{2,-10:G7}\t{3}",
                           biasStats.Estimate,
                           biasStats.StandardError,
                           biasStats.ZScore,
                           decorateProbabilityString(biasStats.PValue));

            foreach (var coeffStat in coeffStats)
            {
                Env.Assert(coeffStat.Index < featureNames.Length);

                writer.WriteLine("{0,-15}\t{1,-10:G7}\t{2,-10:G7}\t{3,-10:G7}\t{4}",
                            featureNames[coeffStat.Index],
                            coeffStat.Estimate,
                            coeffStat.StandardError,
                            coeffStat.ZScore,
                            decorateProbabilityString(coeffStat.PValue));
            }

            writer.WriteLine("---");
            writer.WriteLine("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
        }

        /// <summary>
        /// Support method for linear models and <see cref="ICanGetSummaryInKeyValuePairs"/>.
        /// </summary>
        internal override void SaveSummaryInKeyValuePairs(DataViewSchema.Column featureColumn, int paramCountCap, List<KeyValuePair<string, object>> resultCollection)
        {
            Env.AssertValue(resultCollection);

            base.SaveSummaryInKeyValuePairs(featureColumn, paramCountCap, resultCollection);

            var biasStats = GetBiasStatistics();
            var coeffStats = GetWeightsCoefficientStatistics(paramCountCap);
            if (coeffStats == null)
                return;

            var featureNames = GetFeatureNames(featureColumn);
            resultCollection.Add(new KeyValuePair<string, object>(
                   "(Bias)",
                   new float[] { biasStats.Estimate, biasStats.StandardError, biasStats.ZScore, biasStats.PValue }));

            foreach (var coeffStat in coeffStats)
            {
                Env.Assert(coeffStat.Index < featureNames.Length);

                resultCollection.Add(new KeyValuePair<string, object>(
                    featureNames[coeffStat.Index],
                    new float[] { coeffStat.Estimate, coeffStat.StandardError, coeffStat.ZScore, coeffStat.PValue }));
            }
        }

        internal override DataViewSchema.Annotations MakeStatisticsMetadata(RoleMappedSchema schema, in VBuffer<ReadOnlyMemory<char>> names)
        {
            Env.AssertValue(schema);

            var builder = new DataViewSchema.Annotations.Builder();
            builder.Add(base.MakeStatisticsMetadata(schema, names), c => true);

            //bias statistics
            var biasStats = GetBiasStatistics();
            builder.AddPrimitiveValue("BiasEstimate", NumberDataViewType.Single, biasStats.Estimate);
            builder.AddPrimitiveValue("BiasStandardError", NumberDataViewType.Single, biasStats.StandardError);
            builder.AddPrimitiveValue("BiasZScore", NumberDataViewType.Single, biasStats.ZScore);
            builder.AddPrimitiveValue("BiasPValue", NumberDataViewType.Single, biasStats.PValue);

            var estimate = default(VBuffer<float>);
            var stdErr = default(VBuffer<float>);
            var zScore = default(VBuffer<float>);
            var pValue = default(VBuffer<float>);
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames;
            GetUnorderedCoefficientStatistics(in names, ref estimate, ref stdErr, ref zScore, ref pValue, out getSlotNames);

            var subMetaBuilder = new DataViewSchema.Annotations.Builder();
            subMetaBuilder.AddSlotNames(stdErr.Length, getSlotNames);
            var subMeta = subMetaBuilder.ToAnnotations();
            var colType = new VectorType(NumberDataViewType.Single, stdErr.Length);

            builder.Add("Estimate", colType, (ref VBuffer<float> dst) => estimate.CopyTo(ref dst), subMeta);
            builder.Add("StandardError", colType, (ref VBuffer<float> dst) => stdErr.CopyTo(ref dst), subMeta);
            builder.Add("ZScore", colType, (ref VBuffer<float> dst) => zScore.CopyTo(ref dst), subMeta);
            builder.Add("PValue", colType, (ref VBuffer<float> dst) => pValue.CopyTo(ref dst), subMeta);

            return builder.ToAnnotations();
        }
    }
}
