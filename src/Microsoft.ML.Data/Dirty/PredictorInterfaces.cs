// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Calibrator;
using Microsoft.ML.Data;

namespace Microsoft.ML.Internal.Internallearn
{

    /// <summary>
    /// Signature for loading from a file name.
    /// </summary>
    public delegate void SignaturePredictorFromFile(string fileName);

    /// <summary>
    /// A generic interface for models that can average parameters from multiple instance of self
    /// </summary>
    [BestFriend]
    internal interface IParameterMixer
    {
        IParameterMixer CombineParameters(IList<IParameterMixer> models);
    }

    /// <summary>
    /// A generic interface for models that can average parameters from multiple instance of self
    /// </summary>
    [BestFriend]
    internal interface IParameterMixer<TOutput>
    {
        IParameterMixer<TOutput> CombineParameters(IList<IParameterMixer<TOutput>> models);
    }

    /// <summary>
    /// Predictor that can specialize for quantile regression. It will produce a <see cref="ISchemaBindableMapper"/>, given
    /// an array of quantiles.
    /// </summary>
    [BestFriend]
    internal interface IQuantileRegressionPredictor
    {
        ISchemaBindableMapper CreateMapper(Double[] quantiles);
    }

    /// <summary>
    /// A generic interface for probability distributions
    /// </summary>
    /// <typeparam name="TResult">Type of statistics result</typeparam>
    [BestFriend]
    internal interface IDistribution<out TResult>
    {
        TResult Minimum { get; }

        TResult Maximum { get; }

        TResult Mean { get; }

        TResult StandardDeviation { get; }
    }

    // REVIEW: How should this quantile stuff work?
    [BestFriend]
    internal interface IQuantileValueMapper
    {
        ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper(float[] quantiles);
    }

    /// <summary>
    /// Interface for quantile distribution
    /// </summary>
    /// <typeparam name="TResult">Type of statistics result</typeparam>
    [BestFriend]
    internal interface IQuantileDistribution<TResult> : IDistribution<TResult>, ISampleableDistribution<TResult>
    {
        TResult Median { get; }

        /// <summary>
        /// Returns an estimate of the p-th quantile, the data value where proportionately p of the data has value
        /// less than or equal to the returned value.
        /// </summary>
        TResult GetQuantile(float p);
    }

    [BestFriend]
    internal interface ISampleableDistribution<TResult> : IDistribution<TResult>
    {
        /// <summary>
        /// Returns Support sample for the distribution.
        /// </summary>
        /// <param name="weights">Weights for the distribution.It will be null if the distribution is uniform.</param>
        /// <returns>Returns Support sample</returns>
        TResult[] GetSupportSample(out TResult[] weights);
    }

    /// <summary>
    /// Predictors that can output themselves in a human-readable text format
    /// </summary>
    [BestFriend]
    internal interface ICanSaveInTextFormat
    {
        void SaveAsText(TextWriter writer, RoleMappedSchema schema);
    }

    /// <summary>
    /// Predictors that can output themselves in the Bing ini format.
    /// </summary>
    [BestFriend]
    internal interface ICanSaveInIniFormat
    {
        void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null);
    }

    /// <summary>
    /// Predictors that can output Summary.
    /// </summary>
    [BestFriend]
    internal interface ICanSaveSummary
    {
        void SaveSummary(TextWriter writer, RoleMappedSchema schema);
    }

    /// <summary>
    /// Predictors that can output Summary in key value pairs.
    /// The content of value 'object' can be any type such as integer, float, string or an array of them.
    /// It is up the caller to check and decide how to consume the values.
    /// </summary>
    [BestFriend]
    internal interface ICanGetSummaryInKeyValuePairs
    {
        /// <summary>
        /// Gets model summary including model statistics (if exists) in key value pairs.
        /// </summary>
        IList<KeyValuePair<string, object>> GetSummaryInKeyValuePairs(RoleMappedSchema schema);
    }

    [BestFriend]
    internal interface ICanGetSummaryAsIRow
    {
        Row GetSummaryIRowOrNull(RoleMappedSchema schema);

        Row GetStatsIRowOrNull(RoleMappedSchema schema);
    }

    [BestFriend]
    internal interface ICanGetSummaryAsIDataView
    {
        IDataView GetSummaryDataView(RoleMappedSchema schema);
    }

    /// <summary>
    /// Predictors that can output themselves in C#/C++ code.
    /// </summary>
    [BestFriend]
    internal interface ICanSaveInSourceCode
    {
        void SaveAsCode(TextWriter writer, RoleMappedSchema schema);
    }

    /// <summary>
    /// Signature for trainers that produce predictors that in turn can be use to score features.
    /// </summary>
    public delegate void SignatureFeatureScorerTrainer();

    /// <summary>
    /// Interface implemented by components that can assign weights to features.
    /// </summary>
    public interface IHaveFeatureWeights
    {
        /// <summary>
        /// Returns the weights for the features.
        /// There should be at most as many weights as there are features.
        /// If there are less weights, it is implied that the remaining features have a weight of zero.
        /// The larger the absolute value of a weights, the more informative/important the feature.
        /// A weights of zero signifies that the feature is not used by the model.
        /// </summary>
        void GetFeatureWeights(ref VBuffer<float> weights);
    }

    /// <summary>
    /// Interface implemented by predictors that can score features.
    /// </summary>
    public interface IPredictorWithFeatureWeights<out TResult> : IHaveFeatureWeights, IPredictorProducing<TResult>
    {
    }

    /// <summary>
    /// Interface for mapping input values to corresponding feature contributions.
    /// This interface is commonly implemented by predictors.
    /// </summary>
    [BestFriend]
    internal interface IFeatureContributionMapper : IPredictor
    {
        /// <summary>
        /// Get a delegate for mapping Contributions to Features.
        /// Result will contain vector with topN positive contributions(if available) and
        /// bottomN negative contributions (if available).
        /// For example linear predictor will have both negative and positive contributions.
        /// For trees we will not have negative contributions, so bottom param will be ignored.
        /// If normalization is requested that resulting values will be normalized to [-1, 1].
        /// </summary>
        ValueMapper<TSrc, VBuffer<float>> GetFeatureContributionMapper<TSrc, TDst>(int top, int bottom, bool normalize);
    }

    /// <summary>
    /// Allows support for feature contribution calculation.
    /// </summary>
    public interface ICalculateFeatureContribution : IPredictor
    {
        FeatureContributionCalculator FeatureContributionClaculator { get; }
    }

    /// <summary>
    /// Support for feature contribution calculation.
    /// </summary>
    public sealed class FeatureContributionCalculator
    {
        internal IFeatureContributionMapper ContributionMapper { get; }
        internal FeatureContributionCalculator(IFeatureContributionMapper contributionMapper) => ContributionMapper = contributionMapper;
    }

    /// <summary>
    /// Interface for predictors that can return a string array containing the label names from the label column they were trained on.
    /// If the training label is a key with text key value metadata, it should return this metadata. The order of the labels should be consistent
    /// with the key values. Otherwise, it returns null.
    /// </summary>
    public interface ICanGetTrainingLabelNames : IPredictor
    {
        string[] GetLabelNamesOrNull(out ColumnType labelType);
    }
}
