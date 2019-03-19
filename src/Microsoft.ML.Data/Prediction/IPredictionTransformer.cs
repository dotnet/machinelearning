// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// An interface for all the transformer that can transform data based on the <see cref="IPredictor"/> field.
    /// The implemendations of this interface either have no feature column, or have more than one feature column, and cannot implement the
    /// <see cref="ISingleFeaturePredictionTransformer{TModel}"/>, which most of the ML.Net tranformer implement.
    /// </summary>
    /// <typeparam name="TModel">The <see cref="IPredictor"/> or <see cref="ICalibrator"/> used for the data transformation.</typeparam>
    public interface IPredictionTransformer<out TModel> : ITransformer
        where TModel : class
    {
        TModel Model { get; }
    }

    /// <summary>
    /// An ISingleFeaturePredictionTransformer contains the name of the <see cref="FeatureColumnName"/>
    /// and its type, <see cref="FeatureColumnType"/>. Implementations of this interface, have the ability
    /// to score the data of an input <see cref="IDataView"/> through the <see cref="ITransformer.Transform(IDataView)"/>
    /// </summary>
    /// <typeparam name="TModel">The <see cref="IPredictor"/> or <see cref="ICalibrator"/> used for the data transformation.</typeparam>
    public interface ISingleFeaturePredictionTransformer<out TModel> : IPredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>The name of the feature column.</summary>
        string FeatureColumnName { get; }

        /// <summary>Holds information about the type of the feature column.</summary>
        DataViewType FeatureColumnType { get; }
    }
}