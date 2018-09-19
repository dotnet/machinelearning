// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// An interface for all the transformer that can transform data based on the <see cref="IPredictor"/> field.
    /// The implemendations of this interface either have no feature column, or have more than one feature column, and cannot implement the
    /// <see cref="ISingleFeaturePredictionTransformer{TModel}"/>, which most of the ML.Net tranformer implement.
    /// </summary>
    /// <typeparam name="TModel">The <see cref="IPredictor"/> used for the data transformation.</typeparam>
    public interface IPredictionTransformer<out TModel> : ITransformer
        where TModel : IPredictor
    {
        TModel Model { get; }
    }

    /// <summary>
    /// An ISingleFeaturePredictionTransformer contains the name of the <see cref="FeatureColumn"/>
    /// and its type, <see cref="FeatureColumnType"/>. Implementations of this interface, have the ability
    /// to score the data of an input <see cref="IDataView"/> through the <see cref="ITransformer.Transform(IDataView)"/>
    /// </summary>
    /// <typeparam name="TModel">The <see cref="IPredictor"/> used for the data transformation.</typeparam>
    public interface ISingleFeaturePredictionTransformer<out TModel> : IPredictionTransformer<TModel>
    where TModel : IPredictor
    {
        /// <summary>The name of the feature column.</summary>
        string FeatureColumn { get; }

        /// <summary>Holds information about the type of the feature column.</summary>
        ColumnType FeatureColumnType { get; }
    }
}
