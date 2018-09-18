// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    public interface IPredictionTransformer<out TModel> : ITransformer
        where TModel : IPredictor
    {
        TModel Model { get; }
    }

    public interface IClassicPredictionTransformer<out TModel> : IPredictionTransformer<TModel>
    where TModel : IPredictor
    {
        string FeatureColumn { get; }

        ColumnType FeatureColumnType { get; }
    }
}
