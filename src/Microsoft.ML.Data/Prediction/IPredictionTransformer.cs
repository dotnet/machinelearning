// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime
{
    public interface IPredictionTransformer<TModel> : ITransformer
        where TModel : IPredictor
    {
        string FeatureColumn { get; }

        ColumnType FeatureColumnType { get; }

        TModel Model { get; }
    }

    public interface ICalibratedBinaryPredictor<TCalibrator, TModel>
        where TCalibrator : ICalibrator
        where TModel : IPredictorProducing<float>
    {
        TCalibrator Calibrator { get; }
        TModel Model { get; }
    }
}
