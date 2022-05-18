// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class ForecastBySsa
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SsaOption param)
        {
            if (param.SeriesLength <= param.WindowSize || param.TrainSize <= 2 * param.WindowSize)
            {
                throw new Exception("ForecastBySsa param check error");
            }

            return context.Forecasting.ForecastBySsa(param.OutputColumnName, param.InputColumnName, param.WindowSize, param.SeriesLength, param.TrainSize, param.Horizon, confidenceLowerBoundColumn: param.ConfidenceLowerBoundColumn, confidenceUpperBoundColumn: param.ConfidenceUpperBoundColumn);
        }
    }
}
