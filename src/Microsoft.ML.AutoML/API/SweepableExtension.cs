// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    internal static class SweepableExtension
    {
        public static SweepableEstimatorPipeline Append(this IEstimator<ITransformer> estimator, SweepableEstimator estimator1)
        {
            return new SweepableEstimatorPipeline().Append(estimator).Append(estimator1);
        }

        public static SweepableEstimatorPipeline Append(this SweepableEstimatorPipeline pipeline, IEstimator<ITransformer> estimator1)
        {
            return pipeline.Append(new SweepableEstimator((context, parameter) => estimator1, new SearchSpace.SearchSpace()));
        }

        public static SweepableEstimatorPipeline Append(this SweepableEstimator estimator, SweepableEstimator estimator1)
        {
            return new SweepableEstimatorPipeline().Append(estimator).Append(estimator1);
        }

        public static SweepableEstimatorPipeline Append(this SweepableEstimator estimator, IEstimator<ITransformer> estimator1)
        {
            return estimator.Append(estimator1);
        }
    }
}
