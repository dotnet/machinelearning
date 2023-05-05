// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    public static class SweepableExtension
    {
        public static SweepablePipeline Append(this IEstimator<ITransformer> estimator, SweepableEstimator estimator1)
        {
            return new SweepablePipeline().Append(estimator).Append(estimator1);
        }

        public static SweepablePipeline Append(this SweepablePipeline pipeline, IEstimator<ITransformer> estimator)
        {
            return pipeline.Append(new SweepableEstimator((context, parameter) => estimator, new SearchSpace.SearchSpace()));
        }

        public static SweepablePipeline Append(this SweepableEstimator estimator, SweepablePipeline estimator1)
        {
            return new SweepablePipeline().Append(estimator).Append(estimator1);
        }

        public static SweepablePipeline Append(this SweepableEstimator estimator, IEstimator<ITransformer> estimator1)
        {
            return new SweepablePipeline().Append(estimator).Append(estimator1);
        }

        public static SweepablePipeline Append(this IEstimator<ITransformer> estimator, SweepablePipeline pipeline)
        {
            var sweepableEstimator = new SweepableEstimator((context, parameter) => estimator, new SearchSpace.SearchSpace());
            var res = new SweepablePipeline().Append(sweepableEstimator).Append(pipeline);

            return res;
        }

        public static SweepablePipeline Append(this SweepableEstimator estimator, params SweepableEstimator[] estimators)
        {
            var pipeline = new SweepablePipeline();
            pipeline = pipeline.Append(estimator);

            return pipeline.Append(estimators);
        }

        public static SweepablePipeline Append(this IEstimator<ITransformer> estimator, params SweepableEstimator[] estimators)
        {
            var sweepableEstimator = new SweepableEstimator((context, parameter) => estimator, new SearchSpace.SearchSpace());
            var pipeline = new SweepablePipeline().Append(sweepableEstimator).Append(estimators);

            return pipeline;
        }
    }
}
