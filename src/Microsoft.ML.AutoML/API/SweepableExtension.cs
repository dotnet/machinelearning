// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    public static class SweepableExtension
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
            return new SweepableEstimatorPipeline().Append(estimator).Append(estimator1);
        }

        public static MultiModelPipeline Append(this IEstimator<ITransformer> estimator, params SweepableEstimator[] estimators)
        {
            var sweepableEstimator = new SweepableEstimator((context, parameter) => estimator, new SearchSpace.SearchSpace());
            var multiModelPipeline = new MultiModelPipeline().Append(sweepableEstimator).Append(estimators);

            return multiModelPipeline;
        }

        public static MultiModelPipeline Append(this MultiModelPipeline pipeline, IEstimator<ITransformer> estimator)
        {
            var sweepableEstimator = new SweepableEstimator((context, parameter) => estimator, new SearchSpace.SearchSpace());
            var multiModelPipeline = pipeline.Append(sweepableEstimator);

            return multiModelPipeline;
        }

        public static MultiModelPipeline Append(this SweepableEstimatorPipeline pipeline, params SweepableEstimator[] estimators)
        {
            var multiModelPipeline = new MultiModelPipeline();
            foreach (var estimator in pipeline.Estimators)
            {
                multiModelPipeline = multiModelPipeline.Append(estimator);
            }

            return multiModelPipeline.Append(estimators);
        }

        public static MultiModelPipeline Append(this SweepableEstimator estimator, params SweepableEstimator[] estimators)
        {
            var multiModelPipeline = new MultiModelPipeline();
            multiModelPipeline = multiModelPipeline.Append(estimator);

            return multiModelPipeline.Append(estimators);
        }
    }
}
