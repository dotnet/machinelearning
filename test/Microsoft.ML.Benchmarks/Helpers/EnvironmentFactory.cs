// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Training;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Benchmarks
{
    internal static class EnvironmentFactory
    {
        internal static MLContext CreateClassificationEnvironment<TLoader, TTransformer, TTrainer>()
           where TLoader : IDataReader<IMultiStreamSource>
           where TTransformer : ITransformer
           where TTrainer : ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>
        {
            var ctx = new MLContext();
            IHostEnvironment environment = ctx;

            environment.ComponentCatalog.RegisterAssembly(typeof(TLoader).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTransformer).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTrainer).Assembly);

            return ctx;
        }

        internal static MLContext CreateRankingEnvironment<TEvaluator, TLoader, TTransformer, TTrainer>()
            where TLoader : IDataReader<IMultiStreamSource>
            where TTransformer : ITransformer
            where TTrainer : ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>
        {
            var ctx = new MLContext();
            IHostEnvironment environment = ctx;

            environment.ComponentCatalog.RegisterAssembly(typeof(TEvaluator).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TLoader).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTransformer).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTrainer).Assembly);

            environment.ComponentCatalog.RegisterAssembly(typeof(MissingValueDroppingTransformer).Assembly);

            return ctx;
        }
    }
}
