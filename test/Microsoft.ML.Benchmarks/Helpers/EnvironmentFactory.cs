// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Benchmarks
{
    internal static class EnvironmentFactory
    {
        internal static ConsoleEnvironment CreateClassificationEnvironment<TLoader, TTransformer, TTrainer>()
           where TLoader : IDataReader<IMultiStreamSource>
           where TTransformer : ITransformer
           where TTrainer : ITrainer
        {
            var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance);

            environment.ComponentCatalog.RegisterAssembly(typeof(TLoader).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTransformer).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTrainer).Assembly);

            return environment;
        }

        internal static ConsoleEnvironment CreateRankingEnvironment<TEvaluator, TLoader, TTransformer, TTrainer>()
            where TEvaluator : IEvaluator
            where TLoader : IDataReader<IMultiStreamSource>
            where TTransformer : ITransformer
            where TTrainer : ITrainer
        {
            var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance);

            environment.ComponentCatalog.RegisterAssembly(typeof(TEvaluator).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TLoader).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTransformer).Assembly);
            environment.ComponentCatalog.RegisterAssembly(typeof(TTrainer).Assembly);

            environment.ComponentCatalog.RegisterAssembly(typeof(MissingValueHandlingTransformer).Assembly);

            return environment;
        }
    }
}
