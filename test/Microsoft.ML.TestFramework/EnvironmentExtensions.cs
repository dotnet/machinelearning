// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.TestFramework
{
    public static class EnvironmentExtensions
    {
        public static TEnvironment AddStandardComponents<TEnvironment>(this TEnvironment env)
            where TEnvironment : IHostEnvironment
        {
            env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly); // ML.Data
            env.ComponentCatalog.RegisterAssembly(typeof(LinearModelParameters).Assembly); // ML.StandardTrainers
            env.ComponentCatalog.RegisterAssembly(typeof(OneHotEncodingTransformer).Assembly); // ML.Transforms
            env.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryModelParameters).Assembly); // ML.FastTree
            env.ComponentCatalog.RegisterAssembly(typeof(EnsembleModelParameters).Assembly); // ML.Ensemble
            env.ComponentCatalog.RegisterAssembly(typeof(KMeansModelParameters).Assembly); // ML.KMeansClustering
            env.ComponentCatalog.RegisterAssembly(typeof(PcaModelParameters).Assembly); // ML.PCA
            env.ComponentCatalog.RegisterAssembly(typeof(CVSplit).Assembly); // ML.EntryPoints
            return env;
        }
    }
}
