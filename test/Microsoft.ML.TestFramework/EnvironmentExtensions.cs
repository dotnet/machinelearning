// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.PCA;

namespace Microsoft.ML.TestFramework
{
    public static class EnvironmentExtensions
    {
        public static TEnvironment AddStandardComponents<TEnvironment>(this TEnvironment env)
            where TEnvironment : IHostEnvironment
        {
            env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly); // ML.Data
            env.ComponentCatalog.RegisterAssembly(typeof(LinearPredictor).Assembly); // ML.StandardLearners
            env.ComponentCatalog.RegisterAssembly(typeof(CategoricalTransform).Assembly); // ML.Transforms
            env.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryPredictor).Assembly); // ML.FastTree
            env.ComponentCatalog.RegisterAssembly(typeof(EnsemblePredictor).Assembly); // ML.Ensemble
            env.ComponentCatalog.RegisterAssembly(typeof(KMeansPredictor).Assembly); // ML.KMeansClustering
            env.ComponentCatalog.RegisterAssembly(typeof(PcaPredictor).Assembly); // ML.PCA
            env.ComponentCatalog.RegisterAssembly(typeof(Experiment).Assembly); // ML.Legacy
            return env;
        }
    }
}
