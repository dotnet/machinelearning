﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Transforms.Categorical;
using System;
using System.Reflection;

namespace Microsoft.ML.Runtime
{
    internal static class AssemblyRegistration
    {
        private static readonly Lazy<bool> _assemblyInitializer = new Lazy<bool>(LoadStandardAssemblies);

        public static void RegisterAssemblies(IHostEnvironment environment)
        {
            // ensure all the assemblies in the Microsoft.ML package have been loaded
            if (!_assemblyInitializer.IsValueCreated)
            {
                _ = _assemblyInitializer.Value;
                Contracts.Assert(_assemblyInitializer.Value);
            }

#pragma warning disable CS0618 // The legacy API that internally uses dependency injection for all calls will be deleted anyway.
            AssemblyLoadingUtils.RegisterCurrentLoadedAssemblies(environment);
#pragma warning restore CS0618
        }

        /// <summary>
        /// Loads all the assemblies in the Microsoft.ML package that contain components.
        /// </summary>
        private static bool LoadStandardAssemblies()
        {
            Assembly apiAssembly = typeof(LambdaTransform).Assembly; // ML.Api
            AssemblyName apiAssemblyName = apiAssembly.GetName();

            _ = typeof(TextLoader).Assembly; // ML.Data
            //_ = typeof(EnsemblePredictor).Assembly); // ML.Ensemble BUG https://github.com/dotnet/machinelearning/issues/1078 Ensemble isn't in a NuGet package
            _ = typeof(FastTreeBinaryModelParameters).Assembly; // ML.FastTree
            _ = typeof(KMeansModelParameters).Assembly; // ML.KMeansClustering
            _ = typeof(Maml).Assembly; // ML.Maml
            _ = typeof(PcaPredictor).Assembly; // ML.PCA
            _ = typeof(SweepCommand).Assembly; // ML.Sweeper
            _ = typeof(OneHotEncodingTransformer).Assembly; // ML.Transforms

            // The following assemblies reference this assembly, so we can't directly reference them
            //_ = typeof(Microsoft.ML.Runtime.Data.LinearPredictor).Assembly); // ML.StandardLearners
            _ = Assembly.Load(new AssemblyName()
            {
                Name = "Microsoft.ML.StandardLearners",
                Version = apiAssemblyName.Version, //assume the same version as ML.Api
            });

            return true;
        }
    }
}
