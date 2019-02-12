// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Diagnostics;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitDefaults
    {
        public const uint TimeoutInSeconds = 60 * 60;
        public const uint MaxIterations = 1000;
    }

    internal class AutoFitSettings
    {
        // All the following settings only capture the surface area of capabilities we want to ship in future.
        // However, most certainly they will not ship using following types and structures
        // These should remain internal until we have rationalized 

        public ExperimentStoppingCriteria StoppingCriteria = new ExperimentStoppingCriteria();
        internal IterationStoppingCriteria IterationStoppingCriteria;
        internal Concurrency Concurrency;
        internal Filters Filters;
        internal CrossValidationSettings CrossValidationSettings;
        internal OptimizingMetric OptimizingMetric;
        internal bool DisableEnsembling;
        internal bool CaclculateModelExplainability;
        internal bool DisableFeaturization;

        internal bool DisableSubSampling;
        internal bool DisableCaching;
        internal bool ExternalizeTraining;
        internal TraceLevel TraceLevel;
    }

    internal class ExperimentStoppingCriteria
    {
        public uint TimeoutInSeconds = AutoFitDefaults.TimeoutInSeconds;
        public uint MaxIterations = AutoFitDefaults.MaxIterations;
        internal bool StopAfterConverging;
        internal double ExperimentExitScore;
    }

    internal class Filters
    {
        internal IEnumerable<Trainers> WhitelistTrainers;
        internal IEnumerable<Trainers> BlackListTrainers;
        internal IEnumerable<Transformers> WhitelistTransformers;
        internal IEnumerable<Transformers> BlacklistTransformers;
        internal uint? Explainability;
        internal uint? InferenceSpeed;
        internal uint? DeploymentSize;
        internal uint? TrainingMemorySize;
        internal bool? GpuTraining;
    }

    internal class IterationStoppingCriteria
    {
        internal int TimeOutInSeconds;
        internal bool TerminateOnLowAccuracy;
    }

    internal class Concurrency
    {
        internal int MaxConcurrentIterations;
        internal int MaxCoresPerIteration;
    }

    internal enum Trainers
    {
    }

    internal enum Transformers
    {
    }

    internal class CrossValidationSettings
    {
        internal int NumberOfFolds;
        internal int ValidationSizePercentage;
        internal IEnumerable<string> StratificationColumnNames;
    }
}
