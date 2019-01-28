// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Diagnostics;

namespace Microsoft.ML.Auto
{
    public class AutoFitSettings
    {
        public ExperimentStoppingCriteria StoppingCriteria = new ExperimentStoppingCriteria();
        internal IterationStoppingCriteria IterationStoppingCriteria;
        internal Concurrency Concurrency;
        internal Filters Filters;
        internal CrossValidationSettings CrossValidationSettings;
        internal OptimizingMetric OptimizingMetric;
        internal bool EnableEnsembling;
        internal bool EnableModelExplainability;
        internal bool EnableAutoTransformation;

        // spec question: Are following automatic or a user setting?
        internal bool EnableSubSampling;
        internal bool EnableCaching;
        internal bool ExternalizeTraining;
        internal TraceLevel TraceLevel; // Should this be controlled through code or appconfig?
    }

    public class ExperimentStoppingCriteria
    {
        public int MaxIterations = 100;
        public int TimeOutInMinutes = 300;
        internal bool StopAfterConverging;
        internal double ExperimentExitScore;
    }

    internal class Filters
    {
        internal IEnumerable<Trainers> WhitelistTrainers;
        internal IEnumerable<Trainers> BlackListTrainers;
        internal IEnumerable<Transformers> WhitelistTransformers;
        internal IEnumerable<Transformers> BlacklistTransformers;
        internal bool PreferExplainability;
        internal bool PreferInferenceSpeed;
        internal bool PreferSmallDeploymentSize;
        internal bool PreferSmallMemoryFootprint;
    }

    public class IterationStoppingCriteria
    {
        internal int TimeOutInSeconds;
        internal bool TerminateOnLowAccuracy;
    }

    public class Concurrency
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
