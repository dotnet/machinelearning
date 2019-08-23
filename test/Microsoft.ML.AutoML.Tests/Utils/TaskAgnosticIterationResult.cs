// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML.Test
{
    internal class TaskAgnosticIterationResult
    {
        internal double PrimaryMetricValue;

        internal Dictionary<string, double> MetricValues = new Dictionary<string, double>();

        internal readonly ITransformer Model;
        internal readonly Exception Exception;
        internal string TrainerName;
        internal double RuntimeInSeconds;
        internal IEstimator<ITransformer> Estimator;
        internal Pipeline Pipeline;
        internal int PipelineInferenceTimeInSeconds;

        private string primaryMetricName;

        private TaskAgnosticIterationResult(RunDetail baseRunDetail, object validationMetrics, string primaryMetricName)
        {
            this.TrainerName = baseRunDetail.TrainerName;
            this.Estimator = baseRunDetail.Estimator;
            this.Pipeline = baseRunDetail.Pipeline;

            this.PipelineInferenceTimeInSeconds = (int)baseRunDetail.PipelineInferenceTimeInSeconds;
            this.RuntimeInSeconds = (int)baseRunDetail.RuntimeInSeconds;

            this.primaryMetricName = primaryMetricName;
            this.PrimaryMetricValue = -1; // default value in case of exception.  TODO: won't work for minimizing metrics, use nullable?

            if (validationMetrics == null)
            {
                return;
            }

            this.MetricValues = MetricValuesToDictionary(validationMetrics);

            this.PrimaryMetricValue = this.MetricValues[this.primaryMetricName];
        }

        public TaskAgnosticIterationResult(RunDetail<RegressionMetrics> runDetail, string primaryMetricName = "RSquared")
            : this(runDetail, runDetail.ValidationMetrics, primaryMetricName)
        {
            if (runDetail.Exception == null)
            {
                this.Model = runDetail.Model;
            }

            this.Exception = runDetail.Exception;
        }

        public TaskAgnosticIterationResult(RunDetail<MulticlassClassificationMetrics> runDetail, string primaryMetricName = "MicroAccuracy")
            : this(runDetail, runDetail.ValidationMetrics, primaryMetricName)
        {
            if (runDetail.Exception == null)
            {
                this.Model = runDetail.Model;
            }

            this.Exception = runDetail.Exception;
        }

        public static Dictionary<string, double> MetricValuesToDictionary<T>(T metric)
        {
            var supportedTypes = new[] { typeof(MulticlassClassificationMetrics), typeof(RegressionMetrics) };

            if (!supportedTypes.Contains(metric.GetType()))
            {
                throw new ArgumentException($"Unsupported metric type {typeof(T).Name}.");
            }

            var propertiesToReport = metric.GetType().GetProperties().Where(p => p.PropertyType == typeof(double));

            return propertiesToReport.ToDictionary(p => p.Name, p => (double)metric.GetType().GetProperty(p.Name).GetValue(metric));
        }
    }
}

