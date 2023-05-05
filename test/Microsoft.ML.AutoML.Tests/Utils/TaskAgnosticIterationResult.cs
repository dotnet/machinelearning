// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

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

        private string _primaryMetricName;

        private TaskAgnosticIterationResult(RunDetail baseRunDetail, object validationMetrics, string primaryMetricName)
        {
            TrainerName = baseRunDetail.TrainerName;
            Estimator = baseRunDetail.Estimator;
            Pipeline = baseRunDetail.Pipeline;

            PipelineInferenceTimeInSeconds = (int)baseRunDetail.PipelineInferenceTimeInSeconds;
            RuntimeInSeconds = (int)baseRunDetail.RuntimeInSeconds;

            _primaryMetricName = primaryMetricName;
            PrimaryMetricValue = -1; // default value in case of exception.  TODO: won't work for minimizing metrics, use nullable?

            if (validationMetrics == null)
            {
                return;
            }

            MetricValues = MetricValuesToDictionary(validationMetrics);

            PrimaryMetricValue = MetricValues[_primaryMetricName];
        }

        public TaskAgnosticIterationResult(RunDetail<RegressionMetrics> runDetail, string primaryMetricName = "RSquared")
            : this(runDetail, runDetail.ValidationMetrics, primaryMetricName)
        {
            if (runDetail.Exception == null)
            {
                Model = runDetail.Model;
            }

            Exception = runDetail.Exception;
        }

        public TaskAgnosticIterationResult(RunDetail<MulticlassClassificationMetrics> runDetail, string primaryMetricName = "MicroAccuracy")
            : this(runDetail, runDetail.ValidationMetrics, primaryMetricName)
        {
            if (runDetail.Exception == null)
            {
                Model = runDetail.Model;
            }

            Exception = runDetail.Exception;
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

