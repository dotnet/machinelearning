// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class RegressionExtensions
    {
        public static IEnumerable<IterationResult<RegressionMetrics>> AutoFit(this RegressionContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, null);
        }
        
        internal static IEnumerable<IterationResult<RegressionMetrics>> AutoFit(this RegressionContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }

            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<RegressionMetrics>(TaskKind.Regression, trainData, label, validationData,
                settings, preFeaturizers, columnPurposes,
                OptimizingMetric.RSquared, debugLogger);

            return autoFitter.Fit();
        }
    }

    public static class BinaryClassificationExtensions
    {
        public static IEnumerable<IterationResult<BinaryClassificationMetrics>> AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, null);
        }

        internal static IEnumerable<IterationResult<BinaryClassificationMetrics>> AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }

            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<BinaryClassificationMetrics>(TaskKind.BinaryClassification, trainData, label, validationData,
                settings, preFeaturizers, columnPurposes, 
                OptimizingMetric.RSquared, debugLogger);
            
            return autoFitter.Fit();
        }
    }

    public static class MulticlassExtensions
    {
        public static IEnumerable<IterationResult<MultiClassClassifierMetrics>> AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, null);
        }

        internal static IEnumerable<IterationResult<MultiClassClassifierMetrics>> AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }
            
            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<MultiClassClassifierMetrics>(TaskKind.MulticlassClassification, trainData, label, validationData,
                settings, preFeaturizers, columnPurposes, OptimizingMetric.RSquared, debugLogger);
            return autoFitter.Fit();
        }
    }

    public class IterationResult<T>
    {
        public readonly T Metrics;
        public readonly ITransformer Model;
        public readonly Exception Exception;
        internal readonly Pipeline Pipeline;

        internal IterationResult(ITransformer model, T metrics, Pipeline pipeline, Exception exception)
        {
            Model = model;
            Metrics = metrics;
            Pipeline = pipeline;
            Exception = exception;
        }
    }
}
