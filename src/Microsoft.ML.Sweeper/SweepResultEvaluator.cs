// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Sweeper;
using ResultProcessor = Microsoft.ML.ResultProcessor;

[assembly: LoadableClass(typeof(InternalSweepResultEvaluator), typeof(InternalSweepResultEvaluator.Options), typeof(SignatureSweepResultEvaluator),
    "TLC Sweep Result Evaluator", "TlcEvaluator", "Tlc")]

namespace Microsoft.ML.Sweeper
{
    public class InternalSweepResultEvaluator : ISweepResultEvaluator<string>
    {
        public sealed class Options
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The sweeper used to get the initial results.", ShortName = "m")]
            public string Metric = "AUC";
        }

        private readonly string _metric;
        private readonly bool _maximizing;

        private readonly IHost _host;

        public InternalSweepResultEvaluator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("InternalSweepResultEvaluator");
            _host.CheckNonEmpty(options.Metric, nameof(options.Metric));
            _metric = FindMetric(options.Metric, out _maximizing);
        }

        private string FindMetric(string userMetric, out bool maximizing)
        {
            StringBuilder sb = new StringBuilder();

            var evaluators = _host.ComponentCatalog.GetAllDerivedClasses(typeof(IMamlEvaluator), typeof(SignatureMamlEvaluator));
            foreach (var evalInfo in evaluators)
            {
                var args = evalInfo.CreateArguments();
                var eval = (IEvaluator)evalInfo.CreateInstance(_host, args, new object[0]);

                sb.AppendFormat("Metrics for {0}:", evalInfo.UserName);
                sb.AppendLine();

                foreach (var metric in eval.GetOverallMetricColumns())
                {
                    string result;
                    if (metric.MetricTarget != MetricColumn.Objective.Info &&
                        (result = metric.GetNameMatch(userMetric)) != null)
                    {
                        maximizing = metric.MetricTarget == MetricColumn.Objective.Maximize;
                        return result;
                    }
                    sb.AppendFormat("{0} ({1})", metric.LoadName, metric.Name);
                    sb.AppendLine();
                    if (metric.CanBeWeighted)
                    {
                        sb.AppendFormat("Weighted{0} ({1})", metric.LoadName, metric.Name);
                        sb.AppendLine();
                    }
                }
            }
            throw _host.Except("Requested metric '{0}' does not exist. Options are:\n{1}", userMetric, sb.ToString());
        }

        public IRunResult GetRunResult(ParameterSet parameterSet, string resultFileName)
        {
            Double result;
            ResultProcessor.ResultProcessor.ProcessResultLines(resultFileName, _metric, out result);
            if (result == 0)
                return new RunResult(parameterSet);

            return new RunResult(parameterSet, result, _maximizing);
        }
    }
}
