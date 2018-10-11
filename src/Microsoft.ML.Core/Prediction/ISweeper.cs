// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Signature for the loaders of sweepers.
    /// </summary>
    public delegate void SignatureSweeper();

    /// <summary>
    /// Signature for the loaders of sweep result evaluators.
    /// </summary>
    public delegate void SignatureSweepResultEvaluator();

    /// <summary>
    /// Signature for SuggestedSweeps parser.
    /// </summary>
    public delegate void SignatureSuggestedSweepsParser();

    /// <summary>
    /// The main interface of the sweeper
    /// </summary>
    public interface ISweeper
    {
        /// <summary>
        /// Returns between 0 and maxSweeps configurations to run.
        /// It expects a list of previous runs such that it can generate configurations that were not already tried.
        /// The list of runs can be null if there were no previous runs.
        /// Some smart sweepers can take advantage of the metric(s) that the caller computes for previous runs.
        /// </summary>
        ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null);
    }

    /// <summary>
    /// This is the interface that each type of parameter sweep needs to implement
    /// </summary>
    public interface IValueGenerator
    {
        /// <summary>
        /// Given a value in the [0,1] range, return a value for this parameter.
        /// </summary>
        IParameterValue CreateFromNormalized(Double normalizedValue);

        /// <summary>
        /// Used mainly in grid sweepers, return the i-th distinct value for this parameter
        /// </summary>
        IParameterValue this[int i] { get; }

        /// <summary>
        /// Used mainly in grid sweepers, return the count of distinct values for this parameter
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Returns the name of the generated parameter
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Returns the string representation of this IValueGenerator in a format used by the Sweeper command
        /// </summary>
        string ToStringParameter(IHostEnvironment env);
    }

    public interface ISweepResultEvaluator<in TResults>
    {
        /// <summary>
        /// Return an IRunResult based on the results given as a TResults object.
        /// </summary>
        IRunResult GetRunResult(ParameterSet parameters, TResults results);
    }

    /// <summary>
    /// Parameter value generated from the sweeping.
    /// The parameter values must be immutable.
    /// Value is converted to string because the runner will usually want to construct a command line for TL.
    /// Implementations of this interface must also override object.GetHashCode() and object.Equals(object) so they are consistent
    /// with IEquatable.Equals(IParameterValue).
    /// </summary>
    public interface IParameterValue : IEquatable<IParameterValue>
    {
        string Name { get; }
        string ValueText { get; }
    }

    /// <summary>
    /// Type safe version of the IParameterValue interface.
    /// </summary>
    public interface IParameterValue<out TValue> : IParameterValue
    {
        TValue Value { get; }
    }

    /// <summary>
    /// A set of parameter values.
    /// The parameter set must be immutable.
    /// </summary>
    public sealed class ParameterSet : IEquatable<ParameterSet>, IEnumerable<IParameterValue>
    {
        private readonly Dictionary<string, IParameterValue> _parameterValues;
        private readonly int _hash;

        public ParameterSet(IEnumerable<IParameterValue> parameters)
        {
            _parameterValues = new Dictionary<string, IParameterValue>();
            foreach (var parameter in parameters)
            {
                _parameterValues.Add(parameter.Name, parameter);
            }

            var parameterNames = _parameterValues.Keys.ToList();
            parameterNames.Sort();
            _hash = 0;
            foreach (var parameterName in parameterNames)
            {
                _hash = Hashing.CombineHash(_hash, _parameterValues[parameterName].GetHashCode());
            }
        }

        public ParameterSet(Dictionary<string, IParameterValue> paramValues, int hash)
        {
            _parameterValues = paramValues;
            _hash = hash;
        }

        public IEnumerator<IParameterValue> GetEnumerator()
        {
            return _parameterValues.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public int Count
        {
            get { return _parameterValues.Count; }
        }

        public IParameterValue this[string name]
        {
            get { return _parameterValues[name]; }
        }

        private bool ContainsParamValue(IParameterValue parameterValue)
        {
            IParameterValue value;
            return _parameterValues.TryGetValue(parameterValue.Name, out value) &&
                   parameterValue.Equals(value);
        }

        public bool Equals(ParameterSet other)
        {
            if (other == null || other._hash != _hash || other._parameterValues.Count != _parameterValues.Count)
                return false;
            return other._parameterValues.Values.All(pv => ContainsParamValue(pv));
        }

        public ParameterSet Clone() =>
            new ParameterSet(new Dictionary<string, IParameterValue>(_parameterValues), _hash);

        public override string ToString()
        {
            return string.Join(" ", _parameterValues.Select(kvp => string.Format("{0}={1}", kvp.Value.Name, kvp.Value.ValueText)).ToArray());
        }

        public override int GetHashCode()
        {
            return _hash;
        }
    }

    /// <summary>
    /// The result of a run.
    /// Contains the parameter set used, useful for the sweeper to not generate the same configuration multiple times.
    /// Also contains the result of a run and the metric value that is used by smart sweepers to generate new configurations
    /// that try to maximize this metric.
    /// </summary>
    public interface IRunResult : IComparable<IRunResult>
    {
        ParameterSet ParameterSet { get; }
        IComparable MetricValue { get; }
        bool IsMetricMaximizing { get; }
    }

    public interface IRunResult<T> : IRunResult
        where T : IComparable<T>
    {
        new T MetricValue { get; }
    }

    /// <summary>
    /// Simple implementation of IRunResult
    /// </summary>
    public sealed class RunResult : IRunResult<Double>
    {
        private readonly ParameterSet _parameterSet;
        private readonly Double? _metricValue;
        private readonly bool _isMetricMaximizing;

        /// <summary>
        /// This switch changes the behavior of the CompareTo function, switching the greater than / less than
        /// behavior, depending on if it is set to True.
        /// </summary>
        public bool IsMetricMaximizing { get { return _isMetricMaximizing; } }

        public ParameterSet ParameterSet
        {
            get { return _parameterSet; }
        }

        public RunResult(ParameterSet parameterSet, Double metricValue, bool isMetricMaximizing)
        {
            _parameterSet = parameterSet;
            _metricValue = metricValue;
            _isMetricMaximizing = isMetricMaximizing;
        }

        public RunResult(ParameterSet parameterSet)
        {
            _parameterSet = parameterSet;
        }

        public Double MetricValue
        {
            get
            {
                if (_metricValue == null)
                    throw Contracts.Except("Run result does not contain a metric");
                return _metricValue.Value;
            }
        }

        public int CompareTo(IRunResult other)
        {
            var otherTyped = other as RunResult;
            Contracts.Check(otherTyped != null);
            if (_metricValue == otherTyped._metricValue)
                return 0;
            return _isMetricMaximizing ^ (_metricValue < otherTyped._metricValue) ? 1 : -1;
        }

        public bool HasMetricValue
        {
            get
            {
                return _metricValue != null;
            }
        }

        IComparable IRunResult.MetricValue
        {
            get { return MetricValue; }
        }
    }

    /// <summary>
    /// The metric class, used by smart sweeping algorithms.
    /// Ideally we would like to move towards the new IDataView/ISchematized, this is
    /// just a simple view instead, and it is decoupled from RunResult so we can move
    /// in that direction in the future.
    /// </summary>
    public sealed class RunMetric
    {
        private readonly Float _primaryMetric;
        private readonly Float[] _metricDistribution;

        public RunMetric(Float primaryMetric, IEnumerable<Float> metricDistribution = null)
        {
            _primaryMetric = primaryMetric;
            if (metricDistribution != null)
                _metricDistribution = metricDistribution.ToArray();
        }

        /// <summary>
        /// The primary metric to optimize.
        /// This metric is usually an aggregate value for the run, e.g. AUC, accuracy etc.
        /// By default, smart sweeping algorithms will maximize this metric.
        /// If you want to minimize, either negate this value or change the option in the arguments of the sweeper constructor.
        /// </summary>
        public Float PrimaryMetric
        {
            get { return _primaryMetric; }
        }

        /// <summary>
        /// The (optional) distribution of the metric.
        /// This distribution can be a secondary measure of how good a run was, e.g per-fold AUC, per-fold accuracy, (sampled) per-instance log loss etc.
        /// </summary>
        public Float[] GetMetricDistribution()
        {
            if (_metricDistribution == null)
                return null;
            var result = new Float[_metricDistribution.Length];
            Array.Copy(_metricDistribution, result, _metricDistribution.Length);
            return result;
        }
    }
}
