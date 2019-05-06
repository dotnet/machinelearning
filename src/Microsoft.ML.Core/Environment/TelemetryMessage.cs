// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// A telemetry message.
    /// </summary>
    [BestFriend]
    internal abstract class TelemetryMessage
    {
        public static TelemetryMessage CreateCommand(string commandName, string commandText)
        {
            return new TelemetryTrace(commandText, commandName, "Command");
        }
        public static TelemetryMessage CreateTrainer(string trainerName, string trainerParams)
        {
            return new TelemetryTrace(trainerParams, trainerName, "Trainer");
        }
        public static TelemetryMessage CreateTransform(string transformName, string transformParams)
        {
            return new TelemetryTrace(transformParams, transformName, "Transform");
        }
        public static TelemetryMessage CreateMetric(string metricName, double metricValue, Dictionary<string, string> properties = null)
        {
            return new TelemetryMetric(metricName, metricValue, properties);
        }
        public static TelemetryMessage CreateException(Exception exception)
        {
            return new TelemetryException(exception);
        }
    }

    /// <summary>
    /// Message with one long text and bunch of small properties (limit on value is ~1020 chars)
    /// </summary>
    [BestFriend]
    internal sealed class TelemetryTrace : TelemetryMessage
    {
        public readonly string Text;
        public readonly string Name;
        public readonly string Type;

        public TelemetryTrace(string text, string name, string type)
        {
            Text = text;
            Name = name;
            Type = type;
        }
    }

    /// <summary>
    /// Message with exception
    /// </summary>
    [BestFriend]
    internal sealed class TelemetryException : TelemetryMessage
    {
        public readonly Exception Exception;
        public TelemetryException(Exception exception)
        {
            Contracts.AssertValue(exception);
            Exception = exception;
        }
    }

    /// <summary>
    /// Message with metric value and it properites
    /// </summary>
    [BestFriend]
    internal sealed class TelemetryMetric : TelemetryMessage
    {
        public readonly string Name;
        public readonly double Value;
        public readonly IDictionary<string, string> Properties;
        public TelemetryMetric(string name, double value, IDictionary<string, string> properties = null)
        {
            Name = name;
            Value = value;
            Properties = properties;
        }
    }
}
