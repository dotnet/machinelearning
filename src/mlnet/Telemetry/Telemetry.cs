// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ApplicationInsights;
using Microsoft.DotNet.Cli.Telemetry;
using Microsoft.DotNet.Cli.Utils;
using Microsoft.DotNet.Configurer;
using Microsoft.DotNet.PlatformAbstractions;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.Commands;

namespace Microsoft.ML.CLI.Telemetry
{
    /// <summary>
    /// Houses CLI telemetry collection and utility methods.
    /// </summary>
    internal static class Telemetry
    {
        private static TelemetryClient _client;
        private static Dictionary<string, string> _commonProperties;
        private static bool _enabled;
        private static Task _initializationTask;

        private const string InstrumentationKey = "c059917c-818d-489a-bfcb-351eaab73f2a";
        private const string MlTelemetryOptout = "MLDOTNET_CLI_TELEMETRY_OPTOUT";
        private const string MachineId = "MachineId";

        public static void Initialize()
        {
            var optedOut = Env.GetEnvironmentVariableAsBool(MlTelemetryOptout, false);
            _enabled = !optedOut;
            if (!_enabled)
            {
                return;
            }

            // Initialize in task to offload to parallel thread
            _initializationTask = Task.Factory.StartNew(() => InitializeTask());
        }

        /// <summary>
        /// Send telemetry event.
        /// </summary>
        public static void TrackEvent(
            string eventName,
            IDictionary<string, string> properties,
            TimeSpan? duration = null,
            Exception ex = null)
        {
            if (!_enabled)
            {
                return;
            }
            _initializationTask.ContinueWith(x => TrackEventTask(eventName, properties, duration, ex));
        }

        /// <summary>
        /// Flush outstanding telemetry, and wait for the specified timeout for this to complete.
        /// </summary>
        public static void Flush(TimeSpan timeout)
        {
            if (!_enabled || _client == null)
            {
                return;
            }
            Task.Run(() => _client.Flush()).Wait(timeout);
        }

        /// <summary>
        /// Get serialized pipeline to log. Be careful to exclude PII.
        /// </summary>
        public static string GetSanitizedPipelineStr(Pipeline pipeline)
        {
            if (pipeline?.Nodes == null)
            {
                return null;
            }
            var transformNodes = pipeline.Nodes.Where(n => n.NodeType == PipelineNodeType.Transform);
            var trainerNode = pipeline.Nodes.FirstOrDefault(n => n.NodeType == PipelineNodeType.Trainer);
            var sb = new StringBuilder();
            foreach (var transformNode in transformNodes)
            {
                sb.Append(transformNode.Name);
                sb.Append(",");
            }
            if (trainerNode != null)
            {
                sb.Append(trainerNode.Name);
                sb.Append("{");
                var serializedHyperparams = trainerNode.Properties
                    .Where(p => SweepableParams.AllHyperparameterNames.Contains(p.Key))
                    .Select(p => $"{p.Key}: {p.Value}");
                sb.Append(string.Join(", ", serializedHyperparams));
                sb.Append("}");
            }
            return sb.ToString();
        }

        private static void InitializeTask()
        {
            try
            {
                // Since the user didn't yet have a chance to read the above message and decide to opt out,
                // don't log any telemetry on the first invocation.
                if (CheckFirstTimeUse())
                {
                    _enabled = false;
                    return;
                }

                _client = new TelemetryClient();
                _client.InstrumentationKey = InstrumentationKey;
                _client.Context.Device.OperatingSystem = RuntimeEnvironment.OperatingSystem;

                // We don't want hostname etc to be sent in plaintext.
                // These need to be set to some non-empty values to override default behavior.
                _client.Context.Cloud.RoleInstance = "-";
                _client.Context.Cloud.RoleName = "-";

                _commonProperties = new TelemetryCommonProperties().GetTelemetryCommonProperties();
                // Add a session ID to each log sent during the life of this process
                _commonProperties["SessionId"] = Guid.NewGuid().ToString();
            }
            catch (Exception e)
            {
                _client = null;
                // We dont want to fail the tool if telemetry fails.
                Debug.Fail(e.ToString());
            }
        }

        private static void TrackEventTask(
            string eventName,
            IDictionary<string, string> properties,
            TimeSpan? duration,
            Exception ex)
        {
            if (_client == null)
            {
                return;
            }

            try
            {
                var eventProperties = GetEventProperties(properties, duration, ex);
                _client.TrackEvent(eventName, eventProperties);
            }
            catch (Exception e)
            {
                Debug.Fail(e.ToString());
            }
        }

        private static Dictionary<string, string> GetEventProperties(IDictionary<string, string> properties,
            TimeSpan? duration, Exception ex)
        {
            var eventProperties = new Dictionary<string, string>(_commonProperties);

            if (duration != null)
            {
                eventProperties["Duration"] = duration.Value.TotalMilliseconds.ToString();
            }

            if (ex != null)
            {
                eventProperties["Exception"] = GetSanitizedExceptionStr(ex);
            }

            if (properties != null)
            {
                foreach (KeyValuePair<string, string> property in properties)
                {
                    if (property.Value != null)
                    {
                        eventProperties[property.Key] = property.Value;
                    }
                }
            }

            eventProperties["Command"] = CommandDefinitions.AutoTrainCommandName;

            return eventProperties;
        }

        private static bool CheckFirstTimeUse()
        {
            using (IFirstTimeUseNoticeSentinel firstTimeUseNoticeSentinel = new FirstTimeUseNoticeSentinel())
            {
                var firstTimeUse = !firstTimeUseNoticeSentinel.Exists();

                if (firstTimeUse)
                {
                    Console.WriteLine(
@"Welcome to the ML.NET CLI!
--------------------------
Learn more about ML.NET CLI: https://aka.ms/mlnet-cli
Use 'mlnet --help' to see available commands or visit: https://aka.ms/mlnet-cli-docs

Telemetry
---------
The ML.NET CLI tool collects usage data in order to help us improve your experience.
The data is anonymous and doesn't include personal information or data from your datasets.
You can opt-out of telemetry by setting the MLDOTNET_CLI_TELEMETRY_OPTOUT environment variable to '1' or 'true' using your favorite shell.

Read more about ML.NET CLI Tool telemetry: https://aka.ms/mlnet-cli-telemetry
");

                    firstTimeUseNoticeSentinel.CreateIfNotExists();

                    return true;
                }

                return false;
            }
        }

        /// <summary>
        /// Get exception string to log. Exclude the exception message, as it
        /// may contain PII.
        /// </summary>
        private static string GetSanitizedExceptionStr(Exception ex)
        {
            return $@"{ex.GetType()}
{ex.StackTrace}";
        }
    }
}