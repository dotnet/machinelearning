// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.ApplicationInsights;
using Microsoft.DotNet.Cli.Utils;
using Microsoft.DotNet.PlatformAbstractions;

namespace Microsoft.DotNet.Cli.Telemetry
{
    public class Telemetry : ITelemetry
    {
        private TelemetryClient _client;
        private Dictionary<string, string> _commonProperties;
        private Task _trackEventTask;

        private const string InstrumentationKey = "c059917c-818d-489a-bfcb-351eaab73f2a";
        private const string MlTelemetryOptout = "MLDOTNET_CLI_TELEMETRY_OPTOUT";
        private const string MachineId = "MachineId";

        public bool Enabled { get; }

        public Telemetry()
        {
             _commonProperties = new Dictionary<string, string>();
            var optedOut = Env.GetEnvironmentVariableAsBool(MlTelemetryOptout, false);

            Enabled = !optedOut;

            if (!Enabled)
            {
                return;
            }

            //initialize in task to offload to parallel thread
            _trackEventTask = Task.Factory.StartNew(() => InitializeTelemetry());
        }

        public void TrackEvent(
            string eventName,
            IDictionary<string, string> properties,
            IDictionary<string, double> measurements)
        {
            if (!Enabled)
            {
                return;
            }

            //continue task in existing parallel thread
            _trackEventTask = _trackEventTask.ContinueWith(
                x => TrackEventTask(eventName, properties, measurements)
            );
        }

        public void ThreadBlockingTrackEvent(string eventName, IDictionary<string, string> properties, IDictionary<string, double> measurements)
        {
            if (!Enabled)
            {
                return;
            }

            TrackEventTask(eventName, properties, measurements);
        }

        private void InitializeTelemetry()
        {
            try
            {
                _client = new TelemetryClient();
                _client.InstrumentationKey = InstrumentationKey;
                _client.Context.Device.OperatingSystem = RuntimeEnvironment.OperatingSystem;

                // we don't want hostname etc to be sent in plain text.
                // these need to be set to some non-empty values to override default behavior.
                _client.Context.Cloud.RoleInstance = "-";
                _client.Context.Cloud.RoleName = "-";

                _commonProperties = new TelemetryCommonProperties().GetTelemetryCommonProperties();
            }
            catch (Exception e)
            {
                _client = null;
                // we dont want to fail the tool if telemetry fails.
                Debug.Fail(e.ToString());
            }
        }

        private void TrackEventTask(
            string eventName,
            IDictionary<string, string> properties,
            IDictionary<string, double> measurements)
        {
            if (_client == null)
            {
                return;
            }

            try
            {
                var eventProperties = GetEventProperties(properties);
                var eventMeasurements = GetEventMeasures(measurements);

                _client.TrackEvent(eventName, eventProperties, eventMeasurements);
                _client.Flush();
            }
            catch (Exception e)
            {
                Debug.Fail(e.ToString());
            }
        }

        private Dictionary<string, double> GetEventMeasures(IDictionary<string, double> measurements)
        {
            Dictionary<string, double> eventMeasurements = new Dictionary<string, double>();
            if (measurements != null)
            {
                foreach (KeyValuePair<string, double> measurement in measurements)
                {
                    eventMeasurements[measurement.Key] = measurement.Value;
                }
            }
            return eventMeasurements;
        }

        private Dictionary<string, string> GetEventProperties(IDictionary<string, string> properties)
        {
            if (properties != null)
            {
                var eventProperties = new Dictionary<string, string>(_commonProperties);
                foreach (KeyValuePair<string, string> property in properties)
                {
                    eventProperties[property.Key] = property.Value;
                }
                return eventProperties;
            }
            else
            {
                return _commonProperties;
            }
        }
    }
}