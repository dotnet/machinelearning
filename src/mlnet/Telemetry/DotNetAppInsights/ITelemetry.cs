
// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.DotNet.Cli.Telemetry
{
    public interface ITelemetry
    {
        bool Enabled { get; }

        void TrackEvent(string eventName, IDictionary<string, string> properties, IDictionary<string, double> measurements);
    }
}