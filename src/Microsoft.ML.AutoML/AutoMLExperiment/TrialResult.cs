// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    public class TrialResult
    {
        public TrialSettings TrialSettings { get; set; }

        public ITransformer Model { get; set; }

        public double Metric { get; set; }

        public bool IsMaximize { get; set; }

        public double DurationInMilliseconds { get; set; }
    }
}
