// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Fairlearn
{
    public sealed class FairlearnCatalog
    {
        public FairlearnMetricCatalog Metric;

        internal FairlearnCatalog(MLContext context)
        {
            Metric = new FairlearnMetricCatalog(context);
        }
    }
}
