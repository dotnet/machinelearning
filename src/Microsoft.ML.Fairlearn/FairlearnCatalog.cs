// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.Fairlearn
{
    public sealed class FairlearnCatalog
    {
        private readonly MLContext _context;
        public FairlearnMetricCatalog Metric;

        internal FairlearnCatalog(MLContext context)
        {
            this._context = context;
            this.Metric = new FairlearnMetricCatalog(context);
        }

    }
}
