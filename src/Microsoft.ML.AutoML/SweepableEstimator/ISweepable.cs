// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal interface ISweepable
    {
        public SearchSpace.SearchSpace SearchSpace { get; }
    }

    internal interface ISweepable<out T> : ISweepable
        where T : IEstimator<ITransformer>
    {
        public T BuildFromOption(MLContext context, Parameter parameter);
    }
}
