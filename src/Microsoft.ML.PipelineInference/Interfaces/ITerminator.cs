// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Interface defining various stopping criteria for pipeline sweeps.
    /// This could include number of total iterations, compute time,
    /// budget expended, etc.
    /// </summary>
    public interface ITerminator
    {
        bool ShouldTerminate(IEnumerable<PipelinePattern> history);
    }
}