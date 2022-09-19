// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// interface for all trial runners.
    /// </summary>
    public interface ITrialRunner : IDisposable
    {
        Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct);
    }
}
