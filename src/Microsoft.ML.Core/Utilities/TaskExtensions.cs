// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.CodeAnalysis;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    internal static class TaskExtensions
    {
        [SuppressMessage("Usage", "VSTHRD002:Avoid problematic synchronous waits", Justification = "The task is completed.")]
        public static TResult CompletedResult<TResult>(this Task<TResult> task)
        {
            Contracts.Check(task.IsCompleted);
            return task.Result;
        }
    }
}
