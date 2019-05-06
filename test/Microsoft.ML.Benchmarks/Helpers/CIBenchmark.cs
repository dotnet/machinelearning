// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace Microsoft.ML.Benchmarks.Harness
{
    /// <summary>
    /// This attribute is used to identify the benchmarks 
    /// which we want to run on the CI.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class CIBenchmark : Attribute
    {
    }
}
