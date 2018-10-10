// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Benchmarks
{
    internal class Errors
    {
        public static string DatasetNotFound = "Could not find {0} Please ensure you have run 'build.cmd -- /t:DownloadExternalTestFiles /p:IncludeBenchmarkData=true' from the root";
    }
}
