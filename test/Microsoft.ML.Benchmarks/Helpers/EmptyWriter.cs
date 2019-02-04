// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;

namespace Microsoft.ML.Benchmarks
{
    // Adding this class to not print anything to the console.
    // This is required for the current version of BenchmarkDotNet
    internal class EmptyWriter : TextWriter
    {
        internal static readonly EmptyWriter Instance = new EmptyWriter();

        public override Encoding Encoding => null;
    }
}
