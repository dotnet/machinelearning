// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.ML.Benchmarks
{
    internal class DataPath
    {
        public static string RootPath = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.Parent.Parent.Parent.Parent.FullName;

        public static string TestDataPath =  Path.Combine(RootPath, @"test/data");

        internal static string GetDataPath(string fileName) => Path.Combine(TestDataPath, fileName);
    }
}
