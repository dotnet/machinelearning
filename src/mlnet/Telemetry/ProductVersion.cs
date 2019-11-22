// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;

namespace Microsoft.ML.CLI.Telemetry
{
    public class Product
    {
        public static readonly string Version = GetProductVersion();

        private static string GetProductVersion()
        {
            var version = typeof(Microsoft.ML.CLI.Program).GetTypeInfo().Assembly.GetCustomAttribute<AssemblyFileVersionAttribute>().Version;

            return version;
        }
    }
}