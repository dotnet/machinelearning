// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Reflection;

namespace Microsoft.DotNet.AutoML
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