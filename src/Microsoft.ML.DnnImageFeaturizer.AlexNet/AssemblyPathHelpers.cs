// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.Transforms
{
    internal static class AssemblyPathHelpers
    {
        public static string GetExecutingAssemblyLocation()
        {
            string codeBaseUri = typeof(AssemblyPathHelpers).Assembly.CodeBase;
            string path = new Uri(codeBaseUri).AbsolutePath;
            return Directory.GetParent(path).FullName;
        }
    }
}