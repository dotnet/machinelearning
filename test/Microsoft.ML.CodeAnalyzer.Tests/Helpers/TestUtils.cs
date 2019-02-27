﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using System.Threading;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    internal static class TestUtils
    {
        public static ref string EnsureSourceLoaded(ref string source, string resourceName)
        {
            if (source == null)
            {
                string loadedSource = LoadSource(resourceName);
                using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
                using (var reader = new StreamReader(stream))
                    loadedSource = reader.ReadToEnd();
                Interlocked.CompareExchange(ref source, loadedSource, null);
            }
            return ref source;
        }

        public static string LoadSource(string resourceName)
        {
            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
            using (var reader = new StreamReader(stream))
                return reader.ReadToEnd();
        }

        public static Lazy<string> LazySource(string resourceName)
        {
            return new Lazy<string>(() => LoadSource(resourceName), true);
        }
    }
}
