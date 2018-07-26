// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Reflection;
using System.Threading;
using Microsoft.CodeAnalysis;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    internal static class TestUtils
    {
        public static DiagnosticResult CreateDiagnosticResult(this DiagnosticDescriptor desc, int line, int column, params object[] formatArgs)
        {
            return new DiagnosticResult
            {
                Id = desc.Id,
                Message = string.Format(desc.MessageFormat.ToString(), formatArgs),
                Severity = desc.DefaultSeverity,
                Location = new DiagnosticResultLocation("Test0.cs", line, column),
            };
        }

        public static ref string EnsureSourceLoaded(ref string source, string resourceName)
        {
            if (source == null)
            {
                string loadedSource;
                using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
                using (var reader = new StreamReader(stream))
                    loadedSource = reader.ReadToEnd();
                Interlocked.CompareExchange(ref source, loadedSource, null);
            }
            return ref source;
        }
    }
}
