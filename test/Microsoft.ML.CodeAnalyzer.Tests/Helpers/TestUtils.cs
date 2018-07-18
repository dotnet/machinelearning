// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
    }
}
