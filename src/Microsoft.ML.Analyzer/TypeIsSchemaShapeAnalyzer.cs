// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.Analyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class TypeIsSchemaShapeAnalyzer : DiagnosticAnalyzer
    {
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => new ImmutableArray<DiagnosticDescriptor>();

        public override void Initialize(AnalysisContext context)
        {
        }
    }
}
