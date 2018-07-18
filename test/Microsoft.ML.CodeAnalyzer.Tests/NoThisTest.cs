// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class NoThisTest : DiagnosticVerifier<NoThisAnalyzer>
    {
        [Fact]
        public void NoThis()
        {
            const string test = @"
namespace TestNamespace
{
    public sealed class TypeName
    {
        private readonly int _foo;
        private readonly int moo;
        private readonly int _bar;

        private TypeName(int foo)
        {
            _foo = foo;
        }

        public TypeName() : this(5)
        {
            this.moo = 3;
            this._bar = 2;
            string s = this.ToString();
            bool isEqual = s == this;
            bool isEqual2 = s.Equals(this);
        }
    }
}";

            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(17, 13),
                diag.CreateDiagnosticResult(18, 24),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
