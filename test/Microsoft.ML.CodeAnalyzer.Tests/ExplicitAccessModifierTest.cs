// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class ExplicitAccessModifierTest : DiagnosticVerifier<ExplicitAccessModifierAnalyzer>
    {
        [Fact]
        public void ExplicitAccessModifier()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(5, 11, "TypeName"),
                diag.CreateDiagnosticResult(8, 13, "Foo"),
                diag.CreateDiagnosticResult(12, 27, "Hi"),
                diag.CreateDiagnosticResult(19, 9, "TypeName"),
                diag.CreateDiagnosticResult(30, 26, "A"),
                diag.CreateDiagnosticResult(32, 17, "C"),
                diag.CreateDiagnosticResult(33, 34, "_d"),
                diag.CreateDiagnosticResult(36, 40, "Yo"),
                diag.CreateDiagnosticResult(38, 80, "Dog"),
                diag.CreateDiagnosticResult(40, 14, "Enum1"),
                diag.CreateDiagnosticResult(46, 14, "Silly"),
                diag.CreateDiagnosticResult(51, 34, "-"),
                diag.CreateDiagnosticResult(59, 15, "ITest2"),
            };

            VerifyCSharpDiagnostic(TestSource, expected);
        }

        internal const string TestSource = @"
using System;

namespace TestNamespace
{
    class TypeName : IEquatable<TypeName>
    {
        // Let's talk about ice-cream.
        int Foo;
        private int _bar;


        static public int Hi;

        static TypeName()
        {
            Hi = 2;
        }

        TypeName() : this(5)
        {
        }

        public TypeName(int value)
        {
            _bar = Foo = value;
        }

        public struct Bubba
        {
            readonly int A;
            public int B { get; }
            int C { get; set; }
            readonly private int _d;
        }

        protected static internal void Yo() {}
        // Hello
        static /* So then I says to Mabel, I says... */ protected internal int Dog() { return 1; }

        enum Enum1 { A, B, C }
        public enum Enum2 { A, B, C }

        bool IEquatable<TypeName>.Equals(TypeName other)
            => other != null && other.Foo == Foo && other._bar == _bar;

        bool Silly() => false;

        public bool Billy() => true;

        public static TypeName operator +(TypeName a, TypeName b) => new TypeName(1);
        static TypeName operator -(TypeName a, TypeName b) => new TypeName(2);
    }

    public interface ITest1
    {
        int GetStuff();
    }

    interface ITest2
    {
        int GetStuffPlus1(ITest1 other);
    }
}";
    }

    public sealed class ExplicitAccessModifierFixTest : CodeFixVerifier<ExplicitAccessModifierAnalyzer, ExplicitAccessModifierFixProvider>
    {
        [Fact]
        public void ExplicitAccessModifierFix()
        {
            VerifyCSharpFix("namespace Bubba { class Foo {}}", "namespace Bubba { internal class Foo {}}");
            VerifyCSharpFix(ExplicitAccessModifierTest.TestSource, ExpectedFix);
        }

        private const string ExpectedFix = @"using System;

namespace TestNamespace
{
    internal class TypeName : IEquatable<TypeName>
    {
        // Let's talk about ice-cream.
        private int Foo;
        private int _bar;


        public static int Hi;

        static TypeName()
        {
            Hi = 2;
        }

        private TypeName() : this(5)
        {
        }

        public TypeName(int value)
        {
            _bar = Foo = value;
        }

        public struct Bubba
        {
            private readonly int A;
            public int B { get; }
            private int C { get; set; }
            private readonly int _d;
        }

        protected internal static void Yo() {}
        // Hello
        protected /* So then I says to Mabel, I says... */ internal static int Dog() { return 1; }

        private enum Enum1 { A, B, C }
        public enum Enum2 { A, B, C }

        bool IEquatable<TypeName>.Equals(TypeName other)
            => other != null && other.Foo == Foo && other._bar == _bar;

        private bool Silly() => false;

        public bool Billy() => true;

        public static TypeName operator +(TypeName a, TypeName b) => new TypeName(1);
        private static TypeName operator -(TypeName a, TypeName b) => new TypeName(2);
    }

    public interface ITest1
    {
        int GetStuff();
    }

    internal interface ITest2
    {
        int GetStuffPlus1(ITest1 other);
    }
}";
    }
}