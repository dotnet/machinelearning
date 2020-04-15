// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
Microsoft.ML.InternalCodeAnalyzer.BaseTestClassAnalyzer,
Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.CodeAnalyzer.Tests.Code
{
    public class BaseTestClassTest
    {
        internal static readonly ReferenceAssemblies ReferenceAssemblies = ReferenceAssemblies.Default
            .AddPackages(ImmutableArray.Create(new PackageIdentity("xunit", "2.4.0")));

        [Fact]
        public async Task TestClassWithFact()
        {
            var code = @"
using Xunit;

public class [|SomeClass|] {
    [Fact]
    public void TestMethod() { }
}
";

            await new VerifyCS.Test
            {
                ReferenceAssemblies = ReferenceAssemblies,
                TestState = { Sources = { code } },
            }.RunAsync();
        }

        [Fact]
        public async Task TestClassWithTheory()
        {
            var code = @"
using Xunit;

public class [|SomeClass|] {
    [Theory, InlineData(0)]
    public void TestMethod(int arg) { }
}
";

            await new VerifyCS.Test
            {
                ReferenceAssemblies = ReferenceAssemblies,
                TestState = { Sources = { code } },
            }.RunAsync();
        }

        [Fact]
        public async Task TestDirectlyExtendsBaseTestClass()
        {
            var code = @"
using Microsoft.ML.TestFramework;
using Xunit;

public class SomeClass : BaseTestClass {
    [Fact]
    public void TestMethod() { }
}

namespace Microsoft.ML.TestFramework {
    public class BaseTestClass { }
}
";

            await new VerifyCS.Test
            {
                ReferenceAssemblies = ReferenceAssemblies,
                TestState = { Sources = { code } },
            }.RunAsync();
        }

        [Fact]
        public async Task TestIndirectlyExtendsBaseTestClass()
        {
            var code = @"
using Microsoft.ML.TestFramework;
using Xunit;

public class SomeClass : IntermediateClass {
    [Fact]
    public void TestMethod() { }
}

public abstract class IntermediateClass : BaseTestClass { }

namespace Microsoft.ML.TestFramework {
    public class BaseTestClass { }
}
";

            await new VerifyCS.Test
            {
                ReferenceAssemblies = ReferenceAssemblies,
                TestState = { Sources = { code } },
            }.RunAsync();
        }
    }
}
