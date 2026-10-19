// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class OnnxAttributeTests : BaseTestClass
    {
        public OnnxAttributeTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void OnnxAttributesRespectProcessArchitecture()
        {
            var fact = new OnnxFactAttribute();
            var theory = new OnnxTheoryAttribute();
            bool isSupported = OnnxFactAttribute.IsOnnxRuntimeSupported;

            Assert.True(Environment.Is64BitProcess || !isSupported);
            Assert.Equal(isSupported, fact.Skip == null);
            Assert.Equal(fact.Skip, theory.Skip);
        }
    }
}
