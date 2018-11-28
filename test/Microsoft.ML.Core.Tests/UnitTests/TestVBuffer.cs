// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Xunit;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class TestVBuffer
    {
        [Fact]
        public void TestApplyAt()
        {
            var buffer = new VBuffer<float>(10, 3, new[] { 0.5f, 1.2f, -3.8f }, new[] { 1, 5, 8 });
            VBufferUtils.ApplyAt(ref buffer, 6, (int slot, ref float value) => { value = value + 1; });
            Assert.Equal(4, buffer.GetValues().Length);
            Assert.Equal(1, buffer.GetValues()[2]);
        }
    }
}
