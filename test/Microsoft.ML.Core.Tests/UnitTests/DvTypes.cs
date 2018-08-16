// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Xunit;
namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class DvTypeTests
    {
        [Fact]
        public void TestComparableInt32()
        {
            const int count = 100;

            var rand = RandomUtils.Create(42);
            var values = new int?[2 * count];
            for (int i = 0; i < count; i++)
            {
                var v = values[i] = rand.Next();
                values[values.Length - i - 1] = v;
            }

            // Assign two NA's at random.
            int iv1 = rand.Next(values.Length);
            int iv2 = rand.Next(values.Length - 1);
            if (iv2 >= iv1)
                iv2++;
            values[iv1] = null;
            values[iv2] = null;
            Array.Sort(values);

            Assert.True(!values[0].HasValue);
            Assert.True(!values[1].HasValue);
            Assert.True(values[2].HasValue);
        }

        [Fact]
        public void TestComparableDvText()
        {
            const int count = 100;

            var rand = RandomUtils.Create(42);
            var chars = new char[2000];
            for (int i = 0; i < chars.Length; i++)
                chars[i] = (char)rand.Next(128);
            var str = new string(chars);

            var values = new DvText[2 * count];
            for (int i = 0; i < count; i++)
            {
                int len = rand.Next(20);
                int ich = rand.Next(str.Length - len + 1);
                var v = values[i] = new DvText(str, ich, ich + len);
                values[values.Length - i - 1] = v;
            }

            // Assign two NA's and an empty at random.
            int iv1 = rand.Next(values.Length);
            int iv2 = rand.Next(values.Length - 1);
            if (iv2 >= iv1)
                iv2++;
            int iv3 = rand.Next(values.Length - 2);
            if (iv3 >= iv1)
                iv3++;
            if (iv3 >= iv2)
                iv3++;

            values[iv1] = DvText.NA;
            values[iv2] = DvText.NA;
            values[iv3] = DvText.Empty;
            Array.Sort(values);

            Assert.True(values[0].IsNA);
            Assert.True(values[1].IsNA);
            Assert.True(values[2].IsEmpty);

            Assert.True((values[0] == values[1]).IsNA);
            Assert.True((values[0] != values[1]).IsNA);
            Assert.True(values[0].Equals(values[1]));
            Assert.True(values[0].CompareTo(values[1]) == 0);

            Assert.True((values[1] == values[2]).IsNA);
            Assert.True((values[1] != values[2]).IsNA);
            Assert.True(!values[1].Equals(values[2]));
            Assert.True(values[1].CompareTo(values[2]) < 0);

            for (int i = 3; i < values.Length; i++)
            {
                DvBool eq = values[i - 1] == values[i];
                DvBool ne = values[i - 1] != values[i];
                bool feq = values[i - 1].Equals(values[i]);
                int cmp = values[i - 1].CompareTo(values[i]);
                Assert.True(!eq.IsNA);
                Assert.True(!ne.IsNA);
                Assert.True(eq.IsTrue == ne.IsFalse);
                Assert.True(feq == eq.IsTrue);
                Assert.True(cmp <= 0);
                Assert.True(feq == (cmp == 0));
            }
        }
    }
}
