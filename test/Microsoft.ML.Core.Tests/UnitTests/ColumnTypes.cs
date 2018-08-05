// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Xunit;
namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class ColumnTypeTests
    {
        [Fact]
        public void TestEqualAndGetHashCode()
        {
            var hashSet = new HashSet<ColumnType>();
            // add PrimitiveTypes, KeyType & corresponding VectorTypes
            PrimitiveType tmp;
            VectorType tmp1, tmp2;
            foreach (var kind in (DataKind[])Enum.GetValues(typeof(DataKind)))
            {
                tmp = PrimitiveType.FromKind(kind);
                Assert.True(hashSet.Add(tmp), tmp.ToString() + " should not be present.");
                for (int size = 0; size < 5; size++)
                {
                    tmp1 = new VectorType(tmp, size);
                    Assert.True(hashSet.Add(tmp1), tmp1.ToString() + " should not be present.");
                    for (int size1 = 0; size1 < 5; size1++)
                    {
                        tmp2 = new VectorType(tmp, size, size1);
                        Assert.True(hashSet.Add(tmp2), tmp2.ToString() + " should not be present.");
                    }
                }

                for (ulong min = 0; min < 5; min++)
                    for (var count = 0; count < 5; count++)
                    {
                        tmp = new KeyType(kind, min, count);
                        Assert.True(hashSet.Add(tmp), tmp.ToString() + " should not be present.");
                        for (int size = 0; size < 5; size++)
                        {
                            tmp1 = new VectorType(tmp, size);
                            Assert.True(hashSet.Add(tmp1), tmp1.ToString() + " should not be present.");
                            for (int size1 = 0; size1 < 5; size1++)
                            {
                                tmp2 = new VectorType(tmp, size, size1);
                                Assert.True(hashSet.Add(tmp2), tmp2.ToString() + " should not be present.");
                            }
                        }
                        tmp = new KeyType(kind, min, count, false);
                        Assert.True(hashSet.Add(tmp), tmp.ToString() + " should not be present.");
                        for (int size = 0; size < 5; size++)
                        {
                            tmp1 = new VectorType(tmp, size);
                            Assert.True(hashSet.Add(tmp1), tmp1.ToString() + " should not be present.");
                            for (int size1 = 0; size1 < 5; size1++)
                            {
                                tmp2 = new VectorType(tmp, size, size1);
                                Assert.True(hashSet.Add(tmp2), tmp2.ToString() + " should not be present.");
                            }
                        }
                    }
            }

            // add ImageTypes
            for (int height = 0; height < 5; height++)
                for (int width = 0; width < 5; width++)
                {
                    var tmp4 = new ImageType(height, width);
                    Assert.True(hashSet.Add(tmp4), tmp4.ToString() + " should not be present.");
                }
        }
    }
}
