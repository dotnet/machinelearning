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
            var dict = new Dictionary<ColumnType, string>();
            // add PrimitiveTypes, KeyType & corresponding VectorTypes
            PrimitiveType tmp;
            VectorType tmp1, tmp2;
            foreach (var kind in (DataKind[])Enum.GetValues(typeof(DataKind)))
            {
                tmp = PrimitiveType.FromKind(kind);
                if(dict.ContainsKey(tmp) && dict[tmp] != tmp.ToString())
                    Assert.True(false, dict[tmp] + " and " + tmp.ToString() + " are duplicates.");
                dict[tmp] = tmp.ToString();
                for (int size = 0; size < 5; size++)
                {
                    tmp1 = new VectorType(tmp, size);
                    if (dict.ContainsKey(tmp1) && dict[tmp1] != tmp1.ToString())
                        Assert.True(false, dict[tmp1] + " and " + tmp1.ToString() + " are duplicates.");
                    dict[tmp1] = tmp1.ToString();
                    for (int size1 = 0; size1 < 5; size1++)
                    {
                        tmp2 = new VectorType(tmp, size, size1);
                        if (dict.ContainsKey(tmp2) && dict[tmp2] != tmp2.ToString())
                            Assert.True(false, dict[tmp2] + " and " + tmp2.ToString() + " are duplicates.");
                        dict[tmp2] = tmp2.ToString();
                    }
                }

                // KeyType & Vector
                if (!KeyType.IsValidDataKind(kind))
                    continue;
                for (ulong min = 0; min < 5; min++)
                {
                    for (var count = 0; count < 5; count++)
                    {
                        tmp = new KeyType(kind, min, count);
                        if (dict.ContainsKey(tmp) && dict[tmp] != tmp.ToString())
                            Assert.True(false, dict[tmp] + " and " + tmp.ToString() + " are duplicates.");
                        dict[tmp] = tmp.ToString();
                        for (int size = 0; size < 5; size++)
                        {
                            tmp1 = new VectorType(tmp, size);
                            if (dict.ContainsKey(tmp1) && dict[tmp1] != tmp1.ToString())
                                Assert.True(false, dict[tmp1] + " and " + tmp1.ToString() + " are duplicates.");
                            dict[tmp1] = tmp1.ToString();
                            for (int size1 = 0; size1 < 5; size1++)
                            {
                                tmp2 = new VectorType(tmp, size, size1);
                                if (dict.ContainsKey(tmp2) && dict[tmp2] != tmp2.ToString())
                                    Assert.True(false, dict[tmp2] + " and " + tmp2.ToString() + " are duplicates.");
                                dict[tmp2] = tmp2.ToString();
                            }
                        }
                    }
                    tmp = new KeyType(kind, min, 0, false);
                    if (dict.ContainsKey(tmp) && dict[tmp] != tmp.ToString())
                        Assert.True(false, dict[tmp] + " and " + tmp.ToString() + " are duplicates.");
                    dict[tmp] = tmp.ToString();
                    for (int size = 0; size < 5; size++)
                    {
                        tmp1 = new VectorType(tmp, size);
                        if (dict.ContainsKey(tmp1) && dict[tmp1] != tmp1.ToString())
                            Assert.True(false, dict[tmp1] + " and " + tmp1.ToString() + " are duplicates.");
                        dict[tmp1] = tmp1.ToString();
                        for (int size1 = 0; size1 < 5; size1++)
                        {
                            tmp2 = new VectorType(tmp, size, size1);
                            if (dict.ContainsKey(tmp2) && dict[tmp2] != tmp2.ToString())
                                Assert.True(false, dict[tmp2] + " and " + tmp2.ToString() + " are duplicates.");
                            dict[tmp2] = tmp2.ToString();
                        }
                    }
                }
            }

            // add ImageTypes
            for (int height = 1; height < 5; height++)
                for (int width = 1; width < 5; width++)
                {
                    var tmp4 = new ImageType(height, width);
                    if (dict.ContainsKey(tmp4) && dict[tmp4] != tmp4.ToString())
                        Assert.True(false, dict[tmp4] + " and " + tmp4.ToString() + " are duplicates.");
                    dict[tmp4] = tmp4.ToString();
                }
        }
    }
}
