// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Xunit;
namespace Microsoft.ML.RunTests
{
    public sealed class ColumnTypeTests
    {
        [Fact]
        public void TestEqualAndGetHashCode()
        {
            var dict = new Dictionary<DataViewType, string>();
            // add PrimitiveTypes, KeyType & corresponding VectorTypes
            VectorType tmp1, tmp2;
            var types = new PrimitiveDataViewType[] { NumberDataViewType.SByte, NumberDataViewType.Int16, NumberDataViewType.Int32, NumberDataViewType.Int64,
                NumberDataViewType.Byte, NumberDataViewType.UInt16, NumberDataViewType.UInt32, NumberDataViewType.UInt64, RowIdDataViewType.Instance,
                TextDataViewType.Instance, BooleanDataViewType.Instance, DateTimeDataViewType.Instance, DateTimeOffsetDataViewType.Instance, TimeSpanDataViewType.Instance };

            foreach (var type in types)
            {
                var tmp = type;
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

                // KeyType & Vector
                var rawType = tmp.RawType;
                if (!KeyType.IsValidDataType(rawType))
                    continue;
                for (ulong min = 0; min < 5; min++)
                {
                    for (var count = 1; count < 5; count++)
                    {
                        tmp = new KeyType(rawType, count);
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
                    Assert.True(rawType.TryGetDataKind(out var kind));
                    tmp = new KeyType(rawType, kind.ToMaxInt());
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
                    if (dict.ContainsKey(tmp4))
                        Assert.True(false, dict[tmp4] + " and " + tmp4.ToString() + " are duplicates.");
                    dict[tmp4] = tmp4.ToString();
                }
        }
    }
}
