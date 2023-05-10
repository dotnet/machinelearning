// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Microsoft.Data.Analysis;

namespace Microsoft.ML.Fairlearn.Tests
{
    public class UtilityTest
    {
        [Fact]
        public void DemographyParityTest()
        {
            var dp = new UtilityParity(differenceBound: 0.01F);

            string[] str = { "a", "b", "a", "a", "b", "a", "b", "b", "a", "b" };
            StringDataFrameColumn sensitiveFeature = new StringDataFrameColumn("group_id", str);

            int[] vs = { 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 };
            PrimitiveDataFrameColumn<int> y = new PrimitiveDataFrameColumn<int>("label", vs);


            DataFrame x = new DataFrame();
            dp.LoadData(x, y, sensitiveFeature: sensitiveFeature);

            float[] fl = { 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F };
            PrimitiveDataFrameColumn<float> ypred = new PrimitiveDataFrameColumn<float>("pred", fl);
            var gSinged = dp.Gamma(ypred);

            Assert.Equal(0.1, Convert.ToSingle(gSinged["value"][0]), 1);
            Assert.Equal(-0.1, Convert.ToSingle(gSinged["value"][1]), 1);
            Assert.Equal(-0.1, Convert.ToSingle(gSinged["value"][2]), 1);
            Assert.Equal(0.1, Convert.ToSingle(gSinged["value"][3]), 1);
        }
    }
}
