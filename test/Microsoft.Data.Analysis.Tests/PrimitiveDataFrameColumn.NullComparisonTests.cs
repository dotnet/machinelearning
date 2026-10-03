// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumn.NullComparisonTests.tt. Do not modify directly

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class PrimitiveDataFrameColumnTests
    {
        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, null, null, false, false, false, true, true, true });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { null, false, true, null, false, true, null, false, true });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, null, null, false, false, false, true, true, true }, false);
            var right = CreateColumnWithValuesUnderNulls("Right", new bool?[] { null, false, true, null, false, true, null, false, true }, true);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null, false, true, null, false, true });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { false, false, false, true, true, true, false, true, false });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { false, false, false, true, true, true, false, true, false });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { null, false, true, null, false, true, null, false, true });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { false, false, true, true, false, true, false, true, null, false, true, null, null, false, true, false, true, null, null, false });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { false, true, false, true, false, false, true, true, false, null, true, null, false, false, true, null, null, true, null, false });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { false, false, true, true, false, true, false, true, null, false, true, null, null, false, true, false, true, null, null, false }, false);
            var right = CreateColumnWithValuesUnderNulls("Right", new bool?[] { false, true, false, true, false, false, true, true, false, null, true, null, false, false, true, null, null, true, null, false }, true);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null });

            var results = left.ElementwiseEquals(false);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null });

            var results = left.ElementwiseEquals(true);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, false, true, null }, false);

            var results = left.ElementwiseEquals(false);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, false, true, null }, true);

            var results = left.ElementwiseEquals(true);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, null, null, false, false, false, true, true, true });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { null, false, true, null, false, true, null, false, true });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, null, null, false, false, false, true, true, true }, false);
            var right = CreateColumnWithValuesUnderNulls("Right", new bool?[] { null, false, true, null, false, true, null, false, true }, true);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null, false, true, null, false, true });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { false, false, false, true, true, true, false, true, false });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { false, false, false, true, true, true, false, true, false });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { null, false, true, null, false, true, null, false, true });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { false, false, true, true, false, true, false, true, null, false, true, null, null, false, true, false, true, null, null, false });
            var right = new PrimitiveDataFrameColumn<bool>("Right", new bool?[] { false, true, false, true, false, false, true, true, false, null, true, null, false, false, true, null, null, true, null, false });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { false, false, true, true, false, true, false, true, null, false, true, null, null, false, true, false, true, null, null, false }, false);
            var right = CreateColumnWithValuesUnderNulls("Right", new bool?[] { false, true, false, true, false, false, true, true, false, null, true, null, false, false, true, null, null, true, null, false }, true);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null });

            var results = left.ElementwiseNotEquals(false);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<bool>("Left", new bool?[] { null, false, true, null });

            var results = left.ElementwiseNotEquals(true);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, false, true, null }, false);

            var results = left.ElementwiseNotEquals(false);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Boolean_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new bool?[] { null, false, true, null }, true);

            var results = left.ElementwiseNotEquals(true);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseEquals((byte)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseEquals((byte)200);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseEquals((byte)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseEquals((byte)200);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseNotEquals((byte)3);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseNotEquals((byte)200);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseNotEquals((byte)3);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseNotEquals((byte)200);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseLessThan((byte)3);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseLessThan((byte)200);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseLessThan((byte)3);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseLessThan((byte)200);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseLessThanOrEqual((byte)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseLessThanOrEqual((byte)200);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseLessThanOrEqual((byte)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseLessThanOrEqual((byte)200);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseGreaterThan((byte)3);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseGreaterThan((byte)200);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseGreaterThan((byte)3);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseGreaterThan((byte)200);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 }, (byte)200);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { null, (byte)3, (byte)200, null, (byte)3, (byte)200, null, (byte)3, (byte)200 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 });
            var right = new PrimitiveDataFrameColumn<byte>("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, (byte)200, (byte)3, (byte)200, null, (byte)3, (byte)200, null, null, (byte)3, (byte)200, (byte)3, (byte)200, null, null, (byte)3 }, (byte)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new byte?[] { (byte)3, (byte)200, (byte)3, (byte)200, (byte)3, (byte)3, (byte)200, (byte)200, (byte)3, null, (byte)200, null, (byte)3, (byte)3, (byte)200, null, null, (byte)200, null, (byte)3 }, (byte)200);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseGreaterThanOrEqual((byte)3);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, (byte)3, (byte)200, null });

            var results = left.ElementwiseGreaterThanOrEqual((byte)200);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)3);

            var results = left.ElementwiseGreaterThanOrEqual((byte)3);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new byte?[] { null, (byte)3, (byte)200, null }, (byte)200);

            var results = left.ElementwiseGreaterThanOrEqual((byte)200);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Byte_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<byte>("Left", new byte?[] { null, null, null, (byte)3, (byte)3, (byte)3, (byte)200, (byte)200, (byte)200 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 200, null, 3, 200, null, 3, 200 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseEquals((sbyte)-5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseEquals((sbyte)7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseEquals((sbyte)-5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseEquals((sbyte)7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseNotEquals((sbyte)-5);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseNotEquals((sbyte)7);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseNotEquals((sbyte)-5);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseNotEquals((sbyte)7);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseLessThan((sbyte)-5);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseLessThan((sbyte)7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseLessThan((sbyte)-5);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseLessThan((sbyte)7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseLessThanOrEqual((sbyte)-5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseLessThanOrEqual((sbyte)7);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseLessThanOrEqual((sbyte)-5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseLessThanOrEqual((sbyte)7);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseGreaterThan((sbyte)-5);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseGreaterThan((sbyte)7);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseGreaterThan((sbyte)-5);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseGreaterThan((sbyte)7);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 }, (sbyte)7);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 });
            var right = new PrimitiveDataFrameColumn<sbyte>("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, null, null, (sbyte)-5 }, (sbyte)-5);
            var right = CreateColumnWithValuesUnderNulls("Right", new sbyte?[] { (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)7, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)-5, null, (sbyte)7, null, (sbyte)-5, (sbyte)-5, (sbyte)7, null, null, (sbyte)7, null, (sbyte)-5 }, (sbyte)7);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseGreaterThanOrEqual((sbyte)-5);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null });

            var results = left.ElementwiseGreaterThanOrEqual((sbyte)7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)-5);

            var results = left.ElementwiseGreaterThanOrEqual((sbyte)-5);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new sbyte?[] { null, (sbyte)-5, (sbyte)7, null }, (sbyte)7);

            var results = left.ElementwiseGreaterThanOrEqual((sbyte)7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_SByte_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<sbyte>("Left", new sbyte?[] { null, null, null, (sbyte)-5, (sbyte)-5, (sbyte)-5, (sbyte)7, (sbyte)7, (sbyte)7 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -5, 7, null, -5, 7, null, -5, 7 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseEquals((short)-300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseEquals((short)300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseEquals((short)-300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseEquals((short)300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseNotEquals((short)-300);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseNotEquals((short)300);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseNotEquals((short)-300);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseNotEquals((short)300);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseLessThan((short)-300);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseLessThan((short)300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseLessThan((short)-300);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseLessThan((short)300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseLessThanOrEqual((short)-300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseLessThanOrEqual((short)300);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseLessThanOrEqual((short)-300);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseLessThanOrEqual((short)300);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseGreaterThan((short)-300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseGreaterThan((short)300);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseGreaterThan((short)-300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseGreaterThan((short)300);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 }, (short)300);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300, (short)-300, (short)300, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { null, (short)-300, (short)300, null, (short)-300, (short)300, null, (short)-300, (short)300 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 });
            var right = new PrimitiveDataFrameColumn<short>("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { (short)-300, (short)-300, (short)300, (short)300, (short)-300, (short)300, (short)-300, (short)300, null, (short)-300, (short)300, null, null, (short)-300, (short)300, (short)-300, (short)300, null, null, (short)-300 }, (short)-300);
            var right = CreateColumnWithValuesUnderNulls("Right", new short?[] { (short)-300, (short)300, (short)-300, (short)300, (short)-300, (short)-300, (short)300, (short)300, (short)-300, null, (short)300, null, (short)-300, (short)-300, (short)300, null, null, (short)300, null, (short)-300 }, (short)300);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseGreaterThanOrEqual((short)-300);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, (short)-300, (short)300, null });

            var results = left.ElementwiseGreaterThanOrEqual((short)300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)-300);

            var results = left.ElementwiseGreaterThanOrEqual((short)-300);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new short?[] { null, (short)-300, (short)300, null }, (short)300);

            var results = left.ElementwiseGreaterThanOrEqual((short)300);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int16_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<short>("Left", new short?[] { null, null, null, (short)-300, (short)-300, (short)-300, (short)300, (short)300, (short)300 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -300, 300, null, -300, 300, null, -300, 300 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseEquals((ushort)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseEquals((ushort)60000);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseEquals((ushort)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseEquals((ushort)60000);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseNotEquals((ushort)3);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseNotEquals((ushort)60000);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseNotEquals((ushort)3);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseNotEquals((ushort)60000);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseLessThan((ushort)3);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseLessThan((ushort)60000);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseLessThan((ushort)3);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseLessThan((ushort)60000);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseLessThanOrEqual((ushort)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseLessThanOrEqual((ushort)60000);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseLessThanOrEqual((ushort)3);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseLessThanOrEqual((ushort)60000);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseGreaterThan((ushort)3);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseGreaterThan((ushort)60000);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseGreaterThan((ushort)3);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseGreaterThan((ushort)60000);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 }, (ushort)60000);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 });
            var right = new PrimitiveDataFrameColumn<ushort>("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, (ushort)3, (ushort)60000, null, null, (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, null, null, (ushort)3 }, (ushort)3);
            var right = CreateColumnWithValuesUnderNulls("Right", new ushort?[] { (ushort)3, (ushort)60000, (ushort)3, (ushort)60000, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)3, null, (ushort)60000, null, (ushort)3, (ushort)3, (ushort)60000, null, null, (ushort)60000, null, (ushort)3 }, (ushort)60000);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseGreaterThanOrEqual((ushort)3);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null });

            var results = left.ElementwiseGreaterThanOrEqual((ushort)60000);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)3);

            var results = left.ElementwiseGreaterThanOrEqual((ushort)3);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ushort?[] { null, (ushort)3, (ushort)60000, null }, (ushort)60000);

            var results = left.ElementwiseGreaterThanOrEqual((ushort)60000);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt16_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ushort>("Left", new ushort?[] { null, null, null, (ushort)3, (ushort)3, (ushort)3, (ushort)60000, (ushort)60000, (ushort)60000 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 60000, null, 3, 60000, null, 3, 60000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseEquals(-7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseEquals(42);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseEquals(-7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseEquals(42);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseNotEquals(-7);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseNotEquals(42);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseNotEquals(-7);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseNotEquals(42);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseLessThan(-7);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseLessThan(42);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseLessThan(-7);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseLessThan(42);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseLessThanOrEqual(-7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseLessThanOrEqual(42);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseLessThanOrEqual(-7);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseLessThanOrEqual(42);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseGreaterThan(-7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseGreaterThan(42);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseGreaterThan(-7);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseGreaterThan(42);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 }, 42);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, -7, 42, 42, 42, -7, 42, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 });
            var right = new PrimitiveDataFrameColumn<int>("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { -7, -7, 42, 42, -7, 42, -7, 42, null, -7, 42, null, null, -7, 42, -7, 42, null, null, -7 }, -7);
            var right = CreateColumnWithValuesUnderNulls("Right", new int?[] { -7, 42, -7, 42, -7, -7, 42, 42, -7, null, 42, null, -7, -7, 42, null, null, 42, null, -7 }, 42);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseGreaterThanOrEqual(-7);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, -7, 42, null });

            var results = left.ElementwiseGreaterThanOrEqual(42);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, -7);

            var results = left.ElementwiseGreaterThanOrEqual(-7);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new int?[] { null, -7, 42, null }, 42);

            var results = left.ElementwiseGreaterThanOrEqual(42);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int32_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<int>("Left", new int?[] { null, null, null, -7, -7, -7, 42, 42, 42 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 42, null, -7, 42, null, -7, 42 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseEquals(3u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseEquals(4000000000u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseEquals(3u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseEquals(4000000000u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseNotEquals(3u);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseNotEquals(4000000000u);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseNotEquals(3u);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseNotEquals(4000000000u);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseLessThan(3u);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseLessThan(4000000000u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseLessThan(3u);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseLessThan(4000000000u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseLessThanOrEqual(3u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseLessThanOrEqual(4000000000u);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseLessThanOrEqual(3u);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseLessThanOrEqual(4000000000u);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseGreaterThan(3u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseGreaterThan(4000000000u);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseGreaterThan(3u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseGreaterThan(4000000000u);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u }, 4000000000u);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { null, 3u, 4000000000u, null, 3u, 4000000000u, null, 3u, 4000000000u });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u });
            var right = new PrimitiveDataFrameColumn<uint>("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { 3u, 3u, 4000000000u, 4000000000u, 3u, 4000000000u, 3u, 4000000000u, null, 3u, 4000000000u, null, null, 3u, 4000000000u, 3u, 4000000000u, null, null, 3u }, 3u);
            var right = CreateColumnWithValuesUnderNulls("Right", new uint?[] { 3u, 4000000000u, 3u, 4000000000u, 3u, 3u, 4000000000u, 4000000000u, 3u, null, 4000000000u, null, 3u, 3u, 4000000000u, null, null, 4000000000u, null, 3u }, 4000000000u);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseGreaterThanOrEqual(3u);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, 3u, 4000000000u, null });

            var results = left.ElementwiseGreaterThanOrEqual(4000000000u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 3u);

            var results = left.ElementwiseGreaterThanOrEqual(3u);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new uint?[] { null, 3u, 4000000000u, null }, 4000000000u);

            var results = left.ElementwiseGreaterThanOrEqual(4000000000u);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt32_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<uint>("Left", new uint?[] { null, null, null, 3u, 3u, 3u, 4000000000u, 4000000000u, 4000000000u });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 4000000000, null, 3, 4000000000, null, 3, 4000000000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseEquals(-7L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseEquals(5000000000L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseEquals(-7L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseEquals(5000000000L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseNotEquals(-7L);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseNotEquals(5000000000L);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseNotEquals(-7L);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseNotEquals(5000000000L);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseLessThan(-7L);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseLessThan(5000000000L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseLessThan(-7L);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseLessThan(5000000000L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseLessThanOrEqual(-7L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseLessThanOrEqual(5000000000L);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseLessThanOrEqual(-7L);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseLessThanOrEqual(5000000000L);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseGreaterThan(-7L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseGreaterThan(5000000000L);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseGreaterThan(-7L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseGreaterThan(5000000000L);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L }, 5000000000L);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { null, -7L, 5000000000L, null, -7L, 5000000000L, null, -7L, 5000000000L });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L });
            var right = new PrimitiveDataFrameColumn<long>("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { -7L, -7L, 5000000000L, 5000000000L, -7L, 5000000000L, -7L, 5000000000L, null, -7L, 5000000000L, null, null, -7L, 5000000000L, -7L, 5000000000L, null, null, -7L }, -7L);
            var right = CreateColumnWithValuesUnderNulls("Right", new long?[] { -7L, 5000000000L, -7L, 5000000000L, -7L, -7L, 5000000000L, 5000000000L, -7L, null, 5000000000L, null, -7L, -7L, 5000000000L, null, null, 5000000000L, null, -7L }, 5000000000L);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseGreaterThanOrEqual(-7L);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, -7L, 5000000000L, null });

            var results = left.ElementwiseGreaterThanOrEqual(5000000000L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, -7L);

            var results = left.ElementwiseGreaterThanOrEqual(-7L);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new long?[] { null, -7L, 5000000000L, null }, 5000000000L);

            var results = left.ElementwiseGreaterThanOrEqual(5000000000L);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Int64_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<long>("Left", new long?[] { null, null, null, -7L, -7L, -7L, 5000000000L, 5000000000L, 5000000000L });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -7, 5000000000, null, -7, 5000000000, null, -7, 5000000000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseEquals(3UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseEquals(5000000000UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseEquals(3UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseEquals(5000000000UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseNotEquals(3UL);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseNotEquals(5000000000UL);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseNotEquals(3UL);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseNotEquals(5000000000UL);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseLessThan(3UL);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseLessThan(5000000000UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseLessThan(3UL);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseLessThan(5000000000UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseLessThanOrEqual(3UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseLessThanOrEqual(5000000000UL);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseLessThanOrEqual(3UL);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseLessThanOrEqual(5000000000UL);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseGreaterThan(3UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseGreaterThan(5000000000UL);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseGreaterThan(3UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseGreaterThan(5000000000UL);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL }, 5000000000UL);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { null, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, 3UL, 5000000000UL });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL });
            var right = new PrimitiveDataFrameColumn<ulong>("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 5000000000UL, null, 3UL, 5000000000UL, null, null, 3UL, 5000000000UL, 3UL, 5000000000UL, null, null, 3UL }, 3UL);
            var right = CreateColumnWithValuesUnderNulls("Right", new ulong?[] { 3UL, 5000000000UL, 3UL, 5000000000UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 3UL, null, 5000000000UL, null, 3UL, 3UL, 5000000000UL, null, null, 5000000000UL, null, 3UL }, 5000000000UL);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseGreaterThanOrEqual(3UL);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, 3UL, 5000000000UL, null });

            var results = left.ElementwiseGreaterThanOrEqual(5000000000UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 3UL);

            var results = left.ElementwiseGreaterThanOrEqual(3UL);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new ulong?[] { null, 3UL, 5000000000UL, null }, 5000000000UL);

            var results = left.ElementwiseGreaterThanOrEqual(5000000000UL);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_UInt64_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<ulong>("Left", new ulong?[] { null, null, null, 3UL, 3UL, 3UL, 5000000000UL, 5000000000UL, 5000000000UL });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, 3, 5000000000, null, 3, 5000000000, null, 3, 5000000000 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseEquals(-1.5f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseEquals(2.25f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseEquals(-1.5f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseEquals(2.25f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseNotEquals(-1.5f);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseNotEquals(2.25f);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseNotEquals(-1.5f);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseNotEquals(2.25f);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseLessThan(-1.5f);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseLessThan(2.25f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseLessThan(-1.5f);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseLessThan(2.25f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseLessThanOrEqual(-1.5f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseLessThanOrEqual(2.25f);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseLessThanOrEqual(-1.5f);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseLessThanOrEqual(2.25f);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseGreaterThan(-1.5f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseGreaterThan(2.25f);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseGreaterThan(-1.5f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseGreaterThan(2.25f);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f }, 2.25f);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { null, -1.5f, 2.25f, null, -1.5f, 2.25f, null, -1.5f, 2.25f });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f });
            var right = new PrimitiveDataFrameColumn<float>("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, 2.25f, -1.5f, 2.25f, null, -1.5f, 2.25f, null, null, -1.5f, 2.25f, -1.5f, 2.25f, null, null, -1.5f }, -1.5f);
            var right = CreateColumnWithValuesUnderNulls("Right", new float?[] { -1.5f, 2.25f, -1.5f, 2.25f, -1.5f, -1.5f, 2.25f, 2.25f, -1.5f, null, 2.25f, null, -1.5f, -1.5f, 2.25f, null, null, 2.25f, null, -1.5f }, 2.25f);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseGreaterThanOrEqual(-1.5f);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, -1.5f, 2.25f, null });

            var results = left.ElementwiseGreaterThanOrEqual(2.25f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, -1.5f);

            var results = left.ElementwiseGreaterThanOrEqual(-1.5f);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new float?[] { null, -1.5f, 2.25f, null }, 2.25f);

            var results = left.ElementwiseGreaterThanOrEqual(2.25f);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Single_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<float>("Left", new float?[] { null, null, null, -1.5f, -1.5f, -1.5f, 2.25f, 2.25f, 2.25f });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseEquals(-1.5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseEquals(2.25);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseEquals(-1.5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseEquals(2.25);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseNotEquals(-1.5);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseNotEquals(2.25);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseNotEquals(-1.5);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseNotEquals(2.25);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseLessThan(-1.5);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseLessThan(2.25);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseLessThan(-1.5);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseLessThan(2.25);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseLessThanOrEqual(-1.5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseLessThanOrEqual(2.25);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseLessThanOrEqual(-1.5);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseLessThanOrEqual(2.25);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseGreaterThan(-1.5);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseGreaterThan(2.25);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseGreaterThan(-1.5);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseGreaterThan(2.25);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, null, null, -1.5, -1.5, -1.5, 2.25, 2.25, 2.25 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 }, 2.25);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, -1.5, 2.25, 2.25, 2.25, -1.5, 2.25, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { -1.5, -1.5, 2.25, 2.25, -1.5, 2.25, -1.5, 2.25, null, -1.5, 2.25, null, null, -1.5, 2.25, -1.5, 2.25, null, null, -1.5 }, -1.5);
            var right = CreateColumnWithValuesUnderNulls("Right", new double?[] { -1.5, 2.25, -1.5, 2.25, -1.5, -1.5, 2.25, 2.25, -1.5, null, 2.25, null, -1.5, -1.5, 2.25, null, null, 2.25, null, -1.5 }, 2.25);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseGreaterThanOrEqual(-1.5);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<double>("Left", new double?[] { null, -1.5, 2.25, null });

            var results = left.ElementwiseGreaterThanOrEqual(2.25);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, -1.5);

            var results = left.ElementwiseGreaterThanOrEqual(-1.5);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Double_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new double?[] { null, -1.5, 2.25, null }, 2.25);

            var results = left.ElementwiseGreaterThanOrEqual(2.25);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseEquals(-1.5m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseEquals(2.25m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseEquals(-1.5m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseEquals(2.25m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseNotEquals(-1.5m);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseNotEquals(2.25m);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseNotEquals(-1.5m);

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseNotEquals(2.25m);

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseNotEquals_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseLessThan(-1.5m);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseLessThan(2.25m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseLessThan(-1.5m);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseLessThan(2.25m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseLessThanOrEqual(-1.5m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseLessThanOrEqual(2.25m);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseLessThanOrEqual(-1.5m);

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseLessThanOrEqual(2.25m);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseLessThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseGreaterThan(-1.5m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseGreaterThan(2.25m);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseGreaterThan(-1.5m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseGreaterThan(2.25m);

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThan_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m }, 2.25m);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { null, -1.5m, 2.25m, null, -1.5m, 2.25m, null, -1.5m, 2.25m });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m });
            var right = new PrimitiveDataFrameColumn<decimal>("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, 2.25m, -1.5m, 2.25m, null, -1.5m, 2.25m, null, null, -1.5m, 2.25m, -1.5m, 2.25m, null, null, -1.5m }, -1.5m);
            var right = CreateColumnWithValuesUnderNulls("Right", new decimal?[] { -1.5m, 2.25m, -1.5m, 2.25m, -1.5m, -1.5m, 2.25m, 2.25m, -1.5m, null, 2.25m, null, -1.5m, -1.5m, 2.25m, null, null, 2.25m, null, -1.5m }, 2.25m);

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseGreaterThanOrEqual(-1.5m);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, -1.5m, 2.25m, null });

            var results = left.ElementwiseGreaterThanOrEqual(2.25m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, -1.5m);

            var results = left.ElementwiseGreaterThanOrEqual(-1.5m);

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new decimal?[] { null, -1.5m, 2.25m, null }, 2.25m);

            var results = left.ElementwiseGreaterThanOrEqual(2.25m);

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Decimal_ElementwiseGreaterThanOrEqual_AgainstDoubleColumn_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<decimal>("Left", new decimal?[] { null, null, null, -1.5m, -1.5m, -1.5m, 2.25m, 2.25m, 2.25m });
            var right = new PrimitiveDataFrameColumn<double>("Right", new double?[] { null, -1.5, 2.25, null, -1.5, 2.25, null, -1.5, 2.25 });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseEquals('a');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseEquals('z');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseEquals('a');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseEquals('z');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseNotEquals('a');

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseNotEquals('z');

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseNotEquals('a');

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseNotEquals('z');

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseLessThan('a');

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseLessThan('z');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseLessThan('a');

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseLessThan('z');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseLessThanOrEqual('a');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseLessThanOrEqual('z');

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseLessThanOrEqual('a');

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseLessThanOrEqual('z');

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseGreaterThan('a');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseGreaterThan('z');

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseGreaterThan('a');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseGreaterThan('z');

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, null, null, 'a', 'a', 'a', 'z', 'z', 'z' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' }, 'z');

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'a', 'z', 'z', 'z', 'a', 'z', 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { null, 'a', 'z', null, 'a', 'z', null, 'a', 'z' });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' });
            var right = new PrimitiveDataFrameColumn<char>("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { 'a', 'a', 'z', 'z', 'a', 'z', 'a', 'z', null, 'a', 'z', null, null, 'a', 'z', 'a', 'z', null, null, 'a' }, 'a');
            var right = CreateColumnWithValuesUnderNulls("Right", new char?[] { 'a', 'z', 'a', 'z', 'a', 'a', 'z', 'z', 'a', null, 'z', null, 'a', 'a', 'z', null, null, 'z', null, 'a' }, 'z');

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseGreaterThanOrEqual('a');

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<char>("Left", new char?[] { null, 'a', 'z', null });

            var results = left.ElementwiseGreaterThanOrEqual('z');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'a');

            var results = left.ElementwiseGreaterThanOrEqual('a');

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_Char_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new char?[] { null, 'a', 'z', null }, 'z');

            var results = left.ElementwiseGreaterThanOrEqual('z');

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { false, true, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseEquals(right);

            AssertComparisonResults(new[] { true, false, false, true, true, false, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseEquals(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseEquals(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseEquals(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseEquals(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, true, false, true, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { true, false, true, true, true, false, true, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseNotEquals(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, true, false, true, true, false, false, true, false, false, true, true, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseNotEquals(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseNotEquals(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseNotEquals(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { true, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseNotEquals_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseNotEquals(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { true, true, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, true, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThan(right);

            AssertComparisonResults(new[] { false, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseLessThan(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseLessThan(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseLessThan(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThan(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThanOrEqual(right);

            AssertComparisonResults(new[] { true, true, false, true, true, false, true, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseLessThanOrEqual(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseLessThanOrEqual(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseLessThanOrEqual(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseLessThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseLessThanOrEqual(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, false, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, false, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, false, false, true, false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThan(right);

            AssertComparisonResults(new[] { false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseGreaterThan(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseGreaterThan(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseGreaterThan(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThan_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThan(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, false, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_EveryNullCombination()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_EveryNullCombination_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, null, null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, false, false, true, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_NullsOnlyOnLeft()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, true, false, false, true, false, false, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_NullsOnlyOnRight()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30) });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { false, true, false, false, true, true, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) });
            var right = new PrimitiveDataFrameColumn<DateTime>("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) });

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AcrossValidityBitmapBytes_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2020, 1, 1) }, new DateTime(2020, 1, 1));
            var right = CreateColumnWithValuesUnderNulls("Right", new DateTime?[] { new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), new DateTime(2021, 6, 30), new DateTime(2020, 1, 1), null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1), new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null, null, new DateTime(2021, 6, 30), null, new DateTime(2020, 1, 1) }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThanOrEqual(right);

            AssertComparisonResults(new[] { true, false, true, true, true, true, false, true, false, false, true, true, false, true, true, false, false, false, true, true }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AgainstLowScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseGreaterThanOrEqual(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AgainstHighScalar()
        {
            var left = new PrimitiveDataFrameColumn<DateTime>("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null });

            var results = left.ElementwiseGreaterThanOrEqual(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AgainstLowScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2020, 1, 1));

            var results = left.ElementwiseGreaterThanOrEqual(new DateTime(2020, 1, 1));

            AssertComparisonResults(new[] { false, true, true, false }, results);
        }

        [Fact]
        public void PrimitiveDataFrameColumn_DateTime_ElementwiseGreaterThanOrEqual_AgainstHighScalar_WithValuesUnderNulls()
        {
            var left = CreateColumnWithValuesUnderNulls("Left", new DateTime?[] { null, new DateTime(2020, 1, 1), new DateTime(2021, 6, 30), null }, new DateTime(2021, 6, 30));

            var results = left.ElementwiseGreaterThanOrEqual(new DateTime(2021, 6, 30));

            AssertComparisonResults(new[] { false, false, true, false }, results);
        }

        private static PrimitiveDataFrameColumn<T> CreateColumnWithValuesUnderNulls<T>(string name, T?[] values, T valueUnderNulls)
            where T : unmanaged
        {
            // Build the column from Arrow formatted buffers, so that null elements are backed by a non default value
            var data = values.Select(v => v ?? valueUnderNulls).ToArray();
            var nullBitMap = new byte[(values.Length + 7) / 8];
            var nullCount = 0;
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i].HasValue)
                    nullBitMap[i / 8] |= (byte)(1 << (i % 8));
                else
                    nullCount++;
            }

            var column = new PrimitiveDataFrameColumn<T>(name, MemoryMarshal.AsBytes(data.AsSpan()).ToArray(), nullBitMap, values.Length, nullCount);

            Assert.Equal(values, column.ToArray());
            Assert.Equal(valueUnderNulls, column.GetReadOnlyDataBuffers().Single().Span[Array.IndexOf(values, null)]);
            return column;
        }

        private static void AssertComparisonResults(bool[] expected, PrimitiveDataFrameColumn<bool> results)
        {
            Assert.Equal(0, results.NullCount);
            Assert.Equal(expected, results.Select(r => r.Value).ToArray());
        }
    }
}
