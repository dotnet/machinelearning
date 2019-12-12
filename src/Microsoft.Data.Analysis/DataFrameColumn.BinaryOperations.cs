
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.BinaryOperations.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public abstract partial class DataFrameColumn
    {
        public virtual DataFrameColumn Add(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise addition on each value in the column
        /// </summary>
        public virtual DataFrameColumn Add<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise addition on each value in the column
        /// </summary>
        public virtual DataFrameColumn ReverseAdd<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Subtract(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise subtraction on each value in the column
        /// </summary>
        public virtual DataFrameColumn Subtract<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise subtraction on each value in the column
        /// </summary>
        public virtual DataFrameColumn ReverseSubtract<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Multiply(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise multiplication on each value in the column
        /// </summary>
        public virtual DataFrameColumn Multiply<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise multiplication on each value in the column
        /// </summary>
        public virtual DataFrameColumn ReverseMultiply<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Divide(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise division on each value in the column
        /// </summary>
        public virtual DataFrameColumn Divide<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise division on each value in the column
        /// </summary>
        public virtual DataFrameColumn ReverseDivide<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Modulo(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise modulus operation on each value in the column
        /// </summary>
        public virtual DataFrameColumn Modulo<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise modulus operation on each value in the column
        /// </summary>
        public virtual DataFrameColumn ReverseModulo<T>(T value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn And(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise boolean And on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> And(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise boolean And on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ReverseAnd(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Or(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise boolean Or on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> Or(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise boolean Or on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ReverseOr(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual DataFrameColumn Xor(DataFrameColumn column, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise boolean Xor on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> Xor(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs a reversed element wise boolean Xor on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ReverseXor(bool value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise left shift on each value in the column
        /// </summary>
        public virtual DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise right shift on each value in the column
        /// </summary>
        public virtual DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseEquals(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise equals on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseEquals<T>(T value)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise not-equals on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseNotEquals<T>(T value)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqual(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise greater than or equal on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqual<T>(T value)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqual(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise less than or equal on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqual<T>(T value)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseGreaterThan(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise greater than on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseGreaterThan<T>(T value)
        {
            throw new NotImplementedException();
        }

        public virtual PrimitiveDataFrameColumn<bool> ElementwiseLessThan(DataFrameColumn column)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Performs an element wise less than on each value in the column
        /// </summary>
        public virtual PrimitiveDataFrameColumn<bool> ElementwiseLessThan<T>(T value)
        {
            throw new NotImplementedException();
        }

    }
}
