
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameBinaryOperations.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame
    {
        public DataFrame Add<T>(IReadOnlyList<T> values, bool inPlace = false)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Add(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise addition on each column
        /// </summary>
        public DataFrame Add<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Add(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Subtract<T>(IReadOnlyList<T> values, bool inPlace = false)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Subtract(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise subtraction on each column
        /// </summary>
        public DataFrame Subtract<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Subtract(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Multiply<T>(IReadOnlyList<T> values, bool inPlace = false)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Multiply(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise multiplication on each column
        /// </summary>
        public DataFrame Multiply<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Multiply(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Divide<T>(IReadOnlyList<T> values, bool inPlace = false)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Divide(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise division on each column
        /// </summary>
        public DataFrame Divide<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Divide(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Modulo<T>(IReadOnlyList<T> values, bool inPlace = false)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Modulo(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise modulus operation on each column
        /// </summary>
        public DataFrame Modulo<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Modulo(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame And(IReadOnlyList<bool> values, bool inPlace = false)
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.And(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise boolean And on each column
        /// </summary>
        public DataFrame And(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.And(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Or(IReadOnlyList<bool> values, bool inPlace = false)
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Or(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise boolean Or on each column
        /// </summary>
        public DataFrame Or(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Or(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame Xor(IReadOnlyList<bool> values, bool inPlace = false)
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Xor(values[i], inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise boolean Xor on each column
        /// </summary>
        public DataFrame Xor(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.Xor(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise left shift on each column
        /// </summary>
        public DataFrame LeftShift(int value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.LeftShift(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise right shift on each column
        /// </summary>
        public DataFrame RightShift(int value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.RightShift(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseEquals<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseEquals(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise equals on each column
        /// </summary>
        public DataFrame ElementwiseEquals<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseEquals(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseNotEquals<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseNotEquals(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise not-equals on each column
        /// </summary>
        public DataFrame ElementwiseNotEquals<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseNotEquals(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseGreaterThanOrEqual<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseGreaterThanOrEqual(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise greater than or equal on each column
        /// </summary>
        public DataFrame ElementwiseGreaterThanOrEqual<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseGreaterThanOrEqual(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseLessThanOrEqual<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseLessThanOrEqual(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise less than or equal on each column
        /// </summary>
        public DataFrame ElementwiseLessThanOrEqual<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseLessThanOrEqual(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseGreaterThan<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseGreaterThan(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise greater than on each column
        /// </summary>
        public DataFrame ElementwiseGreaterThan<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseGreaterThan(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        public DataFrame ElementwiseLessThan<T>(IReadOnlyList<T> values)
            where T : unmanaged
        {
            if (values.Count != Columns.Count)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));
            }
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseLessThan(values[i]);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs an element wise less than on each column
        /// </summary>
        public DataFrame ElementwiseLessThan<T>(T value)
            where T : unmanaged
        {
            DataFrame retDataFrame = new DataFrame();

            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ElementwiseLessThan(value);
                retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }

        /// <summary>
        /// Performs a reversed element wise addition on each column
        /// </summary>
        public DataFrame ReverseAdd<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseAdd(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise subtraction on each column
        /// </summary>
        public DataFrame ReverseSubtract<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseSubtract(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise multiplication on each column
        /// </summary>
        public DataFrame ReverseMultiply<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseMultiply(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise division on each column
        /// </summary>
        public DataFrame ReverseDivide<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseDivide(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise modulus operation on each column
        /// </summary>
        public DataFrame ReverseModulo<T>(T value, bool inPlace = false)
            where T : unmanaged
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseModulo(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise boolean And on each column
        /// </summary>
        public DataFrame ReverseAnd(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseAnd(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise boolean Or on each column
        /// </summary>
        public DataFrame ReverseOr(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseOr(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
        /// <summary>
        /// Performs a reversed element wise boolean Xor on each column
        /// </summary>
        public DataFrame ReverseXor(bool value, bool inPlace = false)
        {
            DataFrame retDataFrame = inPlace ? this : new DataFrame();
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn baseColumn = _columnCollection[i];
                DataFrameColumn newColumn = baseColumn.ReverseXor(value, inPlace);
                if (inPlace)
                    retDataFrame.Columns[i] = newColumn;
                else
                    retDataFrame.Columns.Insert(i, newColumn);
            }
            return retDataFrame;
        }
    }
}
