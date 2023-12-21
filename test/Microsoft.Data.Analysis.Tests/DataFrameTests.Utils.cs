// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Apache.Arrow;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
    {
        public static DataFrame MakeDataFrameWithTwoColumns(int length, bool withNulls = true)
        {
            DataFrameColumn dataFrameColumn1 = new Int32DataFrameColumn("Int1", Enumerable.Range(0, length).Select(x => x));
            DataFrameColumn dataFrameColumn2 = new Int32DataFrameColumn("Int2", Enumerable.Range(10, length).Select(x => x));
            if (withNulls)
            {
                dataFrameColumn1[length / 2] = null;
                dataFrameColumn2[length / 2] = null;
            }
            DataFrame dataFrame = new DataFrame();
            dataFrame.Columns.Insert(0, dataFrameColumn1);
            dataFrame.Columns.Insert(1, dataFrameColumn2);
            return dataFrame;
        }

        public static ArrowStringDataFrameColumn CreateArrowStringColumn(int length, bool withNulls = true)
        {
            byte[] dataMemory = new byte[length * 3];
            byte[] nullMemory = new byte[BitUtility.ByteCount(length)];
            byte[] offsetMemory = new byte[(length + 1) * 4];

            // Initialize offset with 0 as the first value
            offsetMemory[0] = 0;
            offsetMemory[1] = 0;
            offsetMemory[2] = 0;
            offsetMemory[3] = 0;

            // Append "foo" length times, with a possible `null` in the middle
            int validStringsIndex = 0;
            for (int i = 0; i < length; i++)
            {
                if (withNulls && i == length / 2)
                {
                    BitUtility.SetBit(nullMemory, i, false);
                }
                else
                {
                    int dataMemoryIndex = validStringsIndex * 3;
                    dataMemory[dataMemoryIndex++] = 102;
                    dataMemory[dataMemoryIndex++] = 111;
                    dataMemory[dataMemoryIndex++] = 111;
                    BitUtility.SetBit(nullMemory, i, true);

                    validStringsIndex++;
                }

                // write the current length to (index + 1)
                int offsetIndex = (i + 1) * 4;
                int offsetValue = 3 * validStringsIndex;
                byte[] offsetValueBytes = BitConverter.GetBytes(offsetValue);
                offsetMemory[offsetIndex++] = offsetValueBytes[0];
                offsetMemory[offsetIndex++] = offsetValueBytes[1];
                offsetMemory[offsetIndex++] = offsetValueBytes[2];
                offsetMemory[offsetIndex++] = offsetValueBytes[3];
            }

            int nullCount = withNulls ? 1 : 0;
            return new ArrowStringDataFrameColumn("ArrowString", dataMemory, offsetMemory, nullMemory, length, nullCount);
        }

        public static VBufferDataFrameColumn<int> CreateVBufferDataFrameColumn(int length)
        {
            var buffers = Enumerable.Repeat(new VBuffer<int>(5, new[] { 0, 1, 2, 3, 4 }), length).ToArray();
            return new VBufferDataFrameColumn<int>("VBuffer", buffers);
        }

        public static DataFrame MakeDataFrameWithAllColumnTypes(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithAllMutableAndArrowColumnTypes(length, withNulls);

            var vBufferColumn = CreateVBufferDataFrameColumn(length);
            df.Columns.Insert(df.Columns.Count, vBufferColumn);

            return df;
        }

        public static DataFrame MakeDataFrameWithAllMutableAndArrowColumnTypes(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(length, withNulls);
            DataFrameColumn arrowStringColumn = CreateArrowStringColumn(length, withNulls);
            df.Columns.Insert(df.Columns.Count, arrowStringColumn);

            return df;
        }

        public static DataFrame MakeDataFrameWithAllMutableColumnTypes(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericStringAndDateTimeColumns(length, withNulls);
            DataFrameColumn boolColumn = new BooleanDataFrameColumn("Bool", Enumerable.Range(0, length).Select(x => x % 2 == 0));
            df.Columns.Insert(df.Columns.Count, boolColumn);
            if (withNulls)
            {
                boolColumn[length / 2] = null;
            }
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericAndBoolColumns(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericColumns(length, withNulls);
            DataFrameColumn boolColumn = new BooleanDataFrameColumn("Bool", Enumerable.Range(0, length).Select(x => x % 2 == 0));
            df.Columns.Insert(df.Columns.Count, boolColumn);
            if (withNulls)
            {
                boolColumn[length / 2] = null;
            }
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericAndStringColumns(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericColumns(length, withNulls);
            DataFrameColumn stringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, length).Select(x => x.ToString()));
            df.Columns.Insert(df.Columns.Count, stringColumn);
            if (withNulls)
            {
                stringColumn[length / 2] = null;
            }

            DataFrameColumn charColumn = new CharDataFrameColumn("Char", Enumerable.Range(0, length).Select(x => (char)(x + 65)));
            df.Columns.Insert(df.Columns.Count, charColumn);
            if (withNulls)
            {
                charColumn[length / 2] = null;
            }
            return df;
        }

        internal static DateTime SampleDateTime = new DateTime(2021, 06, 04);
        public static DataFrame MakeDataFrameWithNumericStringAndDateTimeColumns(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericAndStringColumns(length, withNulls);

            DataFrameColumn dateTimeColumn = new DateTimeDataFrameColumn("DateTime", Enumerable.Range(0, length).Select(x => SampleDateTime.AddDays(x)));
            df.Columns.Insert(df.Columns.Count, dateTimeColumn);
            if (withNulls)
            {
                dateTimeColumn[length / 2] = null;
            }
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericColumns(int length, bool withNulls = true, int startingFrom = 0)
        {
            IEnumerable<int> range = Enumerable.Range(startingFrom, length);

            var byteColumn = new ByteDataFrameColumn("Byte", range.Select(x => (byte)x));
            var decimalColumn = new DecimalDataFrameColumn("Decimal", range.Select(x => (decimal)x));
            var doubleColumn = new DoubleDataFrameColumn("Double", range.Select(x => (double)x));
            var floatColumn = new SingleDataFrameColumn("Float", range.Select(x => (float)x));
            var intColumn = new Int32DataFrameColumn("Int", range.Select(x => x));
            var longColumn = new Int64DataFrameColumn("Long", range.Select(x => (long)x));
            var sbyteColumn = new SByteDataFrameColumn("Sbyte", range.Select(x => (sbyte)x));
            var shortColumn = new Int16DataFrameColumn("Short", range.Select(x => (short)x));
            var uintColumn = new UInt32DataFrameColumn("Uint", range.Select(x => (uint)x));
            var ulongColumn = new UInt64DataFrameColumn("Ulong", range.Select(x => (ulong)x));
            var ushortColumn = new UInt16DataFrameColumn("Ushort", range.Select(x => (ushort)x));

            var columnsList = new List<DataFrameColumn>
            {
                byteColumn,
                decimalColumn,
                doubleColumn,
                floatColumn,
                intColumn,
                longColumn,
                sbyteColumn,
                shortColumn,
                uintColumn,
                ulongColumn,
                ushortColumn
            };

            var dataFrame = new DataFrame(columnsList);

            if (withNulls)
            {
                for (var i = 0; i < dataFrame.Columns.Count; i++)
                {
                    dataFrame.Columns[i][length / 2] = null;
                }
            }

            return dataFrame;
        }

        public static DataFrame MakeDataFrame<T1, T2>(int length, bool withNulls = true)
            where T1 : unmanaged
            where T2 : unmanaged
        {
            DataFrameColumn baseColumn1 = DataFrameColumn.Create("Column1", Enumerable.Range(0, length).Select(x => (T1)Convert.ChangeType(x % 2 == 0 ? 0 : 1, typeof(T1))));
            DataFrameColumn baseColumn2 = DataFrameColumn.Create("Column2", Enumerable.Range(0, length).Select(x => (T2)Convert.ChangeType(x % 2 == 0 ? 0 : 1, typeof(T2))));
            DataFrame dataFrame = new DataFrame(new List<DataFrameColumn> { baseColumn1, baseColumn2 });

            if (withNulls)
            {
                for (int i = 0; i < dataFrame.Columns.Count; i++)
                {
                    dataFrame.Columns[i][length / 2] = null;
                }
            }

            return dataFrame;
        }

        public DataFrame SplitTrainTest(DataFrame input, float testRatio, out DataFrame Test)
        {
            IEnumerable<int> randomIndices = Enumerable.Range(0, (int)input.Rows.Count);
            IEnumerable<int> trainIndices = randomIndices.Take((int)(input.Rows.Count * testRatio));
            IEnumerable<int> testIndices = randomIndices.Skip((int)(input.Rows.Count * testRatio));
            Test = input[testIndices];
            return input[trainIndices];
        }
    }
}
