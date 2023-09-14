// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;

namespace Microsoft.Data.Analysis.PerformanceTests
{
    public class PerformanceTests
    {
        private const int ItemsCount = 1000000;

        private Int32DataFrameColumn _int32Column1;
        private Int32DataFrameColumn _int32Column2;

        private Int16DataFrameColumn _int16Column1;
        private Int16DataFrameColumn _int16Column2;

        private DoubleDataFrameColumn _doubleColumn1;
        private DoubleDataFrameColumn _doubleColumn2;

        private SingleDataFrameColumn _floatColumn1;
        private SingleDataFrameColumn _floatColumn2;

        [GlobalSetup]
        public void SetUp()
        {
            _int32Column1 = new Int32DataFrameColumn("Column1", ItemsCount);
            _int32Column2 = new Int32DataFrameColumn("Column2", ItemsCount);

            _int16Column1 = new Int16DataFrameColumn("Column1", ItemsCount);
            _int16Column2 = new Int16DataFrameColumn("Column2", ItemsCount);

            _doubleColumn1 = new DoubleDataFrameColumn("Column1", ItemsCount);
            _doubleColumn2 = new DoubleDataFrameColumn("Column2", ItemsCount);

            _floatColumn1 = new SingleDataFrameColumn("Column1", ItemsCount);
            _floatColumn2 = new SingleDataFrameColumn("Column2", ItemsCount);
        }

        #region Addition

        [Benchmark]
        public void Add_Int32()
        {
            var column = _int32Column1 + _int32Column2;
        }

        [Benchmark]
        public void Add_Int16()
        {
            var column = _int16Column1 + _int16Column2;
        }

        [Benchmark]
        public void Add_Double()
        {
            var column = _doubleColumn1 + _doubleColumn2;
        }

        [Benchmark]
        public void Add_Float()
        {
            var column = _floatColumn1 + _floatColumn2;
        }

        [Benchmark]
        public void Add_Int32_Int16()
        {
            var column = _int32Column1 + _int16Column2;
        }

        [Benchmark]
        public void Add_Double_Float()
        {
            var column = _doubleColumn1 + _floatColumn2;
        }
        #endregion

        #region Substract
        [Benchmark]
        public void Substract_Int32()
        {
            var column = _int32Column1 - _int32Column2;
        }

        [Benchmark]
        public void Substract_Int16()
        {
            var column = _int16Column1 - _int16Column2;
        }

        [Benchmark]
        public void Substract_Double()
        {
            var column = _doubleColumn1 - _doubleColumn2;
        }

        [Benchmark]
        public void Substract_Float()
        {
            var column = _floatColumn1 - _floatColumn2;
        }

        [Benchmark]
        public void Substract_Int32_Int16()
        {
            var column = _int32Column1 - _int16Column2;
        }

        [Benchmark]
        public void Substract_Double_Float()
        {
            var column = _doubleColumn1 - _floatColumn2;
        }
        #endregion

        #region Multiply
        [Benchmark]
        public void Multiply_Int32()
        {
            var column = _int32Column1 * _int32Column2;
        }

        [Benchmark]
        public void Multiply_Int16()
        {
            var column = _int16Column1 * _int16Column2;
        }

        [Benchmark]
        public void Multiply_Double()
        {
            var column = _doubleColumn1 * _doubleColumn2;
        }

        [Benchmark]
        public void Multiply_Float()
        {
            var column = _floatColumn1 * _floatColumn2;
        }

        [Benchmark]
        public void Multiply_Int32_Int16()
        {
            var column = _int32Column1 * _int16Column2;
        }

        [Benchmark]
        public void Multiply_Double_Float()
        {
            var column = _doubleColumn1 * _floatColumn2;
        }
        #endregion

        #region Divide
        [Benchmark]
        public void Divide_Int32()
        {
            var column = _int32Column1 / _int32Column2;
        }

        [Benchmark]
        public void Divide_Int16()
        {
            var column = _int16Column1 / _int16Column2;
        }

        [Benchmark]
        public void Divide_Double()
        {
            var column = _doubleColumn1 / _doubleColumn2;
        }

        [Benchmark]
        public void Divide_Float()
        {
            var column = _floatColumn1 / _floatColumn2;
        }

        [Benchmark]
        public void Divide_Int32_Int16()
        {
            var column = _int32Column1 / _int16Column2;
        }

        [Benchmark]
        public void Divide_Double_Float()
        {
            var column = _doubleColumn1 / _floatColumn2;
        }
        #endregion
    }
}
