

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumn.Computations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class PrimitiveDataFrameColumn<T> : DataFrameColumn
        where T : unmanaged
    {
        /// <inheritdoc/>
        public override DataFrameColumn Abs(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.Abs(ret._columnContainer);
            return ret;
        }
        /// <inheritdoc/>
        public override bool All()
        {
            PrimitiveColumnComputation<T>.Instance.All(_columnContainer, out bool ret);
            return ret;
        }
        /// <inheritdoc/>
        public override bool Any()
        {
            PrimitiveColumnComputation<T>.Instance.Any(_columnContainer, out bool ret);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeMax(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeMax(ret._columnContainer);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeMax(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeMax(ret._columnContainer, rowIndices);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeMin(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeMin(ret._columnContainer);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeMin(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeMin(ret._columnContainer, rowIndices);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeProduct(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeProduct(ret._columnContainer);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeProduct(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeProduct(ret._columnContainer, rowIndices);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeSum(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeSum(ret._columnContainer);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn CumulativeSum(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.CumulativeSum(ret._columnContainer, rowIndices);
            return ret;
        }
        /// <inheritdoc/>
        public override object Max()
        {
            PrimitiveColumnComputation<T>.Instance.Max(_columnContainer, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Max(IEnumerable<long> rowIndices)
        {
            PrimitiveColumnComputation<T>.Instance.Max(_columnContainer, rowIndices, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Min()
        {
            PrimitiveColumnComputation<T>.Instance.Min(_columnContainer, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Min(IEnumerable<long> rowIndices)
        {
            PrimitiveColumnComputation<T>.Instance.Min(_columnContainer, rowIndices, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Product()
        {
            PrimitiveColumnComputation<T>.Instance.Product(_columnContainer, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Product(IEnumerable<long> rowIndices)
        {
            PrimitiveColumnComputation<T>.Instance.Product(_columnContainer, rowIndices, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Sum()
        {
            PrimitiveColumnComputation<T>.Instance.Sum(_columnContainer, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override object Sum(IEnumerable<long> rowIndices)
        {
            PrimitiveColumnComputation<T>.Instance.Sum(_columnContainer, rowIndices, out T ret);
            return ret;
        }
        /// <inheritdoc/>
        public override DataFrameColumn Round(bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();
            PrimitiveColumnComputation<T>.Instance.Round(ret._columnContainer);
            return ret;
        }
    }
}
