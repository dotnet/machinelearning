
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.Computations.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public abstract partial class DataFrameColumn
    {
        /// <summary>
        /// Updates each numeric element with its absolute numeric value
        /// </summary>
        public virtual DataFrameColumn Abs(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns whether all the elements are True
        /// </summary>
        public virtual bool All()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns whether any element is True
        /// </summary>
        public virtual bool Any()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates each element with its cumulative maximum
        /// </summary>
        public virtual DataFrameColumn CumulativeMax(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates column values at rowIndices with its cumulative rowIndices maximum
        /// </summary>
        public virtual DataFrameColumn CumulativeMax(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates each element with its cumulative minimum
        /// </summary>
        public virtual DataFrameColumn CumulativeMin(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates column values at rowIndices with its cumulative rowIndices minimum
        /// </summary>
        public virtual DataFrameColumn CumulativeMin(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates each element with its cumulative product
        /// </summary>
        public virtual DataFrameColumn CumulativeProduct(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates column values at rowIndices with its cumulative rowIndices product
        /// </summary>
        public virtual DataFrameColumn CumulativeProduct(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates each element with its cumulative sum
        /// </summary>
        public virtual DataFrameColumn CumulativeSum(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Updates column values at rowIndices with its cumulative rowIndices sum
        /// </summary>
        public virtual DataFrameColumn CumulativeSum(IEnumerable<long> rowIndices, bool inPlace = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the maximum of the values in the column
        /// </summary>
        public virtual object Max()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the maximum of the values at rowIndices
        /// </summary>
        public virtual object Max(IEnumerable<long> rowIndices)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of the values in the column
        /// </summary>
        public virtual object Min()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of the values at the rowIndices
        /// </summary>
        public virtual object Min(IEnumerable<long> rowIndices)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the product of the values in the column
        /// </summary>
        public virtual object Product()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the product of the values at the rowIndices
        /// </summary>
        public virtual object Product(IEnumerable<long> rowIndices)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of the values in the column
        /// </summary>
        public virtual object Sum()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of the values at the rowIndices
        /// </summary>
        public virtual object Sum(IEnumerable<long> rowIndices)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Calls Math.Round on each value in a column
        /// </summary>
        public virtual DataFrameColumn Round(bool inPlace = false)
        {
            throw new NotImplementedException();
        }

    }
}
