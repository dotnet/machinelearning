// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public enum JoinAlgorithm
    {
        Left,
        Right,
        FullOuter,
        Inner
    }

    /// <summary>
    /// A DataFrame to support indexing, binary operations, sorting, selection and other APIs. This will eventually also expose an IDataView for ML.NET
    /// </summary>
    public partial class DataFrame
    {

        private void SetSuffixForDuplicatedColumnNames(DataFrame dataFrame, DataFrameColumn column, string leftSuffix, string rightSuffix)
        {
            int index = dataFrame._columnCollection.IndexOf(column.Name);
            while (index != -1)
            {
                // Pre-existing column. Change name
                DataFrameColumn existingColumn = dataFrame.Columns[index];
                dataFrame._columnCollection.SetColumnName(existingColumn, existingColumn.Name + leftSuffix);
                column.SetName(column.Name + rightSuffix);
                index = dataFrame._columnCollection.IndexOf(column.Name);
            }
        }

        /// <summary>
        /// Joins columns of another <see cref="DataFrame"/>
        /// </summary>
        /// <param name="other">The other <see cref="DataFrame"/> to join.</param>
        /// <param name="leftSuffix">The suffix to add to this <see cref="DataFrame"/>'s column if there are common column names</param>
        /// <param name="rightSuffix">The suffix to add to the <paramref name="other"/>'s column if there are common column names</param>
        /// <param name="joinAlgorithm">The <see cref="JoinAlgorithm"/> to use.</param>
        /// <returns>A new <see cref="DataFrame"/></returns>
        public DataFrame Join(DataFrame other, string leftSuffix = "_left", string rightSuffix = "_right", JoinAlgorithm joinAlgorithm = JoinAlgorithm.Left)
        {
            DataFrame ret = new DataFrame();
            if (joinAlgorithm == JoinAlgorithm.Left)
            {
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn newColumn = Columns[i].Clone();
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
                long minLength = Math.Min(Rows.Count, other.Rows.Count);
                PrimitiveDataFrameColumn<long> mapIndices = new PrimitiveDataFrameColumn<long>("mapIndices", minLength);
                for (long i = 0; i < minLength; i++)
                {
                    mapIndices[i] = i;
                }
                for (int i = 0; i < other.Columns.Count; i++)
                {
                    DataFrameColumn newColumn;
                    if (other.Rows.Count < Rows.Count)
                    {
                        newColumn = other.Columns[i].Clone(numberOfNullsToAppend: Rows.Count - other.Rows.Count);
                    }
                    else
                    {
                        newColumn = other.Columns[i].Clone(mapIndices);
                    }
                    SetSuffixForDuplicatedColumnNames(ret, newColumn, leftSuffix, rightSuffix);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.Right)
            {
                long minLength = Math.Min(Rows.Count, other.Rows.Count);
                PrimitiveDataFrameColumn<long> mapIndices = new PrimitiveDataFrameColumn<long>("mapIndices", minLength);
                for (long i = 0; i < minLength; i++)
                {
                    mapIndices[i] = i;
                }
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn newColumn;
                    if (Rows.Count < other.Rows.Count)
                    {
                        newColumn = Columns[i].Clone(numberOfNullsToAppend: other.Rows.Count - Rows.Count);
                    }
                    else
                    {
                        newColumn = Columns[i].Clone(mapIndices);
                    }
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
                for (int i = 0; i < other.Columns.Count; i++)
                {
                    DataFrameColumn newColumn = other.Columns[i].Clone();
                    SetSuffixForDuplicatedColumnNames(ret, newColumn, leftSuffix, rightSuffix);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.FullOuter)
            {
                long newRowCount = Math.Max(Rows.Count, other.Rows.Count);
                long numberOfNulls = newRowCount - Rows.Count;
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn newColumn = Columns[i].Clone(numberOfNullsToAppend: numberOfNulls);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
                numberOfNulls = newRowCount - other.Rows.Count;
                for (int i = 0; i < other.Columns.Count; i++)
                {
                    DataFrameColumn newColumn = other.Columns[i].Clone(numberOfNullsToAppend: numberOfNulls);
                    SetSuffixForDuplicatedColumnNames(ret, newColumn, leftSuffix, rightSuffix);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.Inner)
            {
                long newRowCount = Math.Min(Rows.Count, other.Rows.Count);
                PrimitiveDataFrameColumn<long> mapIndices = new PrimitiveDataFrameColumn<long>("mapIndices", newRowCount);
                for (long i = 0; i < newRowCount; i++)
                {
                    mapIndices[i] = i;
                }
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn newColumn = Columns[i].Clone(mapIndices);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
                for (int i = 0; i < other.Columns.Count; i++)
                {
                    DataFrameColumn newColumn = other.Columns[i].Clone(mapIndices);
                    SetSuffixForDuplicatedColumnNames(ret, newColumn, leftSuffix, rightSuffix);
                    ret.Columns.Insert(ret.Columns.Count, newColumn);
                }
            }
            return ret;
        }

        // TODO: Merge API with an "On" parameter that merges on a column common to 2 dataframes 

        /// <summary> 
        /// Merge DataFrames with a database style join 
        /// </summary> 
        /// <param name="other"></param> 
        /// <param name="leftJoinColumn"></param> 
        /// <param name="rightJoinColumn"></param> 
        /// <param name="leftSuffix"></param> 
        /// <param name="rightSuffix"></param> 
        /// <param name="joinAlgorithm"></param> 
        /// <returns></returns> 
        public DataFrame Merge<TKey>(DataFrame other, string leftJoinColumn, string rightJoinColumn, string leftSuffix = "_left", string rightSuffix = "_right", JoinAlgorithm joinAlgorithm = JoinAlgorithm.Left)
        {
            // A simple hash join 
            DataFrame ret = new DataFrame();
            DataFrame leftDataFrame = this;
            DataFrame rightDataFrame = other;

            // The final table size is not known until runtime 
            long rowNumber = 0;
            PrimitiveDataFrameColumn<long> leftRowIndices = new PrimitiveDataFrameColumn<long>("LeftIndices");
            PrimitiveDataFrameColumn<long> rightRowIndices = new PrimitiveDataFrameColumn<long>("RightIndices");
            if (joinAlgorithm == JoinAlgorithm.Left)
            {
                // First hash other dataframe on the rightJoinColumn 
                DataFrameColumn otherColumn = other.Columns[rightJoinColumn];
                Dictionary<TKey, ICollection<long>> multimap = otherColumn.GroupColumnValues<TKey>(out HashSet<long> otherColumnNullIndices);

                // Go over the records in this dataframe and match with the dictionary 
                DataFrameColumn thisColumn = Columns[leftJoinColumn];

                for (long i = 0; i < thisColumn.Length; i++)
                {
                    var thisColumnValue = thisColumn[i];
                    if (thisColumnValue != null)
                    {
                        if (multimap.TryGetValue((TKey)thisColumnValue, out ICollection<long> rowNumbers))
                        {
                            foreach (long row in rowNumbers)
                            {
                                leftRowIndices.Append(i);
                                rightRowIndices.Append(row);
                            }
                        }
                        else
                        {
                            leftRowIndices.Append(i);
                            rightRowIndices.Append(null);
                        }
                    }
                    else
                    {
                        foreach (long row in otherColumnNullIndices)
                        {
                            leftRowIndices.Append(i);
                            rightRowIndices.Append(row);
                        }
                    }
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.Right)
            {
                DataFrameColumn thisColumn = Columns[leftJoinColumn];
                Dictionary<TKey, ICollection<long>> multimap = thisColumn.GroupColumnValues<TKey>(out HashSet<long> thisColumnNullIndices);

                DataFrameColumn otherColumn = other.Columns[rightJoinColumn];
                for (long i = 0; i < otherColumn.Length; i++)
                {
                    var otherColumnValue = otherColumn[i];
                    if (otherColumnValue != null)
                    {
                        if (multimap.TryGetValue((TKey)otherColumnValue, out ICollection<long> rowNumbers))
                        {
                            foreach (long row in rowNumbers)
                            {
                                leftRowIndices.Append(row);
                                rightRowIndices.Append(i);
                            }
                        }
                        else
                        {
                            leftRowIndices.Append(null);
                            rightRowIndices.Append(i);
                        }
                    }
                    else
                    {
                        foreach (long thisColumnNullIndex in thisColumnNullIndices)
                        {
                            leftRowIndices.Append(thisColumnNullIndex);
                            rightRowIndices.Append(i);
                        }
                    }
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.Inner)
            {
                // Hash the column with the smaller RowCount 
                long leftRowCount = Rows.Count;
                long rightRowCount = other.Rows.Count;

                bool leftColumnIsSmaller = leftRowCount <= rightRowCount;
                DataFrameColumn hashColumn = leftColumnIsSmaller ? Columns[leftJoinColumn] : other.Columns[rightJoinColumn];
                DataFrameColumn otherColumn = ReferenceEquals(hashColumn, Columns[leftJoinColumn]) ? other.Columns[rightJoinColumn] : Columns[leftJoinColumn];
                Dictionary<TKey, ICollection<long>> multimap = hashColumn.GroupColumnValues<TKey>(out HashSet<long> smallerDataFrameColumnNullIndices);

                for (long i = 0; i < otherColumn.Length; i++)
                {
                    var otherColumnValue = otherColumn[i];
                    if (otherColumnValue != null)
                    {
                        if (multimap.TryGetValue((TKey)otherColumnValue, out ICollection<long> rowNumbers))
                        {
                            foreach (long row in rowNumbers)
                            {
                                leftRowIndices.Append(leftColumnIsSmaller ? row : i);
                                rightRowIndices.Append(leftColumnIsSmaller ? i : row);
                            }
                        }
                    }
                    else
                    {
                        foreach (long nullIndex in smallerDataFrameColumnNullIndices)
                        {
                            leftRowIndices.Append(leftColumnIsSmaller ? nullIndex : i);
                            rightRowIndices.Append(leftColumnIsSmaller ? i : nullIndex);
                        }
                    }
                }
            }
            else if (joinAlgorithm == JoinAlgorithm.FullOuter)
            {
                DataFrameColumn otherColumn = other.Columns[rightJoinColumn];
                Dictionary<TKey, ICollection<long>> multimap = otherColumn.GroupColumnValues<TKey>(out HashSet<long> otherColumnNullIndices);
                Dictionary<TKey, long> intersection = new Dictionary<TKey, long>(EqualityComparer<TKey>.Default);

                // Go over the records in this dataframe and match with the dictionary 
                DataFrameColumn thisColumn = Columns[leftJoinColumn];
                Int64DataFrameColumn thisColumnNullIndices = new Int64DataFrameColumn("ThisColumnNullIndices");

                for (long i = 0; i < thisColumn.Length; i++)
                {
                    var thisColumnValue = thisColumn[i];
                    if (thisColumnValue != null)
                    {
                        if (multimap.TryGetValue((TKey)thisColumnValue, out ICollection<long> rowNumbers))
                        {
                            foreach (long row in rowNumbers)
                            {
                                leftRowIndices.Append(i);
                                rightRowIndices.Append(row);
                                if (!intersection.ContainsKey((TKey)thisColumnValue))
                                {
                                    intersection.Add((TKey)thisColumnValue, rowNumber);
                                }
                            }
                        }
                        else
                        {
                            leftRowIndices.Append(i);
                            rightRowIndices.Append(null);
                        }
                    }
                    else
                    {
                        thisColumnNullIndices.Append(i);
                    }
                }
                for (long i = 0; i < otherColumn.Length; i++)
                {
                    var value = otherColumn[i];
                    if (value != null)
                    {
                        if (!intersection.ContainsKey((TKey)value))
                        {
                            leftRowIndices.Append(null);
                            rightRowIndices.Append(i);
                        }
                    }
                }

                // Now handle the null rows
                foreach (long? thisColumnNullIndex in thisColumnNullIndices)
                {
                    foreach (long otherColumnNullIndex in otherColumnNullIndices)
                    {
                        leftRowIndices.Append(thisColumnNullIndex.Value);
                        rightRowIndices.Append(otherColumnNullIndex);
                    }
                    if (otherColumnNullIndices.Count == 0)
                    {
                        leftRowIndices.Append(thisColumnNullIndex.Value);
                        rightRowIndices.Append(null);
                    }
                }
                if (thisColumnNullIndices.Length == 0)
                {
                    foreach (long otherColumnNullIndex in otherColumnNullIndices)
                    {
                        leftRowIndices.Append(null);
                        rightRowIndices.Append(otherColumnNullIndex);
                    }
                }
            }
            else
                throw new NotImplementedException(nameof(joinAlgorithm));

            for (int i = 0; i < leftDataFrame.Columns.Count; i++)
            {
                ret.Columns.Insert(i, leftDataFrame.Columns[i].Clone(leftRowIndices));
            }
            for (int i = 0; i < rightDataFrame.Columns.Count; i++)
            {
                DataFrameColumn column = rightDataFrame.Columns[i].Clone(rightRowIndices);
                SetSuffixForDuplicatedColumnNames(ret, column, leftSuffix, rightSuffix);
                ret.Columns.Insert(ret.Columns.Count, column);
            }
            return ret;
        }

    }

}
