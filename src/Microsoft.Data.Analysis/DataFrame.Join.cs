// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

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

        private static bool IsAnyNullValueInColumns (IReadOnlyCollection<DataFrameColumn> columns, long index)
        {
            foreach (var column in columns)
            {
                if (column[index] == null)
                    return true;
            }
            return false;
        }
                      
        private static HashSet<long> Merge(DataFrame retainedDataFrame, DataFrame supplementaryDataFrame, string[] retainedJoinColumnNames, string[] supplemetaryJoinColumnNames, out PrimitiveDataFrameColumn<long> retainedRowIndices, out PrimitiveDataFrameColumn<long> supplementaryRowIndices, bool isInner = false, bool calculateIntersection = false)
        {
            if (retainedJoinColumnNames == null)
                throw new ArgumentNullException(nameof(retainedJoinColumnNames));

            if (supplemetaryJoinColumnNames == null)
                throw new ArgumentNullException(nameof(supplemetaryJoinColumnNames));

            if (retainedJoinColumnNames.Length != supplemetaryJoinColumnNames.Length)
                throw new ArgumentException("", nameof(retainedJoinColumnNames));  //TODO provide correct message for the exception
            

            HashSet<long> intersection = calculateIntersection ? new HashSet<long>() : null;
                        
            Dictionary<long, ICollection<long>> occurrences = null;

            // Get occurrences of values in columns used for join in the retained and supplementary dataframes
            Dictionary<long, long> retainedIndicesReverseMapping = null; 
            Dictionary<long, ICollection<long>> rowOccurrences = null;

            HashSet<long> supplementaryJoinColumnsNullIndices = new HashSet<long>();

            for (int colNameIndex =  0; colNameIndex < retainedJoinColumnNames.Length; colNameIndex++)
            {
                DataFrameColumn shrinkedRetainedColumn = retainedDataFrame.Columns[retainedJoinColumnNames[colNameIndex]];

                //shrink retained column by row occurrences from previouse step
                if (rowOccurrences != null)
                {
                    var shrinkedRetainedIndices = rowOccurrences.Keys.ToArray();
                                                            
                    var newRetainedIndicesReverseMapping = new Dictionary<long, long>();
                    for (int i = 0; i < shrinkedRetainedIndices.Length; i++)
                    { 
                        //store reverse mapping to restore original dataframe indices from indices in shrinked row
                        newRetainedIndicesReverseMapping.Add(i, retainedIndicesReverseMapping != null ? retainedIndicesReverseMapping[shrinkedRetainedIndices[i]] : shrinkedRetainedIndices[i] );
                    }

                    retainedIndicesReverseMapping = newRetainedIndicesReverseMapping;
                    shrinkedRetainedColumn = shrinkedRetainedColumn.Clone(new Int64DataFrameColumn("Indices", shrinkedRetainedIndices));
                }
                
                DataFrameColumn supplementaryColumn = supplementaryDataFrame.Columns[supplemetaryJoinColumnNames[colNameIndex]];

                var newOccurrences = shrinkedRetainedColumn.GetGroupedOccurrences(supplementaryColumn, out HashSet<long> supplementaryColumnNullIndices);

                supplementaryJoinColumnsNullIndices.UnionWith(supplementaryColumnNullIndices);
                
                // shrink join result on current column by previouse join columns (if any)
                if (rowOccurrences != null)
                {
                    var shrinkedOccurences = new Dictionary<long, ICollection<long>>();

                    foreach (var kvp in newOccurrences)
                    {
                        var newValue = kvp.Value.Where(i => rowOccurrences[retainedIndicesReverseMapping[kvp.Key]].Contains(i)).ToArray();
                        if (newValue.Any())
                        {
                            shrinkedOccurences.Add(kvp.Key, newValue);
                        }
                    }
                    newOccurrences = shrinkedOccurences;
                }
                rowOccurrences = newOccurrences;
            }

            //Restore occurences
            occurrences = rowOccurrences.ToDictionary(kvp => retainedIndicesReverseMapping == null ? kvp.Key : retainedIndicesReverseMapping[kvp.Key], kvp => kvp.Value);
            
            retainedRowIndices = new Int64DataFrameColumn("RetainedIndices");
            supplementaryRowIndices = new Int64DataFrameColumn("SupplementaryIndices");

            //Perform Merging 
            var retainJoinColumns = retainedJoinColumnNames.Select(name => retainedDataFrame.Columns[name]).ToArray();
            for (long i = 0; i < retainedDataFrame.Columns.RowCount; i++)
            {
                if (!IsAnyNullValueInColumns(retainJoinColumns, i))
                {
                    //Get all row indexes from supplementary dataframe that sutisfy JOIN condition
                    if (occurrences.TryGetValue(i, out ICollection<long> rowIndices))
                    {
                        foreach (long supplementaryRowIndex in rowIndices)
                        {
                            retainedRowIndices.Append(i);
                            supplementaryRowIndices.Append(supplementaryRowIndex);

                            //store intersection if required
                            if (calculateIntersection)
                            {
                                if (!intersection.Contains(supplementaryRowIndex))
                                {
                                    intersection.Add(supplementaryRowIndex);
                                }
                            }
                        }
                    }
                    else
                    {
                        if (isInner)
                            continue;

                        retainedRowIndices.Append(i);
                        supplementaryRowIndices.Append(null);
                    }
                }
                else
                {                    
                    foreach (long row in supplementaryJoinColumnsNullIndices)
                    {
                        retainedRowIndices.Append(i);
                        supplementaryRowIndices.Append(row);
                    }
                    
                }
            }
                    
            return intersection;
        }

        /// <summary> 
        /// Merge DataFrames with a database style join (for backward compatibility)
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
            return Merge(other, new[] { leftJoinColumn }, new[] { rightJoinColumn }, leftSuffix, rightSuffix, joinAlgorithm);
        }
        
            
        public DataFrame Merge(DataFrame other, string[] leftJoinColumns, string[] rightJoinColumns, string leftSuffix = "_left", string rightSuffix = "_right", JoinAlgorithm joinAlgorithm = JoinAlgorithm.Left)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            //In Outer join the joined dataframe retains each row — even if no other matching row exists in supplementary dataframe.
            //Outer joins subdivide further into left outer joins (left dataframe is retained), right outer joins (rightdataframe is retained), in full outer both are retained

            PrimitiveDataFrameColumn<long> retainedRowIndices;
            PrimitiveDataFrameColumn<long> supplementaryRowIndices;
            DataFrame supplementaryDataFrame;
            DataFrame retainedDataFrame;
            bool isLeftDataFrameRetained;

            if (joinAlgorithm == JoinAlgorithm.Left || joinAlgorithm == JoinAlgorithm.Right)
            {
                isLeftDataFrameRetained = (joinAlgorithm == JoinAlgorithm.Left);

                supplementaryDataFrame = isLeftDataFrameRetained ? other : this;
                var supplementaryJoinColumns = isLeftDataFrameRetained ? rightJoinColumns : leftJoinColumns;

                retainedDataFrame = isLeftDataFrameRetained ? this : other;
                var retainedJoinColumns = isLeftDataFrameRetained ? leftJoinColumns : rightJoinColumns;

                Merge(retainedDataFrame, supplementaryDataFrame, retainedJoinColumns, supplementaryJoinColumns, out retainedRowIndices, out supplementaryRowIndices);

            }
            else if (joinAlgorithm == JoinAlgorithm.Inner)
            {
                // use as supplementary (for Hashing) the dataframe with the smaller RowCount 
                isLeftDataFrameRetained = (Rows.Count > other.Rows.Count);

                supplementaryDataFrame = isLeftDataFrameRetained ? other : this;
                var supplementaryJoinColumns = isLeftDataFrameRetained ? rightJoinColumns : leftJoinColumns;

                retainedDataFrame = isLeftDataFrameRetained ? this : other;
                var retainedJoinColumns = isLeftDataFrameRetained ? leftJoinColumns : rightJoinColumns;

                Merge(retainedDataFrame, supplementaryDataFrame, retainedJoinColumns, supplementaryJoinColumns, out retainedRowIndices, out supplementaryRowIndices, true);
            }
            else if (joinAlgorithm == JoinAlgorithm.FullOuter)
            {
                //In full outer join we would like to retain data from both side, so we do it into 2 steps: one first we do LEFT JOIN and then add lost data from the RIGHT side
                
                //Step 1
                //Do LEFT JOIN
                isLeftDataFrameRetained = true;

                supplementaryDataFrame = isLeftDataFrameRetained ? other : this;
                var supplementaryJoinColumns = isLeftDataFrameRetained ? rightJoinColumns : leftJoinColumns;

                retainedDataFrame = isLeftDataFrameRetained ? this : other;
                var retainedJoinColumns = isLeftDataFrameRetained ? leftJoinColumns : rightJoinColumns;

                var intersection = Merge(retainedDataFrame, supplementaryDataFrame, retainedJoinColumns, supplementaryJoinColumns, out retainedRowIndices, out supplementaryRowIndices, calculateIntersection: true);

                /*
                //Step 2
                //Do RIGHT JOIN to retain all data from supplementary DataFrame too (take into account data intersection from the first step to avoid duplicates)
                DataFrameColumn supplementaryColumn = supplementaryDataFrame.Columns[supplementaryJoinColumn];

                for (long i = 0; i < supplementaryColumn.Length; i++)
                {
                    var value = supplementaryColumn[i];
                    if (value != null)
                    {
                        if (!intersection.Contains(i))
                        {
                            retainedRowIndices.Append(null);
                            supplementaryRowIndices.Append(i);
                        }
                    }
                }
                */
            }
            else
                throw new NotImplementedException(nameof(joinAlgorithm));
                                    
            DataFrame ret = new DataFrame();
                       
            //insert columns from left dataframe (this)
            for (int i = 0; i < this.Columns.Count; i++)
            {
                ret.Columns.Insert(i, this.Columns[i].Clone(isLeftDataFrameRetained ? retainedRowIndices : supplementaryRowIndices));
            }

            //insert columns from right dataframe (other)
            for (int i = 0; i < other.Columns.Count; i++)
            {
                DataFrameColumn column = other.Columns[i].Clone(isLeftDataFrameRetained ? supplementaryRowIndices : retainedRowIndices);
                SetSuffixForDuplicatedColumnNames(ret, column, leftSuffix, rightSuffix);
                ret.Columns.Insert(ret.Columns.Count, column);
            }
            return ret;
        }

    }

}
