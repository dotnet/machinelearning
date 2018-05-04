// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.ML
{
    /// <summary>
    /// The debug proxy class for a LearningPipeline.
    /// Displays the current columns and values in the debugger Watch window.
    /// </summary>
    internal sealed class LearningPipelineDebugProxy
    {
        // load more rows than we display in order for transforms like CategoricalOneHotVectorizer
        // to see more rows, and get a more accurate cardinality of the column.
        private const int MaxLoaderRows = 100;
        private const int MaxDisplayRows = 10;
        private const int MaxSlotNamesToDisplay = 100;

        private readonly LearningPipeline _pipeline;
        private readonly TlcEnvironment _environment;
        private IDataView _preview;
        private Exception _pipelineExecutionException;
        private PipelineItemDebugColumn[] _columns;
        private PipelineItemDebugRow[] _rows;

        public LearningPipelineDebugProxy(LearningPipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));

            _pipeline = new LearningPipeline();

            // use a ConcurrencyFactor of 1 so other threads don't need to run in the debugger
            _environment = new TlcEnvironment(conc: 1);

            foreach (ILearningPipelineItem item in pipeline)
            {
                _pipeline.Add(item);

                if (item is ILearningPipelineLoader loaderItem)
                {
                    // add a take filter to any loaders, so it returns in a reasonable
                    // amount of time
                    _pipeline.Add(new RowTakeFilter() { Count = MaxLoaderRows });
                }
            }
        }

        /// <summary>
        /// Gets the column information of the pipeline.
        /// </summary>
        public PipelineItemDebugColumn[] Columns
        {
            get
            {
                if (_columns == null)
                {
                    _columns = BuildColumns();
                }
                return _columns;
            }
        }

        private PipelineItemDebugColumn[] BuildColumns()
        {
            IDataView dataView = ExecutePipeline();

            var colIndices = GetColIndices(dataView);
            var colCount = colIndices.Count;

            PipelineItemDebugColumn[] result = new PipelineItemDebugColumn[colCount];

            for (int i = 0; i < colCount; i++)
            {
                var colIndex = colIndices[i];
                result[i] = new PipelineItemDebugColumn()
                {
                    Name = dataView.Schema.GetColumnName(colIndex),
                    Type = dataView.Schema.GetColumnType(colIndex).ToString()
                };

                if (dataView.Schema.GetColumnType(colIndex).IsVector)
                {
                    var n = dataView.Schema.GetColumnType(colIndex).VectorSize;
                    if (dataView.Schema.HasSlotNames(colIndex, n))
                    {
                        var slots = default(VBuffer<DvText>);
                        dataView.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colIndex, ref slots);

                        bool appendEllipse = false;
                        IEnumerable<DvText> slotNames = slots.Items(true).Select(x => x.Value);
                        if (slots.Length > MaxSlotNamesToDisplay)
                        {
                            appendEllipse = true;
                            slotNames = slotNames.Take(MaxSlotNamesToDisplay);
                        }

                        result[i].SlotNames = string.Join(",", slotNames);

                        if (appendEllipse)
                        {
                            result[i].SlotNames += ",...";
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Gets the row information of the pipeline.
        /// </summary>
        public PipelineItemDebugRow[] Rows
        {
            get
            {
                if (_rows == null)
                {
                    _rows = BuildRows();
                }
                return _rows;
            }
        }

        private IDataView ExecutePipeline()
        {
            if (_preview == null)
            {
                if (_pipeline != null)
                {
                    try
                    {
                        _preview = _pipeline.Execute(_environment);
                    }
                    catch (Exception e)
                    {
                        _pipelineExecutionException = e;
                        var fakeColumn = new KeyValuePair<string, ColumnType>("Blank", TextType.Instance);
                        _preview = new EmptyDataView(_environment, new SimpleSchema(_environment, fakeColumn));
                    }
                }
            }
            return _preview;
        }

        private PipelineItemDebugRow[] BuildRows()
        {
            PipelineItemDebugRow[] result = new PipelineItemDebugRow[MaxDisplayRows];

            int i = 0;
            IDataView pipelineResult = ExecutePipeline();
            if (_pipelineExecutionException != null)
            {
                result[0] = new PipelineItemDebugRow()
                {
                    Values = _pipelineExecutionException.ToString()
                };
                return result;
            }

            StringBuilder valuesBuilder = new StringBuilder();
            using (var cursor = pipelineResult.GetRowCursor(c => true))
            {
                var colIndices = GetColIndices(pipelineResult);
                var colCount = colIndices.Count;

                var getters = DataViewUtils.PopulateGetterArray(cursor, colIndices);

                var row = new DvText[colCount];
                while (cursor.MoveNext() && i < MaxDisplayRows)
                {
                    for (int column = 0; column < colCount; column++)
                    {
                        if (column != 0)
                        {
                            valuesBuilder.Append(" | ");
                        }

                        getters[column](ref row[column]);
                        valuesBuilder.Append(row[column]);
                    }

                    result[i] = new PipelineItemDebugRow()
                    {
                        Values = valuesBuilder.ToString()
                    };

                    valuesBuilder.Clear();
                    i++;
                }
            }

            return result;
        }

        private static List<int> GetColIndices(IDataView dataView)
        {
            int totalColCount = dataView.Schema.ColumnCount;
            // getting distinct columns
            HashSet<string> columnNames = new HashSet<string>();
            var colIndices = new List<int>();
            for (int i = totalColCount - 1; i >= 0; i--)
            {
                var name = dataView.Schema.GetColumnName(i);
                if (columnNames.Add(name))
                    colIndices.Add(i);
            }
            colIndices.Reverse();

            return colIndices;
        }
    }

    [DebuggerDisplay("{Name} {Type}{SlotNames}")]
    internal class PipelineItemDebugColumn
    {
        public string Name { get; set; }
        public string Type { get; set; }
        public string SlotNames { get; set; } = string.Empty;
    }

    [DebuggerDisplay("{Values}")]
    internal class PipelineItemDebugRow
    {
        public string Values { get; set; }
    }
}
