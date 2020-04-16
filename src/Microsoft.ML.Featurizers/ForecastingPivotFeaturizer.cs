// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.SchemaShape.Column;

[assembly: LoadableClass(typeof(ForecastingPivotTransformer), null, typeof(SignatureLoadModel),
    ForecastingPivotTransformer.UserName, ForecastingPivotTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(ForecastingPivotTransformer), null, typeof(SignatureLoadDataTransform),
   ForecastingPivotTransformer.UserName, ForecastingPivotTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ForecastingPivotTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ForecastingPivotFeaturizerExtensionClass
    {

        /// <summary>
        /// Takes in a list of columns of type vector to pivot. Pivots them, and adds a horizon column. It will auto generate the output columns based on the annotations provided.
        /// If any of the vector slots are double.NaN, the entire row will be dropped. This currently works with the output columns from LagLeadOperator and RollingWindowFeaturizer.
        ///
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="columnsToPivot">The list of columns to pivot.</param>
        /// <param name="horizonColumnName">The of the horizon column to add.</param>
        /// <returns></returns>
        public static ForecastingPivotFeaturizerEstimator PivotForecastingData(this TransformsCatalog catalog, string[] columnsToPivot, string horizonColumnName = "Horizon")
        {
            var options = new ForecastingPivotFeaturizerEstimator.Options
            {
                ColumnsToPivot = columnsToPivot,
                HorizonColumnName = horizonColumnName
            };

            return new ForecastingPivotFeaturizerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// Takes in a list of columns of type vector to pivot. Pivots them, and adds a horizon column. It will auto generate the output columns based on the annotations provided.
    /// If any of the vector slots are double.NaN, the entire row will be dropped. This currently works with the output columns from LagLeadOperator and RollingWindowFeaturizer.
    ///
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Vector of Double |
    /// | Output column data type | double |
    /// | Exportable to ONNX | No |
    ///
    /// This featurizer takes in a list of vector columns, and pivots those columns into columns of type double, and then drops rows when the pivot results in a column with an na value.
    /// For example, given this input
    ///
    ///      +-------+--------------------------------+--------------------------------+
    ///      | Index |     RW                         |        LL                      |
    ///      +=======+================================+================================+
    ///      | 0     | [ [na, na, na] ]               | [ [na, na, na], [na, na, na] ] |
    ///      +-------+--------------------------------+--------------------------------+
    ///      | 1     | [ [1, 2, 3] ]                  | [ [na, na, na], [na, na, na] ] |
    ///      +-------+--------------------------------+--------------------------------+
    ///      | 2     | [ [1, 2, 3] ]                  | [ [na, na, na], [na, na, na] ] |
    ///      +-------+--------------------------------+--------------------------------+
    ///      | 3     | [ [1, 2, 3] ]                  | [ [3, 2, 1], [na, na, na] ]    |
    ///      +-------+--------------------------------+--------------------------------+
    ///      | 4     | [ [1, 2, 3] ]                  | [ [3, 2, 1], [2, na, na] ]     |
    ///      +-------+--------------------------------+--------------------------------+
    ///      | 5     | [ [1, 2, 3] ]                  | [ [2, 2, 1], [3, na, 4] ]      |
    ///      +-------+--------------------------------+--------------------------------+
    ///
    /// After the pivot but before rows with na values get dropped this would be
    ///
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | Index | RW_Mean_MinWin1_MaxWin_1   |   LL_Lag1  |   LL_Lead_1  | Horizon  |
    ///      +=======+============================+============+==============+==========+
    ///      | 0     | na                         | na         | na           | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 0     | na                         | na         | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 0     | na                         | na         | na           | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 1     | 1                          | na         | na           | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 1     | 2                          | na         | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 1     | 3                          | na         | na           | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 2     | 1                          | na         | na           | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 2     | 2                          | na         | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 2     | 3                          | na         | na           | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 3     | 1                          | 3          | na           | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 3     | 2                          | 2          | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 3     | 3                          | 1          | na           | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 4     | 1                          | 3          | 2            | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 4     | 2                          | 2          | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 4     | 3                          | 1          | na           | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 5     | 1                          | 2          | 3            | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 5     | 2                          | 2          | na           | 2        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 5     | 3                          | 1          | 4            | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///
    /// And then when those rows with na values are dropped the result would be
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | Index | RW_Mean_MinWin1_MaxWin_1   |   LL_Lag1  |   LL_Lead_1  | Horizon  |
    ///      +=======+============================+============+==============+==========+
    ///      | 4     | 1                          | 3          | 2            | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 5     | 1                          | 2          | 3            | 3        |
    ///      +-------+----------------------------+------------+--------------+----------+
    ///      | 5     | 3                          | 1          | 4            | 1        |
    ///      +-------+----------------------------+------------+--------------+----------+
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ForecastingPivotFeaturizerExtensionClass.PivotForecastingData(TransformsCatalog, string[], string)"/>
    public sealed class ForecastingPivotFeaturizerEstimator : IEstimator<ForecastingPivotTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of columns to pivot", Name = "ColumnsToPivot", ShortName = "cols", SortOrder = 0)]
            public string[] ColumnsToPivot;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the horizon column generated.", Name = "HorizonColumnName", ShortName = "hor", SortOrder = 1)]
            public string HorizonColumnName = "Horizon";
        }

        #endregion

        internal ForecastingPivotFeaturizerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = Contracts.CheckRef(env, nameof(env)).Register("ShortDropEstimator");
            _host.CheckValue(options.ColumnsToPivot, nameof(options.ColumnsToPivot), "ColumnsToPivot should not be null.");
            _host.CheckNonEmpty(options.ColumnsToPivot, nameof(options.ColumnsToPivot), "Need at least one column.");

            _options = options;
        }

        public ForecastingPivotTransformer Fit(IDataView input)
        {
            return new ForecastingPivotTransformer(_host, _options, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // A horizon column is always added to the schema.
            // Additionally, the number of the other new columns added is equal to the sum of Dimension[0] for each input column.

            var columns = inputSchema.ToDictionary(x => x.Name);

            columns[_options.HorizonColumnName] = new SchemaShape.Column(_options.HorizonColumnName, VectorKind.Scalar, NumberDataViewType.UInt32, false);

            // Make sure all ColumnsToPivot are vectors of type double and the same number of columns.
            // Make new columns based on parsing the input column names.
            foreach (var col in _options.ColumnsToPivot)
            {
                // Make sure the column exists
                var found = inputSchema.TryFindColumn(col, out SchemaShape.Column column);
                if (!found)
                    throw new InvalidOperationException($"Pivot column {col} not found in input");

                var colType = column.ItemType;
                if (column.Kind != VectorKind.Vector)
                    throw new InvalidOperationException($"Pivot column {col} must be a vector");

                if (column.ItemType != NumberDataViewType.Double)
                    throw new InvalidOperationException($"Pivot column {col} must be a vector of type double");

                // By this point the input column should have the correct format.
                // Parse the input column annotations to figure out if its from rolling window or lag lead.
                var annotations = column.Annotations;

                // TODO: Fix how we get annotations when the ability is added to get annotations if they are known at this point.
                // Getting the annotation this way is a temporary fix since even though the values are known at this time,
                // SchemaShape does not expose them. To work around this the annotations are stored in the format
                // Annotation=Value. We will just parse this and get the value.
                var columnNamesAnnotationName = annotations.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

                var columnNames = columnNamesAnnotationName.Split('=')[1].Split(',');
                foreach (var name in columnNames)
                {
                    columns[name] = new SchemaShape.Column(name, VectorKind.Scalar, NumberDataViewType.Double, false);
                }
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class ForecastingPivotTransformer : ITransformer
    {
        #region Class data members

        internal const string Summary = "Pivots the input colums and drops any rows with N/A";
        internal const string UserName = "ForecastingPivot";
        internal const string ShortName = "fpivot";
        internal const string LoadName = "ForecastingPivot";
        internal const string LoaderSignature = "ForecastingPivot";

        private readonly IHost _host;
        private readonly ForecastingPivotFeaturizerEstimator.Options _options;

        #endregion

        // Normal constructor.
        internal ForecastingPivotTransformer(IHostEnvironment host, ForecastingPivotFeaturizerEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ForecastingPivotTransformer));
            _options = options;

            var firstCol = input.Schema[_options.ColumnsToPivot[0]];
            var dimensionsToMatch = (firstCol.Type as VectorDataViewType)?.Dimensions;

            foreach (var col in _options.ColumnsToPivot)
            {
                var inputSchema = input.Schema;
                // Make sure the column exists
                var column = inputSchema[col];

                var colType = column.Type as VectorDataViewType;
                if (colType == null)
                    throw new InvalidOperationException($"Pivot column {col} must be a vector");

                if (colType.RawType != typeof(VBuffer<double>))
                    throw new InvalidOperationException($"Pivot column {col} must be a vector of type double");

                if (colType.Dimensions.Length != dimensionsToMatch.Value.Length || colType.Dimensions[1] != dimensionsToMatch.Value[1])
                    throw new InvalidOperationException($"All columns must have the same number of dimensions and the second dimension must be the same size.");
            }
        }

        // Factory method for SignatureLoadModel.
        internal ForecastingPivotTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(ForecastingPivotTransformer));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");

            // *** Binary format ***
            // Horizon column name
            // length of grain column array
            // all column names in grain column array

            var horizonColumnName = ctx.Reader.ReadString();

            var pivotColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < pivotColumns.Length; i++)
                pivotColumns[i] = ctx.Reader.ReadString();

            _options = new ForecastingPivotFeaturizerEstimator.Options
            {
                ColumnsToPivot = pivotColumns,
                HorizonColumnName = horizonColumnName
            };
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new ForecastingPivotTransformer(env, ctx).Transform(input));
        }

        public bool IsRowToRowMapper => false;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumns(inputSchema.AsEnumerable());

            // Will always add a Horizon columns
            schemaBuilder.AddColumn(_options.HorizonColumnName, NumberDataViewType.UInt32);

            foreach (var col in _options.ColumnsToPivot)
            {
                var annotations = inputSchema.GetColumnOrNull(col).Value.Annotations;

                // TODO: Fix how we get annotations when the ability is added to get annotations if they are known at this point.
                // Getting the annotation this way is a temporary fix since even though the values are known at this time,
                // SchemaShape does not expose them. To work around this the annotations are stored in the format
                // Annotation=Value. We will just parse this and get the value.
                var columnNamesAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

                var columnNames = columnNamesAnnotationName.Split('=')[1].Split(',');
                foreach (var name in columnNames)
                {
                    schemaBuilder.AddColumn(name, NumberDataViewType.Double);

                }
            }

            return schemaBuilder.ToSchema();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FPIVOT T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ForecastingPivotTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Horizon column name
            // length of columns array
            // all column names in column array

            ctx.Writer.Write(_options.HorizonColumnName);

            ctx.Writer.Write(_options.ColumnsToPivot.Length);
            foreach (var column in _options.ColumnsToPivot)
                ctx.Writer.Write(column);
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal ForecastingPivotFeaturizerDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new ForecastingPivotFeaturizerDataView(_host, input, _options, this);
        }

        #region IDataView

        internal sealed class ForecastingPivotFeaturizerDataView : IDataTransform
        {
            #region Typed Columns
            private ForecastingPivotTransformer _parent;

            #endregion

            private readonly IHostEnvironment _host;
            private readonly IDataView _source;
            private readonly ForecastingPivotFeaturizerEstimator.Options _options;
            private readonly DataViewSchema _schema;

            internal ForecastingPivotFeaturizerDataView(IHostEnvironment env, IDataView input, ForecastingPivotFeaturizerEstimator.Options options, ForecastingPivotTransformer parent)
            {
                _host = env;
                _source = input;

                _options = options;
                _parent = parent;

                _schema = _parent.GetOutputSchema(input.Schema);
            }

            public bool CanShuffle => false;

            public DataViewSchema Schema => _schema;

            public IDataView Source => _source;

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.AssertValueOrNull(rand);

                var input = _source.GetRowCursorForAllColumns();
                return new Cursor(_host, input, _options, _schema);
            }

            // Can't use parallel cursors so this defaults to calling non-parallel version
            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
                 new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

            // Since we may delete rows we don't know the row count
            public long? GetRowCount() => null;

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            private sealed class Cursor : DataViewRowCursor
            {
                private class PivotColumn
                {
                    private ValueGetter<VBuffer<double>> _getter;
                    private VBuffer<double> _curValues;

                    private readonly int _row;
                    private int _curCol;

                    public PivotColumn(DataViewRowCursor input, string sourceColumnName, int row)
                    {
                        _getter = input.GetGetter<VBuffer<double>>(input.Schema[sourceColumnName]);
                        _curValues = default;

                        var dimensions = ((VectorDataViewType)(input.Schema[sourceColumnName].Type)).Dimensions;
                        ColumnCount = dimensions[1];

                        _row = row;
                        _curCol = 0;
                    }

                    public int ColumnCount { get; private set; }

                    public double GetValueAndSetColumn(int col)
                    {
                        _curCol = col;
                        return GetValue(_curCol);
                    }

                    public void MoveNext()
                    {
                        _getter(ref _curValues);
                    }

                    private double GetValue(int col)
                    {
                        return _curValues.GetItemOrDefault((_row * ColumnCount) + col);
                    }

                    public double GetStoredValue()
                    {
                        return GetValue(_curCol);
                    }
                }

                private readonly IChannelProvider _ch;
                private DataViewRowCursor _input;
                private long _position;
                private bool _isGood;
                private readonly DataViewSchema _schema;
                private readonly ForecastingPivotFeaturizerEstimator.Options _options;

                private Dictionary<string, PivotColumn> _pivotColumns;
                private readonly UInt32 _maxCols;
                private int _currentCol;
                private UInt32 _currentHorizon;

                public Cursor(IChannelProvider provider, DataViewRowCursor input, ForecastingPivotFeaturizerEstimator.Options options, DataViewSchema schema)
                {
                    _ch = provider;
                    _ch.CheckValue(input, nameof(input));

                    // Start isGood at true. This is not exposed outside of the class.
                    _isGood = true;
                    _input = input;
                    _position = -1;
                    _schema = schema;
                    _options = options;

                    _pivotColumns = new Dictionary<string, PivotColumn>();
                    _currentCol = -1;

                    foreach (var col in _options.ColumnsToPivot)
                    {
                        var sourceCol = input.Schema[col];
                        var annotations = sourceCol.Annotations;

                        // Getting the annotation this way is a temporary fix since even though the values are known at this time,
                        // SchemaShape does not expose them. To work around this the annotations are stored in the format
                        // Annotation=Value. We will just parse this and get the value.
                        var columnNamesAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

                        var columnNames = columnNamesAnnotationName.Split('=')[1].Split(',');

                        for (int i = 0; i < columnNames.Length; i++)
                        {
                            _pivotColumns[columnNames[i]] = new PivotColumn(input, col, i);
                        }
                    }

                    // All columns should have the same amount of slots in the vector
                    _maxCols = (UInt32)_pivotColumns.First().Value.ColumnCount;

                    // The horizon starts with the _maxCols + 1 amount and decreases until 1, when it is reset back to this value.
                    // The reason it is + 1 is because maxCols is 0 to N, where horizon is 1 to N + 1.
                    _currentHorizon = _maxCols + 1;
                }

                public sealed override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                           (ref DataViewRowId val) =>
                           {
                               _ch.Check(_isGood, RowCursorUtils.FetchValueStateError);
                               val = new DataViewRowId((ulong)Position, 0);
                           };
                }

                public sealed override DataViewSchema Schema => _schema;

                /// <summary>
                /// Since rows will be dropped
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column) => true;

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    _ch.Check(IsColumnActive(column));

                    var thisCol = _schema[column.Name];

                    if ((_pivotColumns.Keys.Contains(column.Name) && thisCol.Name == column.Name && thisCol.Type == column.Type) || column.Name == _options.HorizonColumnName)
                    {
                        if (column.Name == _options.HorizonColumnName)
                            return MakeHorizonGetter() as ValueGetter<TValue>;
                        else
                            return MakePivotGetter(column) as ValueGetter<TValue>;
                    }
                    else
                    {
                        return _input.GetGetter<TValue>(column);
                    }
                }

                public override bool MoveNext()
                {
                    bool exitLoop = false;
                    while (_isGood && !exitLoop)
                    {
                        // If we haven't done anything yet or we are at the end of a row, advance our source pointer.
                        // We also need to reset _curHorizon to its max value here.
                        if (_currentCol == _maxCols || _currentCol == -1)
                        {
                            // Advance source coursor and break if no more data.
                            _isGood = _input.MoveNext();
                            if (!_isGood)
                                break;

                            _currentHorizon = _maxCols + 1;
                            _currentCol = 0;
                            foreach (var column in _pivotColumns.Values)
                            {
                                column.MoveNext();
                            }
                        }

                        for (int col = _currentCol; col < _maxCols; col++)
                        {
                            var nanFound = false;

                            foreach (var column in _pivotColumns.Values)
                            {
                                if (double.IsNaN(column.GetValueAndSetColumn(col)))
                                {
                                    nanFound = true;
                                    break;
                                }
                            }

                            // Break from loop because we now have valid values
                            // Update the _currentCol so we start from the correct place next time.
                            // Decrease currentHorizon by 1.
                            // Update our current position.
                            _currentHorizon -= 1;
                            _currentCol = col + 1;
                            if (!nanFound)
                            {
                                exitLoop = true;
                                _position++;
                                break;
                            }
                        }
                    }

                    return _isGood;
                }

                public sealed override long Position => _position;

                public sealed override long Batch => _input.Batch;

                private Delegate MakePivotGetter(DataViewSchema.Column column)
                {
                    var pivotColumn = _pivotColumns[column.Name];
                    ValueGetter<double> result = (ref double dst) =>
                    {
                        dst = pivotColumn.GetStoredValue();
                    };

                    return result;
                }

                private Delegate MakeHorizonGetter()
                {
                    ValueGetter<UInt32> result = (ref UInt32 dst) =>
                    {
                        dst = _currentHorizon;
                    };

                    return result;
                }
            }
        }

        #endregion
    }

    internal static class ForecastingPivotTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ForecastingPivot",
            Desc = ForecastingPivotTransformer.Summary,
            UserName = ForecastingPivotTransformer.UserName,
            ShortName = ForecastingPivotTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ShortDrop(IHostEnvironment env, ForecastingPivotFeaturizerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ForecastingPivotTransformer.ShortName, input);
            var xf = new ForecastingPivotFeaturizerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
