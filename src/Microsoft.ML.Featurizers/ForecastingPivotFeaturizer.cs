using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
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
