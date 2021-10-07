// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ExpressionTransformer.Summary, typeof(IDataTransform), typeof(ExpressionTransformer), typeof(ExpressionTransformer.Options), typeof(SignatureDataTransform),
    "Expression Transform", "Expression", "ExpressionTransform", ExpressionTransformer.LoaderSignature, "Expr")]

[assembly: LoadableClass(ExpressionTransformer.Summary, typeof(IDataTransform), typeof(ExpressionTransformer), null, typeof(SignatureLoadDataTransform),
    "Expression Transform", ExpressionTransformer.LoaderSignature)]

[assembly: LoadableClass(ExpressionTransformer.Summary, typeof(ExpressionTransformer), null, typeof(SignatureLoadModel),
    "Expression Transform", ExpressionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ExpressionTransformer), null, typeof(SignatureLoadRowMapper),
    "Expression Transform", ExpressionTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This estimator applies a user provided expression (specified as a string) to input column values to produce new output column values.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | float, double, int, long, bool or text.  |
    /// | Output column data type | Can be float, double, int, long, bool or text, depending on the expression. |
    ///
    /// The resulting [ExpressionTransformer](xref:Microsoft.ML.Transforms.ExpressionTransformer) creates a new column,
    /// named as specified in the output column name parameters, where the expression is applied to the input values.
    /// At most one of the input columns can be of type [VectorDataViewType](xref:Microsoft.ML.Data.VectorDataViewType), and when the input contains a vector column, the expression
    /// is computed independently on each element of the vector, to create a vector output with the same length as that input.
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/expression-estimator.md)]
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="ExpressionCatalog.Expression(TransformsCatalog, string, string, string[])"/>
    public sealed class ExpressionEstimator : IEstimator<ExpressionTransformer>
    {
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnNames"/>.</summary>
            public readonly string Name;
            public readonly string[] InputColumnNames;
            public string Expression;

            public ColumnOptions(string name, string[] inputColumnNames, string expression)
            {
                Name = name;
                InputColumnNames = inputColumnNames;
                Expression = expression;
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        internal ExpressionEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ExpressionEstimator));
            _host.CheckNonEmpty(columns, nameof(columns));
            _host.Check(columns.All(col => !string.IsNullOrWhiteSpace(col.Expression)));
            _host.Check(columns.All(col => !string.IsNullOrWhiteSpace(col.Name)));
            _host.Check(columns.All(col => Utils.Size(col.InputColumnNames) > 0));
            _host.CheckParam(columns.All(col => Utils.Size(col.InputColumnNames) <= 5), nameof(ColumnOptions.InputColumnNames), "maximum number of inputs exceeded");

            _columns = columns;
        }

        internal static LambdaNode ParseAndBindLambda(IHostEnvironment env, string expression, int ivec, DataViewType[] inputTypes, out int[] perm)
        {

            perm = Utils.GetIdentityPermutation(inputTypes.Length);
            if (ivec >= 0)
            {
                if (ivec > 0)
                {
                    perm[0] = ivec;
                    for (int i = 1; i <= ivec; i++)
                        perm[i] = i - 1;
                }
            }
            CharCursor chars = new CharCursor(expression);

            var node = LambdaParser.Parse(out List<Error> errors, out List<int> lineMap, chars, perm, inputTypes);
            if (Utils.Size(errors) > 0)
                throw env.ExceptParam(nameof(expression), $"parsing failed: {errors[0].GetMessage()}");

            using (var ch = env.Start("LabmdaBinder.Run"))
                LambdaBinder.Run(env, ref errors, node, msg => ch.Error(msg));

            if (Utils.Size(errors) > 0)
                throw env.ExceptParam(nameof(expression), $"binding failed: {errors[0].GetMessage()}");
            return node;
        }

        private static int FindVectorInputColumn(IHostEnvironment env, IReadOnlyList<string> inputColumnNames, SchemaShape inputSchema, DataViewType[] inputTypes)
        {
            int ivec = -1;
            for (int isrc = 0; isrc < inputColumnNames.Count; isrc++)
            {
                if (!inputSchema.TryFindColumn(inputColumnNames[isrc], out var col))
                    throw env.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnNames[isrc]);

                if (col.Kind != SchemaShape.Column.VectorKind.Scalar)
                {
                    if (ivec >= 0)
                        throw env.ExceptUserArg(nameof(inputColumnNames), "Can have at most one vector-valued source column");
                    ivec = isrc;
                }
                inputTypes[isrc] = col.ItemType;
            }

            return ivec;
        }

        private static int FindVectorInputColumn(IHostEnvironment env, IReadOnlyList<string> inputColumnNames, DataViewSchema inputSchema, DataViewType[] inputTypes)
        {
            int ivec = -1;
            for (int isrc = 0; isrc < inputColumnNames.Count; isrc++)
            {
                var col = inputSchema.GetColumnOrNull(inputColumnNames[isrc]);
                if (col == null)
                    throw env.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnNames[isrc]);

                if (col.Value.Type is VectorDataViewType)
                {
                    if (ivec >= 0)
                        throw env.ExceptUserArg(nameof(inputColumnNames), "Can have at most one vector-valued source column");
                    ivec = isrc;
                }
                inputTypes[isrc] = col.Value.Type.GetItemType();
            }

            return ivec;
        }

        public ExpressionTransformer Fit(IDataView input)
        {
            var columns = new ExpressionTransformer.ColumnInfo[_columns.Length];
            for (int i = 0; i < _columns.Length; i++)
            {
                // Make sure there is at most one vector valued source column.
                var inputTypes = new DataViewType[_columns[i].InputColumnNames.Length];
                var ivec = FindVectorInputColumn(_host, _columns[i].InputColumnNames, input.Schema, inputTypes);
                var node = ParseAndBindLambda(_host, _columns[i].Expression, ivec, inputTypes, out var perm);
                columns[i] = new ExpressionTransformer.ColumnInfo(_host, _columns[i].InputColumnNames, inputTypes, _columns[i].Expression, _columns[i].Name, ivec, node, perm);
            }
            return new ExpressionTransformer(_host, columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var columnDictionary = inputSchema.ToDictionary(x => x.Name);
            for (int i = 0; i < _columns.Length; i++)
            {
                for (int j = 0; j < _columns[i].InputColumnNames.Length; j++)
                {
                    if (!inputSchema.TryFindColumn(_columns[i].InputColumnNames[j], out var inputCol))
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _columns[i].InputColumnNames[j]);
                }

                // Make sure there is at most one vector valued source column.
                var inputTypes = new DataViewType[_columns[i].InputColumnNames.Length];
                var ivec = FindVectorInputColumn(_host, _columns[i].InputColumnNames, inputSchema, inputTypes);
                var node = ParseAndBindLambda(_host, _columns[i].Expression, ivec, inputTypes, out var perm);

                var typeRes = node.ResultType;
                _host.Assert(typeRes is PrimitiveDataViewType);

                // If one of the input columns is a vector column, we pass through the slot names metadata.
                SchemaShape.Column.VectorKind outputVectorKind;
                var metadata = new List<SchemaShape.Column>();
                if (ivec == -1)
                    outputVectorKind = SchemaShape.Column.VectorKind.Scalar;
                else
                {
                    inputSchema.TryFindColumn(_columns[i].InputColumnNames[ivec], out var vectorCol);
                    outputVectorKind = vectorCol.Kind;
                    if (vectorCol.HasSlotNames())
                    {
                        var b = vectorCol.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotNames);
                        _host.Assert(b);
                        metadata.Add(slotNames);
                    }
                }
                var outputSchemaShapeColumn = new SchemaShape.Column(_columns[i].Name, outputVectorKind, typeRes, false, new SchemaShape(metadata));
                columnDictionary[_columns[i].Name] = outputSchemaShapeColumn;
            }
            return new SchemaShape(columnDictionary.Values);
        }
    }

    public sealed class ExpressionTransformer : RowToRowTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, ShortName = "expr", SortOrder = 2, HelpText = "Lambda expression which will be applied.")]
            public string Expression = "(x) : x";
        }

        internal sealed class Column : ManyToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "expr", SortOrder = 2, HelpText = "Lambda expression which will be applied.")]
            public string Expression;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Expression != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        internal sealed class ColumnInfo
        {
            public readonly string OutputColumnName;
            public readonly string[] InputColumnNames;
            public readonly PrimitiveDataViewType OutputColumnItemType;
            public readonly int VectorInputColumn;
            public readonly Delegate Del;
            public readonly int[] Perm;
            public readonly string Expression;
            public readonly InternalDataKind[] InputKinds;

            public ColumnInfo(IExceptionContext ectx, string[] inputColumnNames, DataViewType[] inputTypes, string expression, string outputColumnName, int vectorInputColumn, LambdaNode node, int[] perm)
            {
                ectx.AssertNonEmpty(inputTypes);
                ectx.Assert(Utils.Size(inputTypes) == Utils.Size(inputColumnNames));
                ectx.AssertNonWhiteSpace(expression);
                ectx.AssertNonWhiteSpace(outputColumnName);
                ectx.AssertValue(node);

                InputColumnNames = inputColumnNames;
                OutputColumnName = outputColumnName;
                OutputColumnItemType = node.ResultType as PrimitiveDataViewType;
                ectx.AssertValue(OutputColumnItemType);
                VectorInputColumn = vectorInputColumn;
                Perm = perm;
                Expression = expression;

                InputKinds = new InternalDataKind[inputTypes.Length];
                for (int i = 0; i < inputTypes.Length; i++)
                    InputKinds[i] = inputTypes[i].GetRawKind();

                Del = LambdaCompiler.Compile(out var errors, node);
                if (Utils.Size(errors) > 0)
                    throw ectx.Except($"generating code failed: {errors[0].GetMessage()}");
            }
        }

        private readonly ColumnInfo[] _columns;

        internal const string Summary = "Executes a given lambda expression on input column values to produce an output column value. " +
                                 "Here are some examples to demonstrate the syntax: " +
                                 "1) expr={x : x / 256} divides the input value by 256 " +
                                 "2) expr={x : x ?? -1} replaces missing values with -1. " +
                                 "3) expr={(x, y) : log(x / y)} computes log odds. " +
                                 "For more details see the documentation.";

        public const string LoaderSignature = "ExprTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXPRTRNF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ExpressionTransformer).Assembly.FullName);
        }

        internal ExpressionTransformer(IHostEnvironment env, ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ExpressionTransformer)))
        {
            Host.AssertNonEmpty(columns);
            _columns = columns;
        }

        // Factory method corresponding to SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckNonEmpty(options.Column, nameof(options.Column));

            var columns = new ExpressionEstimator.ColumnOptions[options.Column.Length];
            for (int i = 0; i < options.Column.Length; i++)
            {
                columns[i] = new ExpressionEstimator.ColumnOptions(options.Column[i].Name,
                    Utils.Size(options.Column[i].Source) == 0 ? new[] { options.Column[i].Name } : options.Column[i].Source,
                    options.Column[i].Expression ?? options.Expression);
            }

            return new ExpressionEstimator(env, columns).Fit(input).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static ExpressionTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: number of output columns
            // for each output column:
            //   int: number of inputs
            //   foreach input
            //     int: Id of the input column name
            //   int: Id of the expression
            //   int: Id of the output column name
            //   int: the index of the vector input (or -1)
            //   int[]: The data kinds of the input columns

            var columnCount = ctx.Reader.ReadInt32();
            env.CheckDecode(columnCount > 0);

            var columns = new ColumnInfo[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                var inputSize = ctx.Reader.ReadInt32();
                env.CheckDecode(inputSize >= 0);
                var inputColumnNames = new string[inputSize];
                for (int j = 0; j < inputSize; j++)
                    inputColumnNames[j] = ctx.LoadNonEmptyString();
                var expression = ctx.LoadNonEmptyString();
                var outputColumnName = ctx.LoadNonEmptyString();
                var vectorInputColumn = ctx.Reader.ReadInt32();
                env.CheckDecode(vectorInputColumn >= -1);

                var inputTypes = new DataViewType[inputSize];
                for (int j = 0; j < inputSize; j++)
                {
                    var dataKindIndex = ctx.Reader.ReadInt32();
                    var kind = InternalDataKindExtensions.FromIndex(dataKindIndex);
                    inputTypes[j] = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
                }
                var node = ExpressionEstimator.ParseAndBindLambda(env, expression, vectorInputColumn, inputTypes, out var perm);
                columns[i] = new ColumnInfo(env, inputColumnNames, inputTypes, expression, outputColumnName, vectorInputColumn, node, perm);
            }
            return new ExpressionTransformer(env, columns);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of output columns
            // for each output column:
            //   int: number of inputs
            //   foreach input
            //     int: Id of the input column name
            //   int: Id of the expression
            //   int: Id of the output column name
            //   int: the index of the vector input (or -1)
            //   int[]: The data kinds of the input columns

            ctx.Writer.Write(_columns.Length);

            for (int i = 0; i < _columns.Length; i++)
            {
                var inputColumnNames = _columns[i].InputColumnNames;
                ctx.Writer.Write(inputColumnNames.Length);
                for (int j = 0; j < inputColumnNames.Length; j++)
                    ctx.SaveNonEmptyString(inputColumnNames[j]);

                ctx.SaveNonEmptyString(_columns[i].Expression);
                ctx.SaveNonEmptyString(_columns[i].OutputColumnName);
                Host.Assert(_columns[i].VectorInputColumn >= -1);
                ctx.Writer.Write(_columns[i].VectorInputColumn);
                for (int j = 0; j < _columns[i].InputKinds.Length; j++)
                    ctx.Writer.Write(_columns[i].InputKinds[j].ToIndex());
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
        {
            return new Mapper(this, schema);
        }

        private Delegate MakeGetter(IExceptionContext ectx, DataViewRow input, int iinfo)
        {
            Func<IExceptionContext, DataViewRow, DataViewSchema.Column[], Delegate, ValueGetter<int>> d;
            var types = _columns[iinfo].Del.GetType().GetGenericArguments();
            switch (types.Length - 1)
            {
                case 1:
                    d = GetGetter<int, int>;
                    break;
                case 2:
                    d = GetGetter<int, int, int>;
                    break;
                case 3:
                    d = GetGetter<int, int, int, int>;
                    break;
                case 4:
                    d = GetGetter<int, int, int, int, int>;
                    break;
                case 5:
                    d = GetGetter<int, int, int, int, int, int>;
                    break;
                default:
                    ectx.Assert(false, "Unsupported src cardinality");
                    throw ectx.ExceptNotSupp();
            }

            var inputColumnNames = _columns[iinfo].InputColumnNames;
            var inputColumns = new DataViewSchema.Column[inputColumnNames.Length];
            for (int i = 0; i < inputColumnNames.Length; i++)
                inputColumns[i] = input.Schema[inputColumnNames[i]];

            var meth = d.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(types);
            return (Delegate)meth.Invoke(this, new object[] { ectx, input, inputColumns, _columns[iinfo].Del });
        }

        private ValueGetter<TDst> GetGetter<T0, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, Delegate del)
        {
            ectx.Assert(inputColumns.Length == 1);

            var fn = (Func<T0, TDst>)del;
            var getSrc0 = input.GetGetter<T0>(inputColumns[0]);
            var src0 = default(T0);
            return
                (ref TDst dst) =>
                {
                    getSrc0(ref src0);
                    dst = fn(src0);
                };
        }

        private ValueGetter<TDst> GetGetter<T0, T1, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, Delegate del)
        {
            ectx.Assert(inputColumns.Length == 2);

            var fn = (Func<T0, T1, TDst>)del;
            var getSrc0 = input.GetGetter<T0>(inputColumns[0]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[1]);
            var src0 = default(T0);
            var src1 = default(T1);
            return
                (ref TDst dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    dst = fn(src0, src1);
                };
        }

        private ValueGetter<TDst> GetGetter<T0, T1, T2, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, Delegate del)
        {
            ectx.Assert(inputColumns.Length == 3);

            var fn = (Func<T0, T1, T2, TDst>)del;
            var getSrc0 = input.GetGetter<T0>(inputColumns[0]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[1]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[2]);
            var src0 = default(T0);
            var src1 = default(T1);
            var src2 = default(T2);
            return
                (ref TDst dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    dst = fn(src0, src1, src2);
                };
        }

        private ValueGetter<TDst> GetGetter<T0, T1, T2, T3, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, Delegate del)
        {
            ectx.Assert(inputColumns.Length == 4);

            var fn = (Func<T0, T1, T2, T3, TDst>)del;
            var getSrc0 = input.GetGetter<T0>(inputColumns[0]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[1]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[2]);
            var getSrc3 = input.GetGetter<T3>(inputColumns[3]);
            var src0 = default(T0);
            var src1 = default(T1);
            var src2 = default(T2);
            var src3 = default(T3);
            return
                (ref TDst dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    getSrc3(ref src3);
                    dst = fn(src0, src1, src2, src3);
                };
        }

        private ValueGetter<TDst> GetGetter<T0, T1, T2, T3, T4, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, Delegate del)
        {
            ectx.Assert(inputColumns.Length == 5);

            var fn = (Func<T0, T1, T2, T3, T4, TDst>)del;
            var getSrc0 = input.GetGetter<T0>(inputColumns[0]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[1]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[2]);
            var getSrc3 = input.GetGetter<T3>(inputColumns[3]);
            var getSrc4 = input.GetGetter<T4>(inputColumns[4]);
            var src0 = default(T0);
            var src1 = default(T1);
            var src2 = default(T2);
            var src3 = default(T3);
            var src4 = default(T4);
            return
                (ref TDst dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    getSrc3(ref src3);
                    getSrc4(ref src4);
                    dst = fn(src0, src1, src2, src3, src4);
                };
        }

        private Delegate MakeGetterVec(IExceptionContext ectx, DataViewRow input, int iinfo)
        {
            ectx.Assert(_columns[iinfo].VectorInputColumn >= 0);

            Func<IExceptionContext, DataViewRow, DataViewSchema.Column[], int[], Delegate, DataViewType, ValueGetter<VBuffer<int>>> d;
            var types = _columns[iinfo].Del.GetType().GetGenericArguments();
            switch (types.Length - 1)
            {
                case 1:
                    d = GetGetterVec<int, int>;
                    break;
                case 2:
                    d = GetGetterVec<int, int, int>;
                    break;
                case 3:
                    d = GetGetterVec<int, int, int, int>;
                    break;
                case 4:
                    d = GetGetterVec<int, int, int, int, int>;
                    break;
                case 5:
                    d = GetGetterVec<int, int, int, int, int, int>;
                    break;
                default:
                    ectx.Assert(false, "Unsupported src cardinality");
                    throw ectx.ExceptNotSupp();
            }

            var inputColumnNames = _columns[iinfo].InputColumnNames;
            var inputColumns = new DataViewSchema.Column[inputColumnNames.Length];
            for (int i = 0; i < inputColumnNames.Length; i++)
                inputColumns[i] = input.Schema[inputColumnNames[i]];

            var meth = d.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(types);
            return (Delegate)meth.Invoke(this, new object[] { ectx, input, inputColumns, _columns[iinfo].Perm, _columns[iinfo].Del, _columns[iinfo].OutputColumnItemType });
        }

        private ValueGetter<VBuffer<TDst>> GetGetterVec<T0, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, int[] perm, Delegate del, DataViewType outputColumnItemType)
        {
            ectx.Assert(inputColumns.Length == 1);
            ectx.Assert(perm.Length == 1);
            ectx.Assert(perm[0] == 0);

            var fn = (Func<T0, TDst>)del;
            var getSrc0 = input.GetGetter<VBuffer<T0>>(inputColumns[0]);
            var src0 = default(VBuffer<T0>);

            var dstDef = fn(default(T0));
            var isDef = Conversions.DefaultInstance.GetIsDefaultPredicate<TDst>(outputColumnItemType);
            if (isDef(in dstDef))
            {
                // Sparsity is preserved.
                return
                    (ref VBuffer<TDst> dst) =>
                    {
                        getSrc0(ref src0);
                        int count = src0.GetValues().Length;

                        var editor = VBufferEditor.Create(ref dst, src0.Length, count);
                        for (int i = 0; i < count; i++)
                            editor.Values[i] = fn(src0.GetValues()[i]);

                        if (!src0.IsDense)
                            src0.GetIndices().CopyTo(editor.Indices);
                        dst = editor.Commit();
                    };
            }
            else
            {
                // Densifies - default(T0) maps to dstDef, which is not default(TDst).
                return
                    (ref VBuffer<TDst> dst) =>
                    {
                        getSrc0(ref src0);
                        int len = src0.Length;
                        var editor = VBufferEditor.Create(ref dst, len);
                        if (src0.IsDense)
                        {
                            for (int i = 0; i < len; i++)
                                editor.Values[i] = fn(src0.GetValues()[i]);
                        }
                        else
                        {
                            int count = src0.GetValues().Length;
                            for (int i = 0, ii = 0; i < len; i++)
                            {
                                ectx.Assert(0 <= ii && ii <= count);
                                ectx.Assert(ii == count || src0.GetIndices()[ii] >= i);
                                if (ii < count && src0.GetIndices()[ii] == i)
                                {
                                    editor.Values[i] = fn(src0.GetValues()[ii]);
                                    ii++;
                                }
                                else
                                    editor.Values[i] = dstDef;
                            }
                        }
                        dst = editor.Commit();
                    };
            }
        }

        private ValueGetter<VBuffer<TDst>> GetGetterVec<T0, T1, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, int[] perm, Delegate del, DataViewType outputColumnItemType)
        {
            ectx.Assert(inputColumns.Length == 2);
            ectx.Assert(perm.Length == 2);

            var isDef = Conversions.DefaultInstance.GetIsDefaultPredicate<TDst>(outputColumnItemType);
            var fn = (Func<T0, T1, TDst>)del;
            var getSrc0 = input.GetGetter<VBuffer<T0>>(inputColumns[perm[0]]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[perm[1]]);
            var src0 = default(VBuffer<T0>);
            var src1 = default(T1);

            return
                (ref VBuffer<TDst> dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    if (src0.IsDense)
                    {
                        int len = src0.Length;
                        var editor = VBufferEditor.Create(ref dst, len);
                        for (int i = 0; i < len; i++)
                            editor.Values[i] = fn(src0.GetValues()[i], src1);
                        dst = editor.Commit();
                    }
                    else
                    {
                        int len = src0.Length;
                        int count = src0.GetValues().Length;
                        var dstDef = fn(default(T0), src1);
                        if (isDef(in dstDef))
                        {
                            var editor = VBufferEditor.Create(ref dst, len, count);
                            for (int i = 0; i < count; i++)
                                editor.Values[i] = fn(src0.GetValues()[i], src1);
                            src0.GetIndices().CopyTo(editor.Indices);
                            dst = editor.Commit();
                        }
                        else
                        {
                            // Densifies - default(T0) maps to dstDef, which is not default(TDst).
                            var editor = VBufferEditor.Create(ref dst, len);
                            for (int i = 0, ii = 0; i < len; i++)
                            {
                                ectx.Assert(0 <= ii && ii <= count);
                                ectx.Assert(ii == count || src0.GetIndices()[ii] >= i);
                                if (ii < count && src0.GetIndices()[ii] == i)
                                {
                                    editor.Values[i] = fn(src0.GetValues()[ii], src1);
                                    ii++;
                                }
                                else
                                    editor.Values[i] = dstDef;
                            }
                            dst = editor.Commit();
                        }
                    }
                };
        }

        private ValueGetter<VBuffer<TDst>> GetGetterVec<T0, T1, T2, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, int[] perm, Delegate del, DataViewType outputColumnItemType)
        {
            ectx.Assert(inputColumns.Length == 3);
            ectx.Assert(perm.Length == 3);

            var isDef = Conversions.DefaultInstance.GetIsDefaultPredicate<TDst>(outputColumnItemType);
            var fn = (Func<T0, T1, T2, TDst>)del;
            var getSrc0 = input.GetGetter<VBuffer<T0>>(inputColumns[perm[0]]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[perm[1]]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[perm[2]]);
            var src0 = default(VBuffer<T0>);
            var src1 = default(T1);
            var src2 = default(T2);

            return
                (ref VBuffer<TDst> dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    if (src0.IsDense)
                    {
                        int len = src0.Length;
                        var editor = VBufferEditor.Create(ref dst, len);
                        for (int i = 0; i < len; i++)
                            editor.Values[i] = fn(src0.GetValues()[i], src1, src2);
                        dst = editor.Commit();
                    }
                    else
                    {
                        int len = src0.Length;
                        int count = src0.GetValues().Length;
                        var dstDef = fn(default(T0), src1, src2);
                        if (isDef(in dstDef))
                        {
                            var editor = VBufferEditor.Create(ref dst, len, count);
                            for (int i = 0; i < count; i++)
                                editor.Values[i] = fn(src0.GetValues()[i], src1, src2);
                            src0.GetIndices().CopyTo(editor.Indices);
                            dst = editor.Commit();
                        }
                        else
                        {
                            // Densifies - default(T0) maps to dstDef, which is not default(TDst).
                            var editor = VBufferEditor.Create(ref dst, len);
                            for (int i = 0, ii = 0; i < len; i++)
                            {
                                ectx.Assert(0 <= ii && ii <= count);
                                ectx.Assert(ii == count || src0.GetIndices()[ii] >= i);
                                if (ii < count && src0.GetIndices()[ii] == i)
                                {
                                    editor.Values[i] = fn(src0.GetValues()[ii], src1, src2);
                                    ii++;
                                }
                                else
                                    editor.Values[i] = dstDef;
                            }
                            dst = editor.Commit();
                        }
                    }
                };
        }

        private ValueGetter<VBuffer<TDst>> GetGetterVec<T0, T1, T2, T3, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, int[] perm, Delegate del, DataViewType outputColumnItemType)
        {
            ectx.Assert(inputColumns.Length == 4);
            ectx.Assert(perm.Length == 4);

            var isDef = Conversions.DefaultInstance.GetIsDefaultPredicate<TDst>(outputColumnItemType);
            var fn = (Func<T0, T1, T2, T3, TDst>)del;
            var getSrc0 = input.GetGetter<VBuffer<T0>>(inputColumns[perm[0]]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[perm[1]]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[perm[2]]);
            var getSrc3 = input.GetGetter<T3>(inputColumns[perm[3]]);
            var src0 = default(VBuffer<T0>);
            var src1 = default(T1);
            var src2 = default(T2);
            var src3 = default(T3);

            return
                (ref VBuffer<TDst> dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    getSrc3(ref src3);
                    if (src0.IsDense)
                    {
                        int len = src0.Length;
                        var editor = VBufferEditor.Create(ref dst, len);
                        for (int i = 0; i < len; i++)
                            editor.Values[i] = fn(src0.GetValues()[i], src1, src2, src3);
                        dst = editor.Commit();
                    }
                    else
                    {
                        int len = src0.Length;
                        int count = src0.GetValues().Length;
                        var dstDef = fn(default(T0), src1, src2, src3);
                        if (isDef(in dstDef))
                        {
                            var editor = VBufferEditor.Create(ref dst, len, count);
                            for (int i = 0; i < count; i++)
                                editor.Values[i] = fn(src0.GetValues()[i], src1, src2, src3);
                            src0.GetIndices().CopyTo(editor.Indices);
                            dst = editor.Commit();
                        }
                        else
                        {
                            // Densifies - default(T0) maps to dstDef, which is not default(TDst).
                            var editor = VBufferEditor.Create(ref dst, len);
                            for (int i = 0, ii = 0; i < len; i++)
                            {
                                ectx.Assert(0 <= ii && ii <= count);
                                ectx.Assert(ii == count || src0.GetIndices()[ii] >= i);
                                if (ii < count && src0.GetIndices()[ii] == i)
                                {
                                    editor.Values[i] = fn(src0.GetValues()[ii], src1, src2, src3);
                                    ii++;
                                }
                                else
                                    editor.Values[i] = dstDef;
                            }
                            dst = editor.Commit();
                        }
                    }
                };
        }

        private ValueGetter<VBuffer<TDst>> GetGetterVec<T0, T1, T2, T3, T4, TDst>(IExceptionContext ectx, DataViewRow input, DataViewSchema.Column[] inputColumns, int[] perm, Delegate del, DataViewType outputColumnItemType)
        {
            ectx.Assert(inputColumns.Length == 5);
            ectx.Assert(perm.Length == 5);

            var isDef = Conversions.DefaultInstance.GetIsDefaultPredicate<TDst>(outputColumnItemType);
            var fn = (Func<T0, T1, T2, T3, T4, TDst>)del;
            var getSrc0 = input.GetGetter<VBuffer<T0>>(inputColumns[perm[0]]);
            var getSrc1 = input.GetGetter<T1>(inputColumns[perm[1]]);
            var getSrc2 = input.GetGetter<T2>(inputColumns[perm[2]]);
            var getSrc3 = input.GetGetter<T3>(inputColumns[perm[3]]);
            var getSrc4 = input.GetGetter<T4>(inputColumns[perm[4]]);
            var src0 = default(VBuffer<T0>);
            var src1 = default(T1);
            var src2 = default(T2);
            var src3 = default(T3);
            var src4 = default(T4);

            return
                (ref VBuffer<TDst> dst) =>
                {
                    getSrc0(ref src0);
                    getSrc1(ref src1);
                    getSrc2(ref src2);
                    getSrc3(ref src3);
                    getSrc4(ref src4);
                    if (src0.IsDense)
                    {
                        int len = src0.Length;
                        var editor = VBufferEditor.Create(ref dst, len);
                        for (int i = 0; i < len; i++)
                            editor.Values[i] = fn(src0.GetValues()[i], src1, src2, src3, src4);
                        dst = editor.Commit();
                    }
                    else
                    {
                        int len = src0.Length;
                        int count = src0.GetValues().Length;
                        var dstDef = fn(default(T0), src1, src2, src3, src4);
                        if (isDef(in dstDef))
                        {
                            var editor = VBufferEditor.Create(ref dst, len, count);
                            for (int i = 0; i < count; i++)
                                editor.Values[i] = fn(src0.GetValues()[i], src1, src2, src3, src4);
                            src0.GetIndices().CopyTo(editor.Indices);
                            dst = editor.Commit();
                        }
                        else
                        {
                            // Densifies - default(T0) maps to dstDef, which is not default(TDst).
                            var editor = VBufferEditor.Create(ref dst, len);
                            for (int i = 0, ii = 0; i < len; i++)
                            {
                                ectx.Assert(0 <= ii && ii <= count);
                                ectx.Assert(ii == count || src0.GetIndices()[ii] >= i);
                                if (ii < count && src0.GetIndices()[ii] == i)
                                {
                                    editor.Values[i] = fn(src0.GetValues()[ii], src1, src2, src3, src4);
                                    ii++;
                                }
                                else
                                    editor.Values[i] = dstDef;
                            }
                            dst = editor.Commit();
                        }
                    }
                };
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ExpressionTransformer _parent;
            private readonly int[][] _inputColumnIndices;

            public Mapper(ExpressionTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host, inputSchema, parent)
            {
                _parent = parent;
                _inputColumnIndices = new int[_parent._columns.Length][];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    var inputColumnNames = _parent._columns[i].InputColumnNames;
                    _inputColumnIndices[i] = new int[inputColumnNames.Length];
                    for (int j = 0; j < inputColumnNames.Length; j++)
                    {
                        if (!InputSchema.TryGetColumnIndex(inputColumnNames[j], out _inputColumnIndices[i][j]))
                            throw Host.Except($"Column {inputColumnNames[j]} does not exist in the input schema");
                    }
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var outputColumns = new DataViewSchema.DetachedColumn[_parent._columns.Length];
                for (int i = 0; i < outputColumns.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    DataViewType type;
                    if (_parent._columns[i].VectorInputColumn >= 0)
                    {
                        var vectorColumn = InputSchema[_parent._columns[i].InputColumnNames[_parent._columns[i].VectorInputColumn]];
                        builder.Add(vectorColumn.Annotations, name => name == AnnotationUtils.Kinds.SlotNames);
                        type = new VectorDataViewType(_parent._columns[i].OutputColumnItemType, vectorColumn.Type.GetValueCount());
                    }
                    else
                        type = _parent._columns[i].OutputColumnItemType;
                    outputColumns[i] = new DataViewSchema.DetachedColumn(_parent._columns[i].OutputColumnName, type, builder.ToAnnotations());
                }
                return outputColumns;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                _parent.Host.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                disposer = null;

                if (_parent._columns[iinfo].VectorInputColumn >= 0)
                    return _parent.MakeGetterVec(_parent.Host, input, iinfo);
                return _parent.MakeGetter(_parent.Host, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col =>
                {
                    for (int i = 0; i < _inputColumnIndices.Length; i++)
                    {
                        if (activeOutput(i) && _inputColumnIndices[i].Any(index => index == col))
                            return true;
                    }
                    return false;
                };
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }
}
