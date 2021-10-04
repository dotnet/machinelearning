// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(MissingValueIndicatorTransformer.Summary, typeof(IDataTransform), typeof(MissingValueIndicatorTransformer), typeof(MissingValueIndicatorTransformer.Options), typeof(SignatureDataTransform),
    MissingValueIndicatorTransformer.FriendlyName, MissingValueIndicatorTransformer.LoadName, "NAIndicator", MissingValueIndicatorTransformer.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(MissingValueIndicatorTransformer.Summary, typeof(IDataTransform), typeof(MissingValueIndicatorTransformer), null, typeof(SignatureLoadDataTransform),
    MissingValueIndicatorTransformer.FriendlyName, MissingValueIndicatorTransformer.LoadName)]

[assembly: LoadableClass(MissingValueIndicatorTransformer.Summary, typeof(MissingValueIndicatorTransformer), null, typeof(SignatureLoadModel),
    MissingValueIndicatorTransformer.FriendlyName, MissingValueIndicatorTransformer.LoadName)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(MissingValueIndicatorTransformer), null, typeof(SignatureLoadRowMapper),
   MissingValueIndicatorTransformer.FriendlyName, MissingValueIndicatorTransformer.LoadName)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="MissingValueIndicatorEstimator"/>.
    /// </summary>
    public sealed class MissingValueIndicatorTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
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
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal const string LoadName = "NaIndicatorTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAIND TF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName,
                loaderAssemblyName: typeof(MissingValueIndicatorTransformer).Assembly.FullName);
        }

        internal const string Summary = "Create a boolean output column with the same number of slots as the input column, where the output value"
            + " is true if the value in the input column is missing.";
        internal const string FriendlyName = "NA Indicator Transform";
        internal const string ShortName = "NAInd";

        private const string RegistrationName = nameof(MissingValueIndicatorTransformer);

        /// <summary>
        /// The names of the output and input column pairs for the transformation.
        /// </summary>
        internal IReadOnlyList<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueIndicatorTransformer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        internal MissingValueIndicatorTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueIndicatorTransformer)), columns)
        {
        }

        internal MissingValueIndicatorTransformer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueIndicatorTransformer)), GetColumnPairs(options.Columns))
        {
        }

        private MissingValueIndicatorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueIndicatorTransformer)), ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(Column[] columns)
            => columns.Select(c => (c.Name, c.Source ?? c.Name)).ToArray();

        // Factory method for SignatureLoadModel
        internal static MissingValueIndicatorTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());

            return new MissingValueIndicatorTransformer(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
            => new MissingValueIndicatorTransformer(env, options).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        /// <summary>
        /// Saves the transform.
        /// </summary>
        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private static readonly FuncStaticMethodInfo1<DataViewType, Delegate> _getIsNADelegateMethodInfo
                = new FuncStaticMethodInfo1<DataViewType, Delegate>(GetIsNADelegate<int>);

            private static readonly FuncInstanceMethodInfo1<Mapper, DataViewRow, int, ValueGetter<bool>> _composeGetterOneMethodInfo
                = FuncInstanceMethodInfo1<Mapper, DataViewRow, int, ValueGetter<bool>>.Create(target => target.ComposeGetterOne<int>);

            private static readonly FuncInstanceMethodInfo1<Mapper, DataViewRow, int, ValueGetter<VBuffer<bool>>> _composeGetterVecMethodInfo
                = FuncInstanceMethodInfo1<Mapper, DataViewRow, int, ValueGetter<VBuffer<bool>>>.Create(target => target.ComposeGetterVec<int>);

            private readonly MissingValueIndicatorTransformer _parent;
            private readonly ColInfo[] _infos;

            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string InputColumnName;
                public readonly DataViewType OutputType;
                public readonly DataViewType InputType;
                public readonly Delegate InputIsNA;

                public ColInfo(string name, string inputColumnName, DataViewType inType, DataViewType outType)
                {
                    Name = name;
                    InputColumnName = inputColumnName;
                    InputType = inType;
                    OutputType = outType;
                    InputIsNA = GetIsNADelegate(InputType);
                }
            }

            public Mapper(MissingValueIndicatorTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
            }

            private ColInfo[] CreateInfos(DataViewSchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    var inType = inputSchema[colSrc].Type;
                    DataViewType outType;
                    if (!(inType is VectorDataViewType vectorType))
                        outType = BooleanDataViewType.Instance;
                    else
                        outType = new VectorDataViewType(BooleanDataViewType.Instance, vectorType.Dimensions);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].outputColumnName, _parent.ColumnPairs[i].inputColumnName, inType, outType);
                }
                return infos;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _infos.Length; iinfo++)
                {
                    InputSchema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new DataViewSchema.Annotations.Builder();
                    builder.Add(InputSchema[colIndex].Annotations, x => x == AnnotationUtils.Kinds.SlotNames);
                    ValueGetter<bool> getter = (ref bool dst) =>
                    {
                        dst = true;
                    };
                    builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, getter);
                    result[iinfo] = new DataViewSchema.DetachedColumn(_infos[iinfo].Name, _infos[iinfo].OutputType, builder.ToAnnotations());
                }
                return result;
            }

            /// <summary>
            /// Returns the isNA predicate for the respective type.
            /// </summary>
            private static Delegate GetIsNADelegate(DataViewType type)
            {
                return Utils.MarshalInvoke(_getIsNADelegateMethodInfo, type.GetItemType().RawType, type);
            }

            private static Delegate GetIsNADelegate<T>(DataViewType type)
            {
                return Data.Conversion.Conversions.DefaultInstance.GetIsNAPredicate<T>(type.GetItemType());
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                if (!(_infos[iinfo].InputType is VectorDataViewType))
                    return ComposeGetterOne(input, iinfo);
                return ComposeGetterVec(input, iinfo);
            }

            /// <summary>
            /// Getter generator for single valued inputs.
            /// </summary>
            private ValueGetter<bool> ComposeGetterOne(DataViewRow input, int iinfo)
                => Utils.MarshalInvoke(_composeGetterOneMethodInfo, this, _infos[iinfo].InputType.RawType, input, iinfo);

            private ValueGetter<bool> ComposeGetterOne<T>(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<T>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(T);
                var isNA = (InPredicate<T>)_infos[iinfo].InputIsNA;

                ValueGetter<bool> getter;

                return getter =
                    (ref bool dst) =>
                    {
                        getSrc(ref src);
                        dst = isNA(in src);
                    };
            }

            /// <summary>
            /// Getter generator for vector valued inputs.
            /// </summary>
            private ValueGetter<VBuffer<bool>> ComposeGetterVec(DataViewRow input, int iinfo)
                => Utils.MarshalInvoke(_composeGetterVecMethodInfo, this, _infos[iinfo].InputType.GetItemType().RawType, input, iinfo);

            private ValueGetter<VBuffer<bool>> ComposeGetterVec<T>(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<T>>(input.Schema[ColMapNewToOld[iinfo]]);
                var isNA = (InPredicate<T>)_infos[iinfo].InputIsNA;
                var val = default(T);
                var defaultIsNA = isNA(in val);
                var src = default(VBuffer<T>);
                var indices = new List<int>();

                ValueGetter<VBuffer<bool>> getter;

                return getter =
                    (ref VBuffer<bool> dst) =>
                    {
                        // Sense indicates if the values added to the indices list represent NAs or non-NAs.
                        bool sense;
                        getSrc(ref src);
                        FindNAs(in src, isNA, defaultIsNA, indices, out sense);
                        FillValues(src.Length, ref dst, indices, sense);
                    };
            }

            /// <summary>
            /// Adds all NAs (or non-NAs) to the indices List.  Whether NAs or non-NAs have been added is indicated by the bool sense.
            /// </summary>
            private void FindNAs<T>(in VBuffer<T> src, InPredicate<T> isNA, bool defaultIsNA, List<int> indices, out bool sense)
            {
                Host.AssertValue(isNA);
                Host.AssertValue(indices);

                // Find the indices of all of the NAs.
                indices.Clear();
                var srcValues = src.GetValues();
                var srcCount = srcValues.Length;
                if (src.IsDense)
                {
                    for (int i = 0; i < srcCount; i++)
                    {
                        if (isNA(in srcValues[i]))
                            indices.Add(i);
                    }
                    sense = true;
                }
                else if (!defaultIsNA)
                {
                    var srcIndices = src.GetIndices();
                    for (int ii = 0; ii < srcCount; ii++)
                    {
                        if (isNA(in srcValues[ii]))
                            indices.Add(srcIndices[ii]);
                    }
                    sense = true;
                }
                else
                {
                    // Note that this adds non-NAs to indices -- this is indicated by sense being false.
                    var srcIndices = src.GetIndices();
                    for (int ii = 0; ii < srcCount; ii++)
                    {
                        if (!isNA(in srcValues[ii]))
                            indices.Add(srcIndices[ii]);
                    }
                    sense = false;
                }
            }

            /// <summary>
            ///  Fills indicator values for vectors.  The indices is a list that either holds all of the NAs or all
            ///  of the non-NAs, indicated by sense being true or false respectively.
            /// </summary>
            private void FillValues(int srcLength, ref VBuffer<bool> dst, List<int> indices, bool sense)
            {
                if (indices.Count == 0)
                {
                    if (sense)
                    {
                        // Return empty VBuffer.
                        VBufferUtils.Resize(ref dst, srcLength, 0);
                        return;
                    }

                    // Return VBuffer filled with 1's.
                    var editor = VBufferEditor.Create(ref dst, srcLength);
                    for (int i = 0; i < srcLength; i++)
                        editor.Values[i] = true;
                    dst = editor.Commit();
                    return;
                }

                if (sense && indices.Count < srcLength / 2)
                {
                    // Will produce sparse output.
                    int dstCount = indices.Count;
                    var editor = VBufferEditor.Create(ref dst, srcLength, dstCount);

                    indices.CopyTo(editor.Indices);
                    for (int ii = 0; ii < dstCount; ii++)
                        editor.Values[ii] = true;

                    Host.Assert(dstCount <= srcLength);
                    dst = editor.Commit();
                }
                else if (!sense && srcLength - indices.Count < srcLength / 2)
                {
                    // Will produce sparse output.
                    int dstCount = srcLength - indices.Count;
                    var editor = VBufferEditor.Create(ref dst, srcLength, dstCount);

                    // Appends the length of the src to make the loop simpler,
                    // as the length of src will never be reached in the loop.
                    indices.Add(srcLength);

                    int iiDst = 0;
                    int iiSrc = 0;
                    int iNext = indices[iiSrc];
                    for (int i = 0; i < srcLength; i++)
                    {
                        Host.Assert(0 <= i && i <= iNext);
                        Host.Assert(iiSrc + iiDst == i);
                        if (i < iNext)
                        {
                            Host.Assert(iiDst < dstCount);
                            editor.Values[iiDst] = true;
                            editor.Indices[iiDst++] = i;
                        }
                        else
                        {
                            Host.Assert(iiSrc + 1 < indices.Count);
                            Host.Assert(iNext < indices[iiSrc + 1]);
                            iNext = indices[++iiSrc];
                        }
                    }
                    Host.Assert(srcLength == iiSrc + iiDst);
                    Host.Assert(iiDst == dstCount);

                    dst = editor.Commit();
                }
                else
                {
                    // Will produce dense output.
                    var editor = VBufferEditor.Create(ref dst, srcLength);

                    // Appends the length of the src to make the loop simpler,
                    // as the length of src will never be reached in the loop.
                    indices.Add(srcLength);

                    int ii = 0;
                    for (int i = 0; i < srcLength; i++)
                    {
                        Host.Assert(0 <= i && i <= indices[ii]);
                        if (i == indices[ii])
                        {
                            editor.Values[i] = sense;
                            ii++;
                            Host.Assert(ii < indices.Count);
                            Host.Assert(indices[ii - 1] < indices[ii]);
                        }
                        else
                            editor.Values[i] = !sense;
                    }

                    dst = editor.Commit();
                }
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    ColInfo info = _infos[iinfo];
                    string inputColumnName = info.InputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(inputColumnName),
                        ctx.AddIntermediateVariable(_infos[iinfo].OutputType, info.Name)))
                    {
                        ctx.RemoveColumn(info.Name, true);
                    }
                }
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
            {
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoadName);

                var inputType = _infos[iinfo].InputType;
                Type rawType = (inputType is VectorDataViewType vectorType) ? vectorType.ItemType.RawType : inputType.RawType;

                if (rawType != typeof(float))
                    return false;

                string opType;
                opType = "IsNaN";
                var isNaNOutput = ctx.AddIntermediateVariable(BooleanDataViewType.Instance, "IsNaNOutput", true);
                var nanNode = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType), "");

                return true;
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for the <see cref="MissingValueIndicatorTransformer"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Vector or scalar value of <xref:System.Single> or <xref:System.Double> |
    /// | Output column data type | If input column was scalar then <xref:System.Boolean> otherwise vector of <xref:System.Boolean>. |
    /// | Exportable to ONNX | Yes |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.MissingValueIndicatorTransformer> creates a new column, named as specified in the output column name parameters, and
    /// fills it with vector of bools where `true` in the i-th position in array indicates the i-th element in input column has missing value and `false` otherwise.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ExtensionsCatalog.IndicateMissingValues(TransformsCatalog, string, string)" />
    /// <seealso cref="ExtensionsCatalog.IndicateMissingValues(TransformsCatalog, InputOutputColumnPair[])" />
    public sealed class MissingValueIndicatorEstimator : TrivialEstimator<MissingValueIndicatorTransformer>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueIndicatorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        internal MissingValueIndicatorEstimator(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueIndicatorTransformer)), new MissingValueIndicatorTransformer(env, columns))
        {
            Contracts.CheckValue(env, nameof(env));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueIndicatorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        internal MissingValueIndicatorEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null)
            : this(env, (outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colPair.inputColumnName, out var col) || !Data.Conversion.Conversions.DefaultInstance.TryGetIsNAPredicate(col.ItemType, out Delegate del))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.inputColumnName);
                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                DataViewType type = !(col.ItemType is VectorDataViewType vectorType) ?
                    (DataViewType)BooleanDataViewType.Instance :
                    new VectorDataViewType(BooleanDataViewType.Instance, vectorType.Dimensions);
                result[colPair.outputColumnName] = new SchemaShape.Column(colPair.outputColumnName, col.Kind, type, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }
}
