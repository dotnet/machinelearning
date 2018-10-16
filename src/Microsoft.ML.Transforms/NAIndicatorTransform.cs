// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), typeof(NAIndicatorTransform.Arguments), typeof(SignatureDataTransform),
    NAIndicatorTransform.FriendlyName, nameof(NAIndicatorTransform), "NAIndicator", NAIndicatorTransform.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), null, typeof(SignatureLoadDataTransform),
    NAIndicatorTransform.FriendlyName, nameof(NAIndicatorTransform))]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(NAIndicatorTransform), null, typeof(SignatureLoadModel),
    NAIndicatorTransform.FriendlyName, nameof(NAIndicatorTransform))]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NAIndicatorTransform), null, typeof(SignatureLoadRowMapper),
   NAIndicatorTransform.FriendlyName, nameof(NAIndicatorTransform))]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="NAIndicator"]'/>
    public sealed class NAIndicatorTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAIND TF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: nameof(NAIndicatorTransform),
                loaderAssemblyName: typeof(NAIndicatorTransform).Assembly.FullName);
        }

        internal const string Summary = "Create a boolean output column with the same number of slots as the input column, where the output value"
            + " is true if the value in the input column is missing.";
        internal const string FriendlyName = "NA Indicator Transform";
        internal const string ShortName = "NAInd";

        private const string RegistrationName = nameof(NAIndicatorTransform);

        /// <summary>
        /// Initializes a new instance of <see cref="NAIndicatorTransform"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        internal NAIndicatorTransform(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAIndicatorTransform)), columns)
        {
        }

        internal NAIndicatorTransform(IHostEnvironment env, Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAIndicatorTransform)), GetColumnPairs(args.Column))
        {
        }

        private NAIndicatorTransform(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAIndicatorTransform)), ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private static (string input, string output)[] GetColumnPairs(Column[] columns)
        {
            var cols = new (string input, string output)[columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = columns[i];

                cols[i].input = item.Source;
                cols[i].output = item.Name;
            };
            return cols;
        }

        // Factory method for SignatureLoadModel
        internal static NAIndicatorTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());

            return new NAIndicatorTransform(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
            => new NAIndicatorTransform(env, args).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        /// <summary>
        /// Saves the transform.
        /// </summary>
        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private sealed class ColInfo
            {
                public readonly string Output;
                public readonly string Input;
                public readonly ColumnType OutputType;
                public readonly ColumnType InputType;
                public readonly Delegate InputIsNA;

                public ColInfo(string input, string output, ColumnType inType, ColumnType outType)
                {
                    Input = input;
                    Output = output;
                    InputType = inType;
                    OutputType = outType;
                    InputIsNA = GetIsNADelegate(InputType); ;
                }
            }

            // are we sure we need this? maybe not?
            private readonly NAIndicatorTransform _parent;
            private readonly ColInfo[] _infos;

            public Mapper(NAIndicatorTransform parent, ISchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
            }

            private ColInfo[] CreateInfos(ISchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    var inType = inputSchema.GetColumnType(colSrc);
                    ColumnType outType;
                    if (!inType.IsVector)
                        outType = BoolType.Instance;
                    else
                        outType = new VectorType(BoolType.Instance, inType.AsVector);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].input, _parent.ColumnPairs[i].output, inType, outType);
                }
                return infos;
            }

            public override RowMapperColumnInfo[] GetOutputColumns()
            {
                var result = new RowMapperColumnInfo[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _infos.Length; iinfo++)
                {
                    InputSchema.TryGetColumnIndex(_infos[iinfo].Input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var colMetaInfo = new ColumnMetadataInfo(_infos[iinfo].Output);
                    var meta = RowColumnUtils.GetMetadataAsRow(InputSchema, colIndex, x => x == MetadataUtils.Kinds.SlotNames || x == MetadataUtils.Kinds.IsNormalized);
                    result[iinfo] = new RowMapperColumnInfo(_infos[iinfo].Output, _infos[iinfo].OutputType, meta);
                }
                return result;
            }

            /// <summary>
            /// Returns the isNA predicate for the respective type.
            /// </summary>
            private static Delegate GetIsNADelegate(ColumnType type)
            {
                Func<ColumnType, Delegate> func = GetIsNADelegate<int>;
                return Utils.MarshalInvoke(func, type.ItemType.RawType, type);
            }

            private static Delegate GetIsNADelegate<T>(ColumnType type)
            {
                return Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                if (!_infos[iinfo].InputType.IsVector)
                    return ComposeGetterOne(input, iinfo);
                return ComposeGetterVec(input, iinfo);
            }

            /// <summary>
            /// Getter generator for single valued inputs.
            /// </summary>
            private ValueGetter<bool> ComposeGetterOne(IRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterOne<int>, _infos[iinfo].InputType.RawType, input, iinfo);

            private ValueGetter<bool> ComposeGetterOne<T>(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<T>(ColMapNewToOld[iinfo]);
                var src = default(T);
                var isNA = (RefPredicate<T>)_infos[iinfo].InputIsNA;

                ValueGetter<bool> getter;

                return getter =
                    (ref bool dst) =>
                    {
                        getSrc(ref src);
                        dst = isNA(ref src);
                    };
            }

            /// <summary>
            /// Getter generator for vector valued inputs.
            /// </summary>
            private ValueGetter<VBuffer<bool>> ComposeGetterVec(IRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterVec<int>, _infos[iinfo].InputType.ItemType.RawType, input, iinfo);

            private ValueGetter<VBuffer<bool>> ComposeGetterVec<T>(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<T>>(ColMapNewToOld[iinfo]);
                var isNA = (RefPredicate<T>)_infos[iinfo].InputIsNA;
                var val = default(T);
                var defaultIsNA = isNA(ref val);
                var src = default(VBuffer<T>);
                var indices = new List<int>();

                ValueGetter<VBuffer<bool>> getter;

                return getter =
                    (ref VBuffer<bool> dst) =>
                    {
                        // Sense indicates if the values added to the indices list represent NAs or non-NAs.
                        bool sense;
                        getSrc(ref src);
                        FindNAs(ref src, isNA, defaultIsNA, indices, out sense);
                        FillValues(src.Length, ref dst, indices, sense);
                    };
            }

            /// <summary>
            /// Adds all NAs (or non-NAs) to the indices List.  Whether NAs or non-NAs have been added is indicated by the bool sense.
            /// </summary>
            private void FindNAs<T>(ref VBuffer<T> src, RefPredicate<T> isNA, bool defaultIsNA, List<int> indices, out bool sense)
            {
                Host.AssertValue(isNA);
                Host.AssertValue(indices);

                // Find the indices of all of the NAs.
                indices.Clear();
                var srcValues = src.Values;
                var srcCount = src.Count;
                if (src.IsDense)
                {
                    for (int i = 0; i < srcCount; i++)
                    {
                        if (isNA(ref srcValues[i]))
                            indices.Add(i);
                    }
                    sense = true;
                }
                else if (!defaultIsNA)
                {
                    var srcIndices = src.Indices;
                    for (int ii = 0; ii < srcCount; ii++)
                    {
                        if (isNA(ref srcValues[ii]))
                            indices.Add(srcIndices[ii]);
                    }
                    sense = true;
                }
                else
                {
                    // Note that this adds non-NAs to indices -- this is indicated by sense being false.
                    var srcIndices = src.Indices;
                    for (int ii = 0; ii < srcCount; ii++)
                    {
                        if (!isNA(ref srcValues[ii]))
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
                var dstValues = dst.Values;
                var dstIndices = dst.Indices;

                if (indices.Count == 0)
                {
                    if (sense)
                    {
                        // Return empty VBuffer.
                        dst = new VBuffer<bool>(srcLength, 0, dstValues, dstIndices);
                        return;
                    }

                    // Return VBuffer filled with 1's.
                    Utils.EnsureSize(ref dstValues, srcLength, false);
                    for (int i = 0; i < srcLength; i++)
                        dstValues[i] = true;
                    dst = new VBuffer<bool>(srcLength, dstValues, dstIndices);
                    return;
                }

                if (sense && indices.Count < srcLength / 2)
                {
                    // Will produce sparse output.
                    int dstCount = indices.Count;
                    Utils.EnsureSize(ref dstValues, dstCount, false);
                    Utils.EnsureSize(ref dstIndices, dstCount, false);

                    indices.CopyTo(dstIndices);
                    for (int ii = 0; ii < dstCount; ii++)
                        dstValues[ii] = true;

                    Host.Assert(dstCount <= srcLength);
                    dst = new VBuffer<bool>(srcLength, dstCount, dstValues, dstIndices);
                }
                else if (!sense && srcLength - indices.Count < srcLength / 2)
                {
                    // Will produce sparse output.
                    int dstCount = srcLength - indices.Count;
                    Utils.EnsureSize(ref dstValues, dstCount, false);
                    Utils.EnsureSize(ref dstIndices, dstCount, false);

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
                            dstValues[iiDst] = true;
                            dstIndices[iiDst++] = i;
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

                    dst = new VBuffer<bool>(srcLength, dstCount, dstValues, dstIndices);
                }
                else
                {
                    // Will produce dense output.
                    Utils.EnsureSize(ref dstValues, srcLength, false);

                    // Appends the length of the src to make the loop simpler,
                    // as the length of src will never be reached in the loop.
                    indices.Add(srcLength);

                    int ii = 0;
                    for (int i = 0; i < srcLength; i++)
                    {
                        Host.Assert(0 <= i && i <= indices[ii]);
                        if (i == indices[ii])
                        {
                            dstValues[i] = sense;
                            ii++;
                            Host.Assert(ii < indices.Count);
                            Host.Assert(indices[ii - 1] < indices[ii]);
                        }
                        else
                            dstValues[i] = !sense;
                    }

                    dst = new VBuffer<bool>(srcLength, dstValues, dstIndices);
                }
            }
        }
    }

    public sealed class NAIndicatorEstimator : TrivialEstimator<NAIndicatorTransform>
    {
        private readonly (string input, string output)[] _columnPairs;

        /// <summary>
        /// Initializes a new instance of <see cref="NAIndicatorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        public NAIndicatorEstimator(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAIndicatorTransform)), new NAIndicatorTransform(env, columns))
        {
            Contracts.CheckValue(env, nameof(env));
            _columnPairs = columns;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NAIndicatorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="input">The name of the input column of the transformation.</param>
        /// <param name="output">The name of the column produced by the transformation.</param>
        public NAIndicatorEstimator(IHostEnvironment env, string input, string output = null)
            : this(env, (input, output ?? input))
        {
        }

        /// <summary>
        /// Returns the schema that would be produced by the transformation.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colPair in _columnPairs)
            {
                if (!inputSchema.TryFindColumn(colPair.input, out var col) || !Conversions.Instance.TryGetIsNAPredicate(col.ItemType, out Delegate del))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.input);
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.IsNormalized, out var normalized))
                    metadata.Add(normalized);
                ColumnType type = !col.ItemType.IsVector ? (ColumnType) BoolType.Instance : new VectorType(BoolType.Instance, col.ItemType.AsVector);
                result[colPair.output] = new SchemaShape.Column(colPair.output, col.Kind, type, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class NAIndicatorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutScalar<TValue> : Scalar<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutScalar(Scalar<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TValue> : Vector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TValue> : VarVector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVarVectorColumn(VarVector<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var columnPairs = new (string input, string output)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    columnPairs[i] = (inputNames[col.Input], outputNames[toOutput[i]]);
                }
                return new NAIndicatorEstimator(env, columnPairs);
            }
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static Scalar<bool> IsMissingValue(this Scalar<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static Scalar<bool> IsMissingValue(this Scalar<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<double>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static Vector<bool> IsMissingValue(this Vector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static Vector<bool> IsMissingValue(this Vector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<double>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static VarVector<bool> IsMissingValue(this VarVector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<float>(input);
        }

        /// <summary>
        /// Produces a column of boolean entries indicating wheter input column entries were missing.
        /// </summary>
        /// <param name="input">The input column.</param>
        /// <returns>A column indicating wheter input column entries were missing.</returns>
        public static VarVector<bool> IsMissingValue(this VarVector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<double>(input);
        }
    }
}