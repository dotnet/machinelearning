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
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueDroppingTransformer), typeof(MissingValueDroppingTransformer.Options), typeof(SignatureDataTransform),
    MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.ShortName, "NADropTransform")]

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueDroppingTransformer), null, typeof(SignatureLoadDataTransform),
    MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.LoaderSignature)]

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(MissingValueDroppingTransformer), null, typeof(SignatureLoadModel),
    MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(MissingValueDroppingTransformer), null, typeof(SignatureLoadRowMapper),
   MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="NADrop"]'/>
    internal sealed class MissingValueDroppingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to drop the NAs for", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
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

        internal const string Summary = "Removes NAs from vector columns.";
        internal const string FriendlyName = "NA Drop Transform";
        internal const string ShortName = "NADrop";
        internal const string LoaderSignature = "NADropTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NADROPXF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MissingValueDroppingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "DropNAs";

        /// <summary>
        /// The names of the input columns of the transformation and the corresponding names for the output columns.
        /// </summary>
        internal IReadOnlyList<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueDroppingTransformer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        internal MissingValueDroppingTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), columns)
        {
        }

        internal MissingValueDroppingTransformer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), GetColumnPairs(options.Columns))
        {
        }

        private MissingValueDroppingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(Column[] columns)
            => columns.Select(c => (c.Name, c.Source ?? c.Name)).ToArray();

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema[srcCol].Type;
            if (!(inType is VectorType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputSchema[srcCol].Name, "vector", inType.ToString());
        }

        // Factory method for SignatureLoadModel
        private static MissingValueDroppingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());

            return new MissingValueDroppingTransformer(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
            => new MissingValueDroppingTransformer(env, options).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
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

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly MissingValueDroppingTransformer _parent;

            private readonly DataViewType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly DataViewType[] _types;
            private readonly Delegate[] _isNAs;

            public Mapper(MissingValueDroppingTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                _isNAs = new Delegate[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                    _types[i] = new VectorType((PrimitiveDataViewType)srcCol.Type.GetItemType());
                    _isNAs[i] = GetIsNADelegate(srcCol.Type);
                }
            }

            /// <summary>
            /// Returns the isNA predicate for the respective type.
            /// </summary>
            private Delegate GetIsNADelegate(DataViewType type)
            {
                Func<DataViewType, Delegate> func = GetIsNADelegate<int>;
                return Utils.MarshalInvoke(func, type.GetItemType().RawType, type);
            }

            private Delegate GetIsNADelegate<T>(DataViewType type) => Data.Conversion.Conversions.Instance.GetIsNAPredicate<T>(type.GetItemType());

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    builder.Add(InputSchema[ColMapNewToOld[i]].Annotations, x => x == AnnotationUtils.Kinds.KeyValues || x == AnnotationUtils.Kinds.IsNormalized);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.ToAnnotations());
                }
                return result;
            }
            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                Func<DataViewRow, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_srcTypes[iinfo].GetItemType().RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
            }

            private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(DataViewRow input, int iinfo)
            {
                var srcGetter = input.GetGetter<VBuffer<TDst>>(input.Schema[_srcCols[iinfo]]);
                var buffer = default(VBuffer<TDst>);
                var isNA = (InPredicate<TDst>)_isNAs[iinfo];
                var def = default(TDst);
                if (isNA(in def))
                {
                    // Case I: NA equals the default value.
                    return
                        (ref VBuffer<TDst> value) =>
                        {
                            srcGetter(ref buffer);
                            DropNAsAndDefaults(ref buffer, ref value, isNA);
                        };
                }

                // Case II: NA is different form default value.
                Host.Assert(!isNA(in def));
                return
                    (ref VBuffer<TDst> value) =>
                    {
                        srcGetter(ref buffer);
                        DropNAs(ref buffer, ref value, isNA);
                    };
            }

            private void DropNAsAndDefaults<TDst>(ref VBuffer<TDst> src, ref VBuffer<TDst> dst, InPredicate<TDst> isNA)
            {
                Host.AssertValue(isNA);

                var srcValues = src.GetValues();
                int newCount = 0;
                for (int i = 0; i < srcValues.Length; i++)
                {
                    if (!isNA(in srcValues[i]))
                        newCount++;
                }
                Host.Assert(newCount <= srcValues.Length);

                if (newCount == 0)
                {
                    VBufferUtils.Resize(ref dst, 0);
                    return;
                }

                if (newCount == srcValues.Length)
                {
                    Utils.Swap(ref src, ref dst);
                    if (!dst.IsDense)
                    {
                        Host.Assert(dst.GetValues().Length == newCount);
                        VBufferUtils.Resize(ref dst, newCount);
                    }
                    return;
                }

                int iDst = 0;

                // Densifying sparse vectors since default value equals NA and hence should be dropped.
                var editor = VBufferEditor.Create(ref dst, newCount);
                for (int i = 0; i < srcValues.Length; i++)
                {
                    if (!isNA(in srcValues[i]))
                        editor.Values[iDst++] = srcValues[i];
                }
                Host.Assert(iDst == newCount);

                dst = editor.Commit();
            }

            private void DropNAs<TDst>(ref VBuffer<TDst> src, ref VBuffer<TDst> dst, InPredicate<TDst> isNA)
            {
                Host.AssertValue(isNA);

                var srcValues = src.GetValues();
                int newCount = 0;
                for (int i = 0; i < srcValues.Length; i++)
                {
                    if (!isNA(in srcValues[i]))
                        newCount++;
                }
                Host.Assert(newCount <= srcValues.Length);

                if (newCount == 0)
                {
                    VBufferUtils.Resize(ref dst, src.Length - srcValues.Length, 0);
                    return;
                }

                if (newCount == srcValues.Length)
                {
                    Utils.Swap(ref src, ref dst);
                    return;
                }

                int iDst = 0;
                if (src.IsDense)
                {
                    var editor = VBufferEditor.Create(ref dst, newCount);
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        if (!isNA(in srcValues[i]))
                        {
                            editor.Values[iDst] = srcValues[i];
                            iDst++;
                        }
                    }
                    Host.Assert(iDst == newCount);
                    dst = editor.Commit();
                }
                else
                {
                    var newLength = src.Length - srcValues.Length - newCount;
                    var editor = VBufferEditor.Create(ref dst, newLength, newCount);

                    var srcIndices = src.GetIndices();
                    int offset = 0;
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        if (!isNA(in srcValues[i]))
                        {
                            editor.Values[iDst] = srcValues[i];
                            editor.Indices[iDst] = srcIndices[i] - offset;
                            iDst++;
                        }
                        else
                            offset++;
                    }
                    Host.Assert(iDst == newCount);
                    Host.Assert(offset == srcValues.Length - newCount);
                    dst = editor.Commit();
                }
            }
        }
    }
}