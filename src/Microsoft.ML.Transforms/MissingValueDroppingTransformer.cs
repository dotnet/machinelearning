// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueDroppingTransformer), typeof(MissingValueDroppingTransformer.Arguments), typeof(SignatureDataTransform),
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
    public sealed class MissingValueDroppingTransformer : OneToOneTransformerBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to drop the NAs for", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
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

        public IReadOnlyList<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueDroppingTransformer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        public MissingValueDroppingTransformer(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), columns)
        {
        }

        internal MissingValueDroppingTransformer(IHostEnvironment env, Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), GetColumnPairs(args.Column))
        {
        }

        private MissingValueDroppingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingTransformer)), ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private static (string input, string output)[] GetColumnPairs(Column[] columns)
            => columns.Select(c => (c.Source ?? c.Name, c.Name)).ToArray();

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema.GetColumnType(srcCol);
            if (!inType.IsVector)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputSchema.GetColumnName(srcCol), "Vector", inType.ToString());
        }

        // Factory method for SignatureLoadModel
        private static MissingValueDroppingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());

            return new MissingValueDroppingTransformer(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
            => new MissingValueDroppingTransformer(env, args).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

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

        protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly MissingValueDroppingTransformer _parent;

            private readonly ColumnType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly ColumnType[] _types;
            private readonly Delegate[] _isNAs;

            public Mapper(MissingValueDroppingTransformer parent, Schema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                _isNAs = new Delegate[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                    _types[i] = new VectorType(srcCol.Type.ItemType.AsPrimitive);
                    _isNAs[i] = GetIsNADelegate(srcCol.Type);
                }
            }

            /// <summary>
            /// Returns the isNA predicate for the respective type.
            /// </summary>
            private Delegate GetIsNADelegate(ColumnType type)
            {
                Func<ColumnType, Delegate> func = GetIsNADelegate<int>;
                return Utils.MarshalInvoke(func, type.ItemType.RawType, type);
            }

            private Delegate GetIsNADelegate<T>(ColumnType type) => Runtime.Data.Conversion.Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new MetadataBuilder();
                    builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, x => x == MetadataUtils.Kinds.KeyValues || x == MetadataUtils.Kinds.IsNormalized);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }
            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                Func<Row, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_srcTypes[iinfo].ItemType.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
            }

            private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(Row input, int iinfo)
            {
                var srcGetter = input.GetGetter<VBuffer<TDst>>(_srcCols[iinfo]);
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
    /// <summary>
    /// Drops missing values from columns.
    /// </summary>
    public sealed class MissingValueDroppingEstimator : TrivialEstimator<MissingValueDroppingTransformer>
    {
        /// <summary>
        /// Drops missing values from columns.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        public MissingValueDroppingEstimator(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueDroppingEstimator)), new MissingValueDroppingTransformer(env, columns))
        {
            Contracts.CheckValue(env, nameof(env));
        }

        /// <summary>
        /// Drops missing values from columns.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="input">The name of the input column of the transformation.</param>
        /// <param name="output">The name of the column produced by the transformation.</param>
        public MissingValueDroppingEstimator(IHostEnvironment env, string input, string output = null)
            : this(env, (input, output ?? input))
        {
        }

        /// <summary>
        /// Returns the schema that would be produced by the transformation.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colPair.input, out var col) || !Runtime.Data.Conversion.Conversions.Instance.TryGetIsNAPredicate(col.ItemType, out Delegate del))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector || col.Kind == SchemaShape.Column.VectorKind.VariableVector))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.input, "Vector", col.GetTypeString());
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMeta))
                    metadata.Add(keyMeta);
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.IsNormalized, out var normMeta))
                    metadata.Add(normMeta);
                result[colPair.output] = new SchemaShape.Column(colPair.output, SchemaShape.Column.VectorKind.VariableVector, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }
    }
}