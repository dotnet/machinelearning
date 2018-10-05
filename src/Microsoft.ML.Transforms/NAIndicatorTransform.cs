// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), typeof(NAIndicatorTransform.Arguments), typeof(SignatureDataTransform),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName, "NAIndicator", NAIndicatorTransform.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), null, typeof(SignatureLoadDataTransform),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(NAIndicatorTransform), null, typeof(SignatureLoadModel),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NAIndicatorTransform), null, typeof(SignatureLoadRowMapper),
   NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

namespace Microsoft.ML.Runtime.Data
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

        public const string LoadName = "NaIndicatorTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                // REVIEW: temporary name
                modelSignature: "NAIND TF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName,
                loaderAssemblyName: typeof(NAIndicatorTransform).Assembly.FullName);
        }

        internal const string Summary = "Create a boolean output column with the same number of slots as the input column, where the output value"
            + " is true if the value in the input column is missing.";
        internal const string FriendlyName = "NA Indicator Transform";
        internal const string ShortName = "NAInd";

        internal static string TestType(ColumnType type)
        {
            // Item type must have an NA value. We'll get the predicate again later when we're ready to use it.
            Delegate del;
            if (Conversions.Instance.TryGetIsNAPredicate(type.ItemType, out del))
                return null;
            return string.Format("Type '{0}' is not supported by {1} since it doesn't have an NA value",
                type, LoadName);
        }

        // TODO: Why do we even need an object for this? maybe because we want to deal with array of these things
        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            public ColumnInfo(string input, string output)
            {
                Input = input;
                Output = output;
            }
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private const string RegistrationName = "NaIndicator";

        // The output column types, parallel to Infos.
        private readonly ColumnType[] _replaceTypes;

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            string reason = TestType(type);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columns"></param>
        public NAIndicatorTransform(IHostEnvironment env, IDataView input, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAReplaceTransform)), GetColumnPairs(columns))
        {
            // Check that all the input columns are present and correct.
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input);
                CheckInputColumn(input.Schema, i, srcCol);
            }
        }

        private NAIndicatorTransform(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            Host.AssertValue(ctx);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];

                cols[i] = new ColumnInfo(item.Source, item.Name);
            };
            return new NAIndicatorTransform(env, input, cols).MakeDataTransform(input);
        }

        public static NAIndicatorTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new NAIndicatorTransform(host, ctx);
        }

        public static IDataTransform Create(IHostEnvironment env, IDataView input, params ColumnInfo[] columns)
            => new NAIndicatorTransform(env, input, columns).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private void WriteTypeAndValue<T>(Stream stream, BinarySaver saver, ColumnType type, T rep)
        {
            Host.AssertValue(stream);
            Host.AssertValue(saver);
            Host.Assert(type.RawType == typeof(T) || type.ItemType.RawType == typeof(T));

            if (!saver.TryWriteTypeAndValue<T>(stream, type, ref rep, out int bytesWritten))
                throw Host.Except("We do not know how to serialize terms of type '{0}'", type);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);
            var saver = new BinarySaver(Host, new BinarySaver.Arguments());
            for (int iinfo = 0; iinfo < _replaceTypes.Length; iinfo++)
            {
                var repType = _replaceTypes[iinfo].ItemType;
                object[] args = new object[] { ctx.Writer.BaseStream, saver, repType, ???? };
                Action<Stream, BinarySaver, ColumnType, int> func = WriteTypeAndValue<int>;
                Host.Assert(repValue.GetType() == _replaceTypes[iinfo].RawType || repValue.GetType() == _replaceTypes[iinfo].ItemType.RawType);
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(repValue.GetType());
                meth.Invoke(this, args);
            }
        }

        private ColumnType[] GetTypesAndMetadata()
        {
            var md = Metadata;
            var types = new ColumnType[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var type = Infos[iinfo].TypeSrc;

                if (!type.IsVector)
                    types[iinfo] = BoolType.Instance;
                else
                    types[iinfo] = new VectorType(BoolType.Instance, type.AsVector);
                // Pass through slot name metadata.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source, MetadataUtils.Kinds.SlotNames))
                {
                    // Output is normalized.
                    bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, true);
                }
            }
            md.Seal();
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            if (!Infos[iinfo].TypeSrc.IsVector)
                return ComposeGetterOne(input, iinfo);
            return ComposeGetterVec(input, iinfo);
        }

        /// <summary>
        /// Getter generator for single valued inputs.
        /// </summary>
        private ValueGetter<bool> ComposeGetterOne(IRow input, int iinfo)
        {
            Func<IRow, int, ValueGetter<bool>> func = ComposeGetterOne<int>;
            return Utils.MarshalInvoke(func, Infos[iinfo].TypeSrc.RawType, input, iinfo);
        }

        /// <summary>
        ///  Tests if a value is NA for scalars.
        /// </summary>
        private ValueGetter<bool> ComposeGetterOne<T>(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<T>(input, iinfo);
            var isNA = Conversions.Instance.GetIsNAPredicate<T>(input.Schema.GetColumnType(Infos[iinfo].Source));
            T src = default(T);
            return
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
        {
            Func<IRow, int, ValueGetter<VBuffer<bool>>> func = ComposeGetterVec<int>;
            return Utils.MarshalInvoke(func, Infos[iinfo].TypeSrc.ItemType.RawType, input, iinfo);
        }

        /// <summary>
        ///  Tests if a value is NA for vectors.
        /// </summary>
        private ValueGetter<VBuffer<bool>> ComposeGetterVec<T>(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<VBuffer<T>>(input, iinfo);
            var isNA = Conversions.Instance.GetIsNAPredicate<T>(input.Schema.GetColumnType(Infos[iinfo].Source).ItemType);
            var val = default(T);
            bool defaultIsNA = isNA(ref val);
            var src = default(VBuffer<T>);
            var indices = new List<int>();
            return
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

        protected override IRowMapper MakeRowMapper(ISchema schema)
        {
        }
    }
}
