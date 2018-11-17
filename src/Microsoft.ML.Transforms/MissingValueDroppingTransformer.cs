// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(MissingValueDroppingTransformer), typeof(MissingValueDroppingTransformer.Arguments), typeof(SignatureDataTransform),
    MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.ShortName, "NADropTransform")]

[assembly: LoadableClass(MissingValueDroppingTransformer.Summary, typeof(MissingValueDroppingTransformer), null, typeof(SignatureLoadDataTransform),
    MissingValueDroppingTransformer.FriendlyName, MissingValueDroppingTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="NADrop"]'/>
    public sealed class MissingValueDroppingTransformer : OneToOneTransformBase
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
        public const string LoaderSignature = "NADropTransform";

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

        // The isNA delegates, parallel to Infos.
        private readonly Delegate[] _isNAs;

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueDroppingTransformer"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        public MissingValueDroppingTransformer(IHostEnvironment env, IDataView input, string name, string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } } }, input)
        {
        }

        public MissingValueDroppingTransformer(IHostEnvironment env, Arguments args, IDataView input)
            : base(Contracts.CheckRef(env, nameof(env)), RegistrationName, env.CheckRef(args, nameof(args)).Column, input, TestType)
        {
            Host.CheckNonEmpty(args.Column, nameof(args.Column));
            _isNAs = InitIsNAAndMetadata();
        }

        private Delegate[] InitIsNAAndMetadata()
        {
            var md = Metadata;
            var isNAs = new Delegate[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var type = Infos[iinfo].TypeSrc;
                isNAs[iinfo] = GetIsNADelegate(type);
                // Register for metadata. Propagate the IsNormalized metadata.
                // SlotNames will not be propagated.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source,
                    MetadataUtils.Kinds.IsNormalized, MetadataUtils.Kinds.KeyValues))
                {
                    // Output does not have missings.
                    bldr.AddPrimitive(MetadataUtils.Kinds.HasMissingValues, BoolType.Instance, false);
                }
            }
            md.Seal();
            return isNAs;
        }

        /// <summary>
        /// Returns the isNA predicate for the respective type.
        /// </summary>
        private Delegate GetIsNADelegate(ColumnType type)
        {
            Func<ColumnType, Delegate> func = GetIsNADelegate<int>;
            return Utils.MarshalInvoke(func, type.ItemType.RawType, type);
        }

        private Delegate GetIsNADelegate<T>(ColumnType type)
        {
            return Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);
        }

        private static string TestType(ColumnType type)
        {
            if (!type.IsVector)
            {
                return string.Format("Type '{0}' is not supported by {1} since it is not a vector",
                    type, LoaderSignature);
            }

            // Item type must have an NA value that exists.
            Func<ColumnType, string> func = TestType<int>;
            return Utils.MarshalInvoke(func, type.ItemType.RawType, type.ItemType);
        }

        private static string TestType<T>(ColumnType type)
        {
            Contracts.Assert(type.ItemType.RawType == typeof(T));
            InPredicate<T> isNA;
            if (!Conversions.Instance.TryGetIsNAPredicate(type.ItemType, out isNA))
            {
                return string.Format("Type '{0}' is not supported by {1} since it doesn't have an NA value",
                    type, LoaderSignature);
            }
            return null;
        }

        public static MissingValueDroppingTransformer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new MissingValueDroppingTransformer(h, ctx, input));
        }

        private MissingValueDroppingTransformer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestType)
        {
            Host.AssertValue(ctx);
            // *** Binary format ***
            // <base>
            Host.AssertNonEmpty(Infos);

            _isNAs = InitIsNAAndMetadata();
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            SaveBase(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return new VectorType(Infos[iinfo].TypeSrc.ItemType.AsPrimitive);
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);

            disposer = null;
            Func<IRow, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(Infos[iinfo].TypeSrc.ItemType.RawType);
            return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
        }

        private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(IRow input, int iinfo)
        {
            var srcGetter = GetSrcGetter<VBuffer<TDst>>(input, iinfo);
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