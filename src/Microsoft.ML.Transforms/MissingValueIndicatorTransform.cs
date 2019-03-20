// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(MissingValueIndicatorTransform), typeof(MissingValueIndicatorTransform.Arguments), typeof(SignatureDataTransform),
    "", "MissingValueIndicatorTransform", "MissingValueTransform", "MissingTransform", "Missing")]

[assembly: LoadableClass(typeof(MissingValueIndicatorTransform), null, typeof(SignatureLoadDataTransform),
    "Missing Value Indicator Transform", MissingValueIndicatorTransform.LoaderSignature, "MissingFeatureFunction")]

namespace Microsoft.ML.Transforms
{
    internal sealed class MissingValueIndicatorTransform : OneToOneTransformBase
    {
        public sealed class Column : OneToOneColumn
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

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        public const string LoaderSignature = "MissingIndicatorFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MISFEATF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Changed to OneToOneColumn
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                // This is an older name and can be removed once we don't care about old code
                // being able to load this.
                loaderSignatureAlt: "MissingFeatureFunction",
                loaderAssemblyName: typeof(MissingValueIndicatorTransform).Assembly.FullName);
        }

        private const string RegistrationName = "MissingIndicator";

        private const string IndicatorSuffix = "_Indicator";

        // The output column types, parallel to Infos.
        private readonly VectorType[] _types;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public MissingValueIndicatorTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Columns,
                input, TestIsFloatItem)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Columns));

            _types = GetTypesAndMetadata();
        }

        private MissingValueIndicatorTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsFloatItem)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            Host.AssertNonEmpty(Infos);

            _types = GetTypesAndMetadata();
        }

        public static MissingValueIndicatorTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(float));
                    return new MissingValueIndicatorTransform(h, ctx, input);
                });
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            ctx.Writer.Write(sizeof(float));
            SaveBase(ctx);
        }

        private VectorType[] GetTypesAndMetadata()
        {
            var md = Metadata;
            var types = new VectorType[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var type = Infos[iinfo].TypeSrc;

                // This ensures that our feature count doesn't overflow.
                Host.Check(type.GetValueCount() < int.MaxValue / 2);

                if (!(type is VectorType vectorType))
                    types[iinfo] = new VectorType(NumberDataViewType.Single, 2);
                else
                {
                    types[iinfo] = new VectorType(NumberDataViewType.Single, vectorType, 2);

                    // Produce slot names metadata iff the source has (valid) slot names.
                    VectorType typeNames;
                    if (!vectorType.IsKnownSize ||
                        (typeNames = Source.Schema[Infos[iinfo].Source].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType) == null ||
                        typeNames.Size != vectorType.Size ||
                        !(typeNames.ItemType is TextDataViewType))
                    {
                        continue;
                    }
                }

                // Add slot names metadata.
                using (var bldr = md.BuildMetadata(iinfo))
                {
                    bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(AnnotationUtils.Kinds.SlotNames,
                        AnnotationUtils.GetNamesType(types[iinfo].Size), GetSlotNames);
                }
            }
            md.Seal();
            return types;
        }

        protected override DataViewType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            int size = _types[iinfo].Size;
            if (size == 0)
                throw AnnotationUtils.ExceptGetAnnotation();

            var editor = VBufferEditor.Create(ref dst, size);

            var type = Infos[iinfo].TypeSrc;
            if (!(type is VectorType srcVectorType))
            {
                Host.Assert(_types[iinfo].Size == 2);
                var columnName = Source.Schema[Infos[iinfo].Source].Name;
                editor.Values[0] = columnName.AsMemory();
                editor.Values[1] = (columnName + IndicatorSuffix).AsMemory();
            }
            else
            {
                Host.Assert(srcVectorType.IsKnownSize);
                Host.Assert(size == 2 * srcVectorType.Size);

                // REVIEW: Do we need to verify that there is metadata or should we just call GetMetadata?
                var typeNames = Source.Schema[Infos[iinfo].Source].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType;
                if (typeNames == null || typeNames.Size != srcVectorType.Size || !(typeNames.ItemType is TextDataViewType))
                    throw AnnotationUtils.ExceptGetAnnotation();

                var names = default(VBuffer<ReadOnlyMemory<char>>);
                Source.Schema[Infos[iinfo].Source].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref names);

                // We both assert and check. If this fails, there is a bug somewhere (possibly in this code
                // but more likely in the implementation of Base. On the other hand, we don't want to proceed
                // if we've received garbage.
                Host.Check(names.Length == srcVectorType.Size, "Unexpected slot name vector size");

                var sb = new StringBuilder();
                int slot = 0;
                foreach (var kvp in names.Items(all: true))
                {
                    Host.Assert(0 <= slot && slot < size);
                    Host.Assert(slot % 2 == 0);

                    sb.Clear();
                    if (kvp.Value.IsEmpty)
                        sb.Append('[').Append(slot / 2).Append(']');
                    else
                        sb.AppendMemory(kvp.Value);

                    int len = sb.Length;
                    sb.Append(IndicatorSuffix);
                    var str = sb.ToString();

                    editor.Values[slot++] = str.AsMemory().Slice(0, len);
                    editor.Values[slot++] = str.AsMemory();
                }
                Host.Assert(slot == size);
            }

            dst = editor.Commit();
        }

        protected override Delegate GetGetterCore(IChannel ch, DataViewRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            ValueGetter<VBuffer<float>> del;
            if (Infos[iinfo].TypeSrc is VectorType)
            {
                var getSrc = GetSrcGetter<VBuffer<float>>(input, iinfo);
                del =
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref dst);
                        FillValues(Host, ref dst);
                    };
            }
            else
            {
                var getSrc = GetSrcGetter<float>(input, iinfo);
                del =
                    (ref VBuffer<float> dst) =>
                    {
                        var src = default(float);
                        getSrc(ref src);
                        FillValues(src, ref dst);
                        Host.Assert(dst.Length == 2);
                    };
            }
            return del;
        }

        private static void FillValues(float input, ref VBuffer<float> result)
        {
            if (input == 0)
            {
                VBufferUtils.Resize(ref result, 2, 0);
                return;
            }

            var editor = VBufferEditor.Create(ref result, 2, 1);
            if (float.IsNaN(input))
            {
                editor.Values[0] = 1;
                editor.Indices[0] = 1;
            }
            else
            {
                editor.Values[0] = input;
                editor.Indices[0] = 0;
            }

            result = editor.Commit();
        }

        // This converts in place.
        private static void FillValues(IExceptionContext ectx, ref VBuffer<float> buffer)
        {
            int size = buffer.Length;
            ectx.Check(0 <= size & size < int.MaxValue / 2);

            var values = buffer.GetValues();
            var editor = VBufferEditor.Create(ref buffer, size * 2, values.Length);
            int iivDst = 0;
            if (buffer.IsDense)
            {
                // Currently, it's dense. We always produce sparse.

                for (int ivSrc = 0; ivSrc < values.Length; ivSrc++)
                {
                    ectx.Assert(iivDst <= ivSrc);
                    var val = values[ivSrc];
                    if (val == 0)
                        continue;
                    if (float.IsNaN(val))
                    {
                        editor.Values[iivDst] = 1;
                        editor.Indices[iivDst] = 2 * ivSrc + 1;
                    }
                    else
                    {
                        editor.Values[iivDst] = val;
                        editor.Indices[iivDst] = 2 * ivSrc;
                    }
                    iivDst++;
                }
            }
            else
            {
                // Currently, it's sparse.

                var indices = buffer.GetIndices();
                int ivPrev = -1;
                for (int iivSrc = 0; iivSrc < values.Length; iivSrc++)
                {
                    ectx.Assert(iivDst <= iivSrc);
                    var val = values[iivSrc];
                    if (val == 0)
                        continue;
                    int iv = indices[iivSrc];
                    ectx.Assert(ivPrev < iv & iv < size);
                    ivPrev = iv;
                    if (float.IsNaN(val))
                    {
                        editor.Values[iivDst] = 1;
                        editor.Indices[iivDst] = 2 * iv + 1;
                    }
                    else
                    {
                        editor.Values[iivDst] = val;
                        editor.Indices[iivDst] = 2 * iv;
                    }
                    iivDst++;
                }
            }

            ectx.Assert(0 <= iivDst & iivDst <= values.Length);
            buffer = editor.CommitTruncated(iivDst);
        }
    }
}
