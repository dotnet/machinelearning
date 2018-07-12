// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(LabelConvertTransform.Summary, typeof(LabelConvertTransform), typeof(LabelConvertTransform.Arguments), typeof(SignatureDataTransform),
    "", "LabelConvert", "LabelConvertTransform")]

[assembly: LoadableClass(LabelConvertTransform.Summary, typeof(LabelConvertTransform), null, typeof(SignatureLoadDataTransform),
    "Label Convert Transform", LabelConvertTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class LabelConvertTransform : OneToOneTransformBase
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

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col")]
            public Column[] Column;
        }

        internal const string Summary = "Convert a label column into a standard floating point representation.";

        public const string LoaderSignature = "LabelConvertTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LABCONVT",
                verWrittenCur: 0x00010001, // Initial.
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "LabelConvert";
        private VectorType _slotType;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the input column.  If this is null '<paramref name="name"/>' will be used.</param>
        public LabelConvertTransform(IHostEnvironment env, IDataView input, string name, string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } } }, input)
        {
        }

        public LabelConvertTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, RowCursorUtils.TestGetLabelGetter)
        {
            Contracts.AssertNonEmpty(Infos);
            Contracts.Assert(Infos.Length == Utils.Size(args.Column));
        }

        private LabelConvertTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
        }

        public static LabelConvertTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    h.CheckDecode(cbFloat == sizeof(Float));
                    return new LabelConvertTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            Host.AssertNonEmpty(Infos);
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo & iinfo < Infos.Length);
            return NumberType.Float;
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source, PassThrough))
                {
                    // No additional metadata.
                }
            }
            md.Seal();
        }

        /// <summary>
        /// Returns whether metadata of the indicated kind should be passed through from the source column.
        /// </summary>
        private bool PassThrough(string kind, int iinfo)
        {
            Contracts.AssertNonEmpty(kind);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            // REVIEW: I'm suppressing this because it would be strange to see a non-key
            // output column with KeyValues metadata, but maybe this output is actually useful?
            // Certainly there's nothing contractual requiring I suppress this. Should I suppress
            // anything else?
            return kind != MetadataUtils.Kinds.KeyValues;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Contracts.AssertValueOrNull(ch);
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

            disposer = null;
            int col = Infos[iinfo].Source;
            var typeSrc = input.Schema.GetColumnType(col);
            Contracts.Assert(RowCursorUtils.TestGetLabelGetter(typeSrc) == null);
            return RowCursorUtils.GetLabelGetter(input, col);
        }

        protected override VectorType GetSlotTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            var srcSlotType = Infos[iinfo].SlotTypeSrc;
            if (srcSlotType == null)
                return null;
            // THe following slot type will be the same for any columns, so we have only one field,
            // as opposed to one for each column.
            Interlocked.CompareExchange(ref _slotType, new VectorType(NumberType.Float, srcSlotType), null);
            return _slotType;
        }

        protected override ISlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(Infos[iinfo].SlotTypeSrc);

            ISlotCursor cursor = InputTranspose.GetSlotCursor(Infos[iinfo].Source);
            return new SlotCursor(Host, cursor, GetSlotTypeCore(iinfo));
        }

        private sealed class SlotCursor : SynchronizedCursorBase<ISlotCursor>, ISlotCursor
        {
            private readonly Delegate _getter;
            private readonly VectorType _type;

            public SlotCursor(IChannelProvider provider, ISlotCursor cursor, VectorType typeDst)
                : base(provider, cursor)
            {
                Ch.AssertValue(typeDst);
                _getter = RowCursorUtils.GetLabelGetter(Input);
                _type = typeDst;
            }

            public VectorType GetSlotType()
            {
                return _type;
            }

            public ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }
    }
}
