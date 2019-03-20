// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(LabelConvertTransform.Summary, typeof(LabelConvertTransform), typeof(LabelConvertTransform.Arguments), typeof(SignatureDataTransform),
    "", "LabelConvert", "LabelConvertTransform")]

[assembly: LoadableClass(LabelConvertTransform.Summary, typeof(LabelConvertTransform), null, typeof(SignatureLoadDataTransform),
    "Label Convert Transform", LabelConvertTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    [BestFriend]
    internal sealed class LabelConvertTransform : OneToOneTransformBase
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
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column", ShortName = "col")]
            public Column[] Columns;
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LabelConvertTransform).Assembly.FullName);
        }

        private const string RegistrationName = "LabelConvert";
        private VectorType _slotType;

        /// <summary>
        /// Initializes a new instance of <see cref="LabelConvertTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="outputColumnName">Name of the output column.</param>
        /// <param name="inputColumnName">Name of the input column.  If this is null '<paramref name="outputColumnName"/>' will be used.</param>
        public LabelConvertTransform(IHostEnvironment env, IDataView input, string outputColumnName, string inputColumnName = null)
            : this(env, new Arguments() { Columns = new[] { new Column() { Source = inputColumnName ?? outputColumnName, Name = outputColumnName } } }, input)
        {
        }

        public LabelConvertTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Columns, input, RowCursorUtils.TestGetLabelGetter)
        {
            Contracts.AssertNonEmpty(Infos);
            Contracts.Assert(Infos.Length == Utils.Size(args.Columns));
            Metadata.Seal();
        }

        private LabelConvertTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>

            Metadata.Seal();
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
                    h.CheckDecode(cbFloat == sizeof(float));
                    return new LabelConvertTransform(h, ctx, input);
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
            Host.AssertNonEmpty(Infos);
            ctx.Writer.Write(sizeof(float));
            SaveBase(ctx);
        }

        protected override DataViewType GetColumnTypeCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo & iinfo < Infos.Length);
            return NumberDataViewType.Single;
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
            return kind != AnnotationUtils.Kinds.KeyValues;
        }

        protected override Delegate GetGetterCore(IChannel ch, DataViewRow input, int iinfo, out Action disposer)
        {
            Contracts.AssertValueOrNull(ch);
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

            disposer = null;
            int col = Infos[iinfo].Source;
            var typeSrc = input.Schema[col].Type;
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
            Interlocked.CompareExchange(ref _slotType, new VectorType(NumberDataViewType.Single, srcSlotType), null);
            return _slotType;
        }

        [BestFriend]
        internal override SlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(Infos[iinfo].SlotTypeSrc);

            var cursor = InputTranspose.GetSlotCursor(Infos[iinfo].Source);
            return new SlotCursorImpl(Host, cursor, GetSlotTypeCore(iinfo));
        }

        private sealed class SlotCursorImpl : SlotCursor.SynchronizedSlotCursor
        {
            private readonly Delegate _getter;
            private readonly VectorType _type;

            public SlotCursorImpl(IChannelProvider provider, SlotCursor cursor, VectorType typeDst)
                : base(provider, cursor)
            {
                Ch.AssertValue(typeDst);
                _getter = RowCursorUtils.GetLabelGetter(cursor);
                _type = typeDst;
            }

            public override VectorType GetSlotType()
            {
                return _type;
            }

            public override ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }
    }
}
