// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

[assembly: LoadableClass(TransposeSaver.Summary, typeof(TransposeSaver), typeof(TransposeSaver.Arguments), typeof(SignatureDataSaver),
    "Transpose Saver", TransposeSaver.LoadName, "TransposedSaver", "Transpose", "Transposed", "trans")]

namespace Microsoft.ML.Runtime.Data.IO
{
    /// <summary>
    /// Saver for a format that can be loaded using the <see cref="TransposeLoader"/>.
    /// </summary>
    /// <seealso cref="TransposeLoader"/>
    public sealed class TransposeSaver : IDataSaver
    {
        public sealed class Arguments
        {
            // REVIEW: Some use cases made clear to me that successfully using this with it *off* was actually
            // incredibly difficult (requiring deep knowledge of why, for instance, scoring cannot be done on slot-wise
            // data), so we will leave turning this off as an "advanced" option.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Write a copy of the data in row-wise format, in addition to the transposed data. This will increase performance for mixed applications while taking significantly more space.", ShortName = "row")]
            public bool WriteRowData = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Suppress any info output (not warnings or errors)", Hide = true)]
            public bool Silent;
        }

        internal const string Summary = "Writes data into a transposed binary TDV file.";
        internal const string LoadName = "TransposeSaver";
        private const ulong WriterVersion = 0x0001000100010001;

        private readonly IHost _host;
        private readonly IDataSaver _internalSaver;
        private readonly bool _writeRowData;
        private readonly bool _silent;

        /// <summary>
        /// Constructs a saver for a data view.
        /// </summary>
        public TransposeSaver(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            _host = env.Register(LoadName);
            _internalSaver = new BinarySaver(_host, new BinarySaver.Arguments() { Silent = true });
            _writeRowData = args.WriteRowData;
            _silent = args.Silent;
        }

        public bool IsColumnSavable(ColumnType type)
        {
            _host.CheckValue(type, nameof(type));
            // We can't transpose variable length columns at all, so nor can we save them.
            if (type.IsVector && !type.IsKnownSizeVector)
                return false;
            // Since we'll be presumably saving vectors of these, attempt to construct
            // an artificial vector type out of this. Obviously if you can't make a vector
            // out of the items, then you could not save each slot's values.
            var itemType = type.ItemType;
            var primitiveType = itemType.AsPrimitive;
            if (primitiveType == null)
                return false;
            var vectorType = new VectorType(primitiveType, size: 2);
            return _internalSaver.IsColumnSavable(vectorType);
        }

        public void SaveData(Stream stream, IDataView data, params int[] cols)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(data, nameof(data));
            _host.CheckParam(stream.CanSeek, nameof(stream), "Must be seekable but is not");
            _host.CheckParam(stream.Position == 0, nameof(stream), "Stream must be at beginning but appears to not be");
            _host.CheckNonEmpty(cols, nameof(cols));

            // If the input dataview is already a transposed data view, with all requested
            // columns set to be transposed, creating the transposer will amount to a no-op,
            // which is totally fine.
            var trans = Transposer.Create(_host, data, forceSave: false, columns: cols);
            using (var ch = _host.Start("Saving"))
            {
                SaveTransposedData(ch, stream, trans, cols);
            }
        }

        private void SaveTransposedData(IChannel ch, Stream stream, ITransposeDataView data, int[] cols)
        {
            _host.AssertValue(ch);
            ch.AssertValue(stream);
            ch.AssertValue(data);
            ch.AssertNonEmpty(cols);
            ch.Assert(stream.CanSeek);

            // Initialize what we can in the header, though we will not be writing out things in the
            // header until we have confidence that things were written out correctly.
            TransposeLoader.Header header = default(TransposeLoader.Header);
            header.Signature = TransposeLoader.Header.SignatureValue;
            header.Version = TransposeLoader.Header.WriterVersion;
            header.CompatibleVersion = TransposeLoader.Header.WriterVersion;
            VectorType slotType = data.TransposeSchema.GetSlotType(cols[0]);
            ch.AssertValue(slotType);
            header.RowCount = slotType.ValueCount;
            header.ColumnCount = cols.Length;

            // We keep track of the offsets of the start of each sub-IDV, for use in writing out the
            // offsets/length table later.
            List<long> offsets = new List<long>();
            // First write a bunch of zeros at the head, as a placeholder for the header that
            // will go there assuming we can successfully load it. We'll keep this array around
            // for the real marshalling and writing of the header bytes structure.
            byte[] headerBytes = new byte[TransposeLoader.Header.HeaderSize];
            stream.Write(headerBytes, 0, headerBytes.Length);
            offsets.Add(stream.Position);

            // This is a convenient delegate to write out an IDV substream, then save the offsets
            // where writing stopped to the offsets list.
            Action<string, IDataView> viewAction =
                (name, view) =>
                {
                    using (var substream = new SubsetStream(stream))
                    {
                        _internalSaver.SaveData(substream, view, Utils.GetIdentityPermutation(view.Schema.ColumnCount));
                        substream.Seek(0, SeekOrigin.End);
                        ch.Info("Wrote {0} data view in {1} bytes", name, substream.Length);
                    }
                    offsets.Add(stream.Position);
                };

            // First write out the no-row data, limited to these columns.
            IDataView subdata = new ChooseColumnsByIndexTransform(_host,
                new ChooseColumnsByIndexTransform.Arguments() { Index = cols }, data);
            // If we want the "dual mode" row-wise and slot-wise file, don't filter out anything.
            if (!_writeRowData)
                subdata = SkipTakeFilter.Create(_host, new SkipTakeFilter.TakeArguments() { Count = 0 }, subdata);

            string msg = _writeRowData ? "row-wise data, schema, and metadata" : "schema and metadata";
            viewAction(msg, subdata);
            foreach (var col in cols)
                viewAction(data.Schema.GetColumnName(col), new TransposerUtils.SlotDataView(_host, data, col));

            // Wrote out the dataview. Write out the table offset.
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
            {
                // Format of the table is offset, length, both as 8-byte integers.
                // As it happens we wrote things out as adjacent sub-IDVs, so the
                // length can be derived from the offsets. The first will be the
                // start of the first sub-IDV, and all subsequent entries will be
                // the start/end of the current/next sub-IDV, respectively, so a total
                // of cols.Length + 2 entries.
                ch.Assert(offsets.Count == cols.Length + 2);
                ch.Assert(offsets[offsets.Count - 1] == stream.Position);
                header.SubIdvTableOffset = stream.Position;
                for (int c = 1; c < offsets.Count; ++c)
                {
                    // 8-byte int for offsets, 8-byte int for length.
                    writer.Write(offsets[c - 1]);
                    writer.Write(offsets[c] - offsets[c - 1]);
                }
                header.TailOffset = stream.Position;
                writer.Write(TransposeLoader.Header.TailSignatureValue);

                // Now we are confident that things will work, so write it out.
                unsafe
                {
                    Marshal.Copy(new IntPtr(&header), headerBytes, 0, Marshal.SizeOf(typeof(Header)));
                }
                writer.Seek(0, SeekOrigin.Begin);
                writer.Write(headerBytes);
            }
        }
    }
}
