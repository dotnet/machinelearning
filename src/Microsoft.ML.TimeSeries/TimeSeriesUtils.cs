using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    internal static class TimeSeriesUtils
    {
        internal static void SerializeFixedSizeQueue(FixedSizeQueue<Single> queue, BinaryWriter writer)
        {
            writer.Write(queue.Capacity);
            writer.Write(queue.Count);
            for (int index = 0; index < queue.Count; index++)
                writer.Write(queue[index]);

            return;
        }

        internal static FixedSizeQueue<Single> DeserializeFixedSizeQueueSingle(BinaryReader reader)
        {
            var q = new FixedSizeQueue<Single>(reader.ReadInt32());
            int count = reader.ReadInt32();

            Contracts.Assert(0 <= count & count <= q.Capacity);

            for (int index = 0; index < count; index++)
                q.AddLast(reader.ReadSingle());

            return q;
        }

        internal static void SerializeFixedSizeQueue(FixedSizeQueue<double> queue, BinaryWriter writer)
        {
            writer.Write(queue.Capacity);
            writer.Write(queue.Count);
            for (int index = 0; index < queue.Count; index++)
                writer.Write(queue[index]);

            return;
        }

        internal static FixedSizeQueue<double> DeserializeFixedSizeQueueDouble(BinaryReader reader)
        {
            var q = new FixedSizeQueue<double>(reader.ReadInt32());
            int count = reader.ReadInt32();

            Contracts.Assert(0 <= count & count <= q.Capacity);

            for (int index = 0; index < count; index++)
                q.AddLast(reader.ReadDouble());

            return q;
        }
    }
}
