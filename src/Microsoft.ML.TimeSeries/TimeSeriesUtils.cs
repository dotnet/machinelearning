using System;
using System.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    internal static class TimeSeriesUtils
    {
        internal static void SerializeFixedSizeQueue(FixedSizeQueue<Single> queue, BinaryWriter writer)
        {
            Contracts.Check(queue != null, nameof(queue));
            Contracts.Assert(queue.Capacity >= 0);
            Contracts.Assert(queue.Count <= queue.Capacity);

            writer.Write(queue.Capacity);
            writer.Write(queue.Count);
            for (int index = 0; index < queue.Count; index++)
                writer.Write(queue[index]);

            return;
        }

        internal static FixedSizeQueue<Single> DeserializeFixedSizeQueueSingle(BinaryReader reader, IHost host)
        {
            int capacity = reader.ReadInt32();

            host.CheckDecode(capacity >= 0);

            var q = new FixedSizeQueue<Single>(capacity);
            int count = reader.ReadInt32();

            host.CheckDecode(0 <= count & count <= capacity);

            for (int index = 0; index < count; index++)
                q.AddLast(reader.ReadSingle());

            return q;
        }

        internal static void SerializeFixedSizeQueue(FixedSizeQueue<double> queue, BinaryWriter writer)
        {
            Contracts.Check(queue != null, nameof(queue));
            Contracts.Assert(queue.Capacity >= 0);
            Contracts.Assert(queue.Count <= queue.Capacity);

            writer.Write(queue.Capacity);
            writer.Write(queue.Count);
            for (int index = 0; index < queue.Count; index++)
                writer.Write(queue[index]);

            return;
        }

        internal static FixedSizeQueue<double> DeserializeFixedSizeQueueDouble(BinaryReader reader, IHost host)
        {
            int capacity = reader.ReadInt32();

            host.CheckDecode(capacity >= 0);

            var q = new FixedSizeQueue<double>(capacity);
            int count = reader.ReadInt32();

            host.CheckDecode(0 <= count & count <= capacity);

            for (int index = 0; index < count; index++)
                q.AddLast(reader.ReadDouble());

            return q;
        }
    }
}
