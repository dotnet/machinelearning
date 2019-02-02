using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML
{
    public static class LocalPathReader
    {
        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, string path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}