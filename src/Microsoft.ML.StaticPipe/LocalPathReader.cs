using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML
{
    public static class LocalPathReader
    {
        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}