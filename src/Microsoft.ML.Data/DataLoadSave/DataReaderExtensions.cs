using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    public static class DataReaderExtensions
    {
        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, string path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}
