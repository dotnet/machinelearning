using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Core.StrongPipe
{
    public sealed class DataReader<TIn, TTupleShape>
        : IDataReader<TIn>
    {
        private readonly IDataReader<TIn> _inner;

        ISchema IDataReader<TIn>.GetOutputSchema() => _inner.GetOutputSchema();
        IDataView IDataReader<TIn>.Read(TIn input) => _inner.Read(input);

        internal DataReader(IDataReader<TIn> inner)
        {
            _inner = inner;
        }
    }
}
