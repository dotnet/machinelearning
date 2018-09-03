// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// The trivial wrapper for a <see cref="IDataReader{TSource}"/> that acts as an estimator and ignores the source.
    /// </summary>
    public sealed class TrivialReaderEstimator<TSource, TReader>: IDataReaderEstimator<TSource, TReader>
        where TReader: IDataReader<TSource>
    {
        private readonly TReader _reader;

        public TrivialReaderEstimator(TReader reader)
        {
            _reader = reader;
        }

        public TReader Fit(TSource input) => _reader;

        public SchemaShape GetOutputSchema() => SchemaShape.Create(_reader.GetOutputSchema());
    }
}
