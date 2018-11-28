// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Static extensions for data preview.
    /// </summary>
    public static class DebuggerExtensions
    {
        /// <summary>
        /// Extract a 'head' of the data view in a view that is convenient to debug.
        /// </summary>
        /// <param name="data">The data view to preview</param>
        /// <param name="maxRows">Maximum number of rows to pull</param>
        public static DataDebuggerPreview Preview(this IDataView data, int maxRows = DataDebuggerPreview.Defaults.MaxRows)
            => new DataDebuggerPreview(data, maxRows);

        /// <summary>
        /// Preview an effect of the <paramref name="transformer"/> on a given <paramref name="data"/>.
        /// </summary>
        /// <param name="transformer">The transformer which effect we are previewing</param>
        /// <param name="data">The data view to use for preview</param>
        /// <param name="maxRows">Maximum number of rows to pull</param>
        public static DataDebuggerPreview Preview(this ITransformer transformer, IDataView data, int maxRows = DataDebuggerPreview.Defaults.MaxRows)
            => new DataDebuggerPreview(transformer.Transform(data), maxRows);

        /// <summary>
        /// Preview an effect of the <paramref name="estimator"/> on a given <paramref name="data"/>.
        /// </summary>
        /// <param name="estimator">The estimnator which effect we are previewing</param>
        /// <param name="data">The data view to use for preview</param>
        /// <param name="maxRows">Maximum number of rows to show in preview</param>
        /// <param name="maxTrainingRows">Maximum number of rows to fit the estimator</param>
        public static DataDebuggerPreview Preview(this IEstimator<ITransformer> estimator, IDataView data, int maxRows = DataDebuggerPreview.Defaults.MaxRows,
            int maxTrainingRows = DataDebuggerPreview.Defaults.MaxRows)
        {
            Contracts.CheckValue(estimator, nameof(estimator));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckParam(maxRows >= 0, nameof(maxRows));
            Contracts.CheckParam(maxTrainingRows >= 0, nameof(maxTrainingRows));

            using (var env = new LocalEnvironment(conc: 1))
            {
                var trainData = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeArguments { Count = maxTrainingRows }, data);
                return new DataDebuggerPreview(estimator.Fit(trainData).Transform(data), maxRows);
            }
        }

        /// <summary>
        /// Preview an effect of the <paramref name="reader"/> on a given <paramref name="source"/>.
        /// </summary>
        /// <param name="reader">The data reader to preview</param>
        /// <param name="source">The source to pull the data from</param>
        /// <param name="maxRows">Maximum number of rows to pull</param>
        public static DataDebuggerPreview Preview<TSource>(this IDataReader<TSource> reader, TSource source, int maxRows = DataDebuggerPreview.Defaults.MaxRows)
            => new DataDebuggerPreview(reader.Read(source), maxRows);
    }
}
