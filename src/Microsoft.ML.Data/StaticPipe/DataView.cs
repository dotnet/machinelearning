// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe.Runtime;
using System.Collections.Generic;
using System;
using System.Linq;

namespace Microsoft.ML.StaticPipe
{
    public class DataView<TShape> : SchemaBearing<TShape>
    {
        public IDataView AsDynamic { get; }

        internal DataView(IHostEnvironment env, IDataView view, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(view);

            AsDynamic = view;
            Shape.Check(Env, AsDynamic.Schema);
        }

        /// <summary>
        /// This function return a <see cref="DataView{TShape}"/> whose columns are all cached in memory.
        /// This returned <see cref="DataView{TShape}"/> is almost the same to the source <see cref="DataView{TShape}"/>.
        /// The only difference are cache-related properties.
        /// </summary>
        public DataView<TShape> Cache()
        {
            // Generate all column indexes in the source data.
            var prefetched = Enumerable.Range(0, AsDynamic.Schema.ColumnCount).ToArray();
            // Create a cached version of the source data by caching all columns.
            return new DataView<TShape>(Env, new CacheDataView(Env, AsDynamic, prefetched), Shape);
        }
    }

    public static class DataViewExtensions
    {
        private static IEnumerable<TOut> GetColumnCore<TOut, TShape>(DataView<TShape> data, Func<TShape, PipelineColumn> column)
        {
            Contracts.CheckValue(data, nameof(data));
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckValue(column, nameof(column));

            var indexer = StaticPipeUtils.GetIndexer(data);
            string columnName = indexer.Get(column(indexer.Indices));

            return data.AsDynamic.GetColumn<TOut>(env, columnName);
        }

        public static IEnumerable<TItem> GetColumn<TItem, TShape>(this DataView<TShape> data, Func<TShape, Scalar<TItem>> column)
            => GetColumnCore<TItem, TShape>(data, column);

        public static IEnumerable<TItem[]> GetColumn<TItem, TShape>(this DataView<TShape> data, Func<TShape, Vector<TItem>> column)
            => GetColumnCore<TItem[], TShape>(data, column);

        public static IEnumerable<TItem[]> GetColumn<TItem, TShape>(this DataView<TShape> data, Func<TShape, VarVector<TItem>> column)
            => GetColumnCore<TItem[], TShape>(data, column);
    }
}
