// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Creates data source for pipeline based on provided collection of data.
    /// </summary>
    public static class CollectionDataSource
    {
        /// <summary>
        /// Creates pipeline data source. Support shuffle.
        /// </summary>
        public static ILearningPipelineLoader Create<T>(IList<T> data) where T : class
        {
            return new ListDataSource<T>(data);
        }

        /// <summary>
        /// Creates pipeline data source which can't be shuffled.
        /// </summary>
        public static ILearningPipelineLoader Create<T>(IEnumerable<T> data) where T : class
        {
            return new EnumerableDataSource<T>(data);
        }

        private abstract class BaseDataSource<TInput> : ILearningPipelineLoader where TInput : class
        {
            private Data.DataViewReference _dataViewEntryPoint;
            private IDataView _dataView;

            public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
            {
                Contracts.Assert(previousStep == null);
                _dataViewEntryPoint = new Data.DataViewReference();
                var importOutput = experiment.Add(_dataViewEntryPoint);
                return new CollectionDataSourcePipelineStep(importOutput.Data);
            }

            public void SetInput(IHostEnvironment environment, Experiment experiment)
            {
                _dataView = GetDataView(environment);
                environment.CheckValue(_dataView, nameof(_dataView));
                experiment.SetInput(_dataViewEntryPoint.Data, _dataView);
            }

            public abstract IDataView GetDataView(IHostEnvironment environment);
        }

        private class EnumerableDataSource<TInput> : BaseDataSource<TInput> where TInput : class
        {
            private readonly IEnumerable<TInput> _enumerableCollection;

            public EnumerableDataSource(IEnumerable<TInput> collection)
            {
                Contracts.CheckValue(collection, nameof(collection));
                _enumerableCollection = collection;
            }

            public override IDataView GetDataView(IHostEnvironment environment)
            {
                return ComponentCreation.CreateStreamingDataView(environment, _enumerableCollection);
            }
        }

        private class ListDataSource<TInput> : BaseDataSource<TInput> where TInput : class
        {
            private readonly IList<TInput> _listCollection;

            public ListDataSource(IList<TInput> collection)
            {
                Contracts.CheckParamValue(Utils.Size(collection) > 0, collection, nameof(collection), "Must be non-empty");
                _listCollection = collection;
            }

            public override IDataView GetDataView(IHostEnvironment environment)
            {
                return ComponentCreation.CreateDataView(environment, _listCollection);
            }
        }

        private class CollectionDataSourcePipelineStep : ILearningPipelineDataStep
        {
            public CollectionDataSourcePipelineStep(Var<IDataView> data)
            {
                Data = data;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model => null;
        }
    }
}
