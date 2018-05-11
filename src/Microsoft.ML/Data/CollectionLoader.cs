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
    public class CollectionLoader
    {
        /// <summary>
        /// Creates pipeline loader. Support shuffle.
        /// </summary>
        public static ILearningPipelineLoader Create<T>(IList<T> data) where T : class
        {
            return new ListCollectionLoader<T>(data);
        }

        /// <summary>
        /// Creates pipeline loader which can't be shuffled.
        /// </summary>
        public static ILearningPipelineLoader Create<T>(IEnumerable<T> data) where T : class
        {
            return new EnumerableCollectionLoader<T>(data);
        }

        private abstract class BaseCollectionLoader<TInput> : ILearningPipelineLoader where TInput : class
        {
            private Data.DataViewReference _dataViewEntryPoint;
            private IDataView _dataView;

            public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
            {
                Contracts.Assert(previousStep == null);
                _dataViewEntryPoint = new Data.DataViewReference();
                var importOutput = experiment.Add(_dataViewEntryPoint);
                return new CollectionLoaderPipelineStep(importOutput.Data);
            }

            public void SetInput(IHostEnvironment environment, Experiment experiment)
            {
                _dataView = GetDataView(environment);
                environment.CheckValue(_dataView, nameof(_dataView));
                experiment.SetInput(_dataViewEntryPoint.Data, _dataView);
            }

            public abstract IDataView GetDataView(IHostEnvironment environment);
        }

        private class EnumerableCollectionLoader<TInput> : BaseCollectionLoader<TInput> where TInput : class
        {
            private readonly IEnumerable<TInput> _enumerableCollection;

            public EnumerableCollectionLoader(IEnumerable<TInput> collection)
            {
                Contracts.CheckValue(collection, nameof(collection));
                _enumerableCollection = collection;
            }

            public override IDataView GetDataView(IHostEnvironment environment)
            {
                return ComponentCreation.CreateStreamingDataView(environment, _enumerableCollection);
            }
        }

        private class ListCollectionLoader<TInput> : BaseCollectionLoader<TInput> where TInput : class
        {
            private readonly IList<TInput> _listCollection;

            public ListCollectionLoader(IList<TInput> collection)
            {
                Contracts.CheckParamValue(Utils.Size(collection) > 0, collection, nameof(collection), "Must be non-empty");
                _listCollection = collection;
            }

            public override IDataView GetDataView(IHostEnvironment environment)
            {
                return ComponentCreation.CreateDataView(environment, _listCollection);
            }
        }

        private class CollectionLoaderPipelineStep : ILearningPipelineDataStep
        {
            public CollectionLoaderPipelineStep(Var<IDataView> data)
            {
                Data = data;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model => null;
        }
    }
}
