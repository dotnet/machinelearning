// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML
{
    public class MemoryCollection<TInput> : ILearningPipelineLoader
        where TInput : class
    {
        public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
        {
            Contracts.Assert(previousStep == null);
            _dataViewEntryPoint = new Data.DataViewReference();
            var importOutput = experiment.Add(_dataViewEntryPoint);
            return new MemoryCollectionPipelineStep(importOutput.Data);
        }

        private readonly IList<TInput> _listCollection;
        private readonly IEnumerable<TInput> _enumerableCollection;

        private Data.DataViewReference _dataViewEntryPoint;
        private IDataView _dataView;

        public MemoryCollection(IList<TInput> collection)
        {
            Contracts.CheckParamValue(Utils.Size(collection) > 0, collection, nameof(collection), "Must be non-empty");
            _listCollection = collection;
        }

        public MemoryCollection(IEnumerable<TInput> collection)
        {
            Contracts.CheckParamValue(collection != null, collection, nameof(collection), "Must be non-null");
            _enumerableCollection = collection;
        }

        public void SetInput(IHostEnvironment env, Experiment experiment)
        {
            if (_listCollection != null)
                _dataView = ComponentCreation.CreateDataView(env, _listCollection);
            if (_enumerableCollection != null)
                _dataView = ComponentCreation.CreateStreamingDataView(env, _listCollection);
            env.CheckValue(_dataView, nameof(_dataView));
            experiment.SetInput(_dataViewEntryPoint.Data, _dataView);
        }

        private class MemoryCollectionPipelineStep : ILearningPipelineDataStep
        {
            public MemoryCollectionPipelineStep(Var<IDataView> data)
            {
                Data = data;
                Model = null;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model { get; }
        }
    }
}
