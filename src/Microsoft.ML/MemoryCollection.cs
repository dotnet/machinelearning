using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

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
            //need validation at some point
            _listCollection = collection;
        }

        public MemoryCollection(IEnumerable<TInput> collection)
        {
            //need validation at some point
            _enumerableCollection = collection;
        }

        public void SetInput(IHostEnvironment env, Experiment experiment)
        {
            if (_listCollection!=null)
            {
                _dataView = DataViewConstructionUtils.CreateFromList(env, _listCollection);
            }
            if (_enumerableCollection!=null)
            {
                _dataView = DataViewConstructionUtils.CreateFromEnumerable(env, _listCollection);
            }
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
