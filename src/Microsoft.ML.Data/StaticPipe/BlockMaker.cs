// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public abstract class BlockMaker<TTupleShape>
    {
        public Estimator<TTupleShape, TTupleOutShape, ITransformer> CreateTransform<TTupleOutShape>(Func<TTupleShape, TTupleOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            Console.WriteLine($"Called {nameof(CreateTransform)} !!!");

            var method = mapper.Method;

            // Construct the dummy column structure, then apply the mapping.
            var input = PipelineColumnAnalyzer.CreateAnalysisInstance<TTupleShape>(out var fakeReconciler);
            KeyValuePair<string, PipelineColumn>[] inPairs = PipelineColumnAnalyzer.GetNames(input, method.GetParameters()[0]);

            // Initially we suppose we've only assigned names to the inputs.
            var inputColToName = new Dictionary<PipelineColumn, string>();
            foreach (var p in inPairs)
                inputColToName[p.Value] = p.Key;
            string NameMap(PipelineColumn col)
            {
                inputColToName.TryGetValue(col, out var val);
                return val;
            }

            StaticPipeUtils.GeneralFunctionAnalyzer(input, fakeReconciler, mapper, out var readerEst, out var est, NameMap);
            Contracts.Assert(readerEst == null);
            Contracts.AssertValue(est);

            Console.WriteLine($"Exiting {nameof(CreateTransform)} !!!");

            return new FakeEstimator<TTupleOutShape>();
        }

        private sealed class FakeEstimator<TTupleOutShape>
            : Estimator<TTupleShape, TTupleOutShape, ITransformer>
        {
            protected override ITransformer FitCore(IDataView input)
            {
                throw new NotImplementedException();
            }

            protected override SchemaShape GetOutputSchemaCore(SchemaShape inputSchema)
            {
                throw new NotImplementedException();
            }
        }
    }

    internal sealed class InvDictionary<T1, T2>
    {
        private readonly Dictionary<T1, T2> _d12;
        private readonly Dictionary<T2, T1> _d21;

        public InvDictionary()
        {
            _d12 = new Dictionary<T1, T2>();
            _d21 = new Dictionary<T2, T1>();
        }

        public bool ContainsKey(T1 k) => _d12.ContainsKey(k);
        public bool ContainsKey(T2 k) => _d21.ContainsKey(k);

        public bool TryGetValue(T1 k, out T2 v) => _d12.TryGetValue(k, out v);
        public bool TryGetValue(T2 k, out T1 v) => _d21.TryGetValue(k, out v);

        public T1 this[T2 key]
        {
            get => _d21[key];
            set {
                Contracts.CheckValue((object)key, nameof(key));
                Contracts.CheckValue((object)value, nameof(value));

                bool removeOldKey = _d12.TryGetValue(value, out var oldKey);
                if (_d21.TryGetValue(key, out var oldValue))
                    _d12.Remove(oldValue);
                if (removeOldKey)
                    _d21.Remove(oldKey);

                _d12[value] = key;
                _d21[key] = value;
                Contracts.Assert(_d12.Count == _d21.Count);
            }
        }

        public T2 this[T1 key]
        {
            get => _d12[key];
            set {
                Contracts.CheckValue((object)key, nameof(key));
                Contracts.CheckValue((object)value, nameof(value));

                bool removeOldKey = _d21.TryGetValue(value, out var oldKey);
                if (_d12.TryGetValue(key, out var oldValue))
                    _d21.Remove(oldValue);
                if (removeOldKey)
                    _d12.Remove(oldKey);

                _d21[value] = key;
                _d12[key] = value;

                Contracts.Assert(_d12.Count == _d21.Count);
            }
        }

        public Dictionary<T1, T2> AsOther(IEnumerable<T1> keys)
        {
            Dictionary<T1, T2> d = new Dictionary<T1, T2>();
            foreach (var v in keys)
                d[v] = _d12[v];
            return d;
        }

        public Dictionary<T2, T1> AsOther(IEnumerable<T2> keys)
        {
            Dictionary<T2, T1> d = new Dictionary<T2, T1>();
            foreach (var v in keys)
                d[v] = _d21[v];
            return d;
        }
    }

}
