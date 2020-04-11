using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal class RandomSweeper: ISweeper
    {
        private readonly Dictionary<string, ParameterAttribute> _parameters;
        private SweeperOutput _next;
        private Random _rand;
        private int maximum;

        public RandomSweeper(Dictionary<string, ParameterAttribute> parameters, int maximum = 100)
        {
            _parameters = parameters;
            this.maximum = maximum;
            Reset();
        }

        public SweeperOutput Current => _next;

        object IEnumerator.Current => Current;

        public void Dispose()
        {
            return;
        }

        public IEnumerator<SweeperOutput> GetEnumerator()
        {
            return this;
        }

        public void Fit(SweeperOutput input, SweeperInput Y)
        {
            return;
        }

        public bool MoveNext()
        {
            if (_parameters is null || maximum <0)
            {
                return false;
            }

            maximum -= 1;
            _next = new SweeperOutput();
            foreach(var kv in _parameters)
            {
                var i = _rand.Next() % kv.Value.Value.Count;
                object value = kv.Value.Value[i];
                _next.Add(kv.Key, value);
            }
            return true;
        }

        public void Reset()
        {
            _rand = new Random();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
