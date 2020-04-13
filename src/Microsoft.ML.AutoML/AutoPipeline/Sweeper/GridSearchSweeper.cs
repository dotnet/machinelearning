using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Sweeper;

namespace Microsoft.ML.AutoPipeline
{
    internal class GridSearchSweeper : ISweeper
    {
        private ParameterSet _next;
        private ParameterSet[] _results;
        private readonly RandomGridSweeper _gridSweeper;
        private int _maximum;

        public GridSearchSweeper(MLContext context, IValueGenerator[] valueGenerators, int maximum = 10000)
        {
            var option = new RandomGridSweeper.Options();
            _maximum = maximum;
            _gridSweeper = new RandomGridSweeper(context, option, valueGenerators);
            _results = _gridSweeper.ProposeSweeps(maximum);
        }

        public ParameterSet Current => _next;

        object IEnumerator.Current => _next;

        public void Dispose()
        {
            return;
        }

        public IEnumerator<ParameterSet> GetEnumerator()
        {
            return this;
        }

        public bool MoveNext()
        {
            if(_maximum <= 0)
            {
                return false;
            }
            _next = _results[_maximum-1];
            _maximum -= 1;
            return true;
        }

        public void Reset()
        {
            return;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }

        public void AddRunHistory(IEnumerable<IRunResult> input, SweeperInput Y)
        {
            throw new NotImplementedException();
        }
    }
}
