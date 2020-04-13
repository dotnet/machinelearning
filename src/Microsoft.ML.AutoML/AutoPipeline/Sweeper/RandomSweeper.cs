using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Sweeper;

namespace Microsoft.ML.AutoPipeline
{
    internal class RandomSweeper: ISweeper
    {
        private ParameterSet _next;
        private int _maximum;

        private UniformRandomSweeper _uniformSweeper;

        public RandomSweeper(MLContext mlContext, IValueGenerator[] valueGenerators, int maximum = 100)
        {
            _maximum = maximum;
            _uniformSweeper = new UniformRandomSweeper(mlContext, new SweeperBase.OptionsBase(), valueGenerators);
        }

        public ParameterSet Current => _next;

        object IEnumerator.Current => Current;

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

            _maximum -= 1;
            var nextArray = this._uniformSweeper.ProposeSweeps(1);
            _next = nextArray[0];
            return true;
        }

        public void Reset()
        {
            return;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void AddRunHistory(IEnumerable<IRunResult> input, SweeperInput Y)
        {
            throw new NotImplementedException();
        }
    }
}
