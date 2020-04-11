using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal class GridSearchSweeper : ISweeper
    {
        private readonly Dictionary<string, ParameterAttribute> _parameters;
        private SweeperOutput _next;
        private IEnumerable<Dictionary<string, int>> _gridSearcher;

        public GridSearchSweeper(Dictionary<string, ParameterAttribute> parameters)
        {
            _parameters = parameters;
            Reset();
            _gridSearcher = GetGridSearcher();
        }

        private IEnumerable<Dictionary<string, int>> GetGridSearcher()
        {
            throw new NotImplementedException();
        }

        public SweeperOutput Current => _next;

        object IEnumerator.Current => _next;

        public void Dispose()
        {
            return;
        }

        public void Fit(SweeperOutput input, SweeperInput Y)
        {
            return;
        }

        public IEnumerator<SweeperOutput> GetEnumerator()
        {
            return this;
        }

        public bool MoveNext()
        {
            return false;
        }

        public void Reset()
        {
            return;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }
    }
}
