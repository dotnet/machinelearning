using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;

namespace Tensorflow
{
    public class optimizer
    {
        public static _OptimizableVariable _get_processor(RefVariable v)
        {
            return new _RefVariableProcessor(v);
        }
    }

    public class _RefVariableProcessor : _OptimizableVariable
    {
        private RefVariable _v;

        public _RefVariableProcessor(RefVariable v)
        {
            _v = v;
        }

        public Tensor target()
        {
            return _v._ref();
        }

        public Operation update_op(Optimizer optimizer, Tensor g)
        {
            Operation update_op = null;

            if (g.Tag == null)
            {
                update_op = optimizer._apply_dense(g, _v);
            }
            else if (g.Tag is IndexedSlices)
            {
                return optimizer._apply_sparse_duplicate_indices(g, _v);
            }

            return update_op;
        }
    }
}
