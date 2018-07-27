using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Core.StrongPipe.Columns
{
    public abstract class Reconciler
    {
    }

    /// <summary>
    /// REconciler for things intended to work with <see cref="BlockMaker{TTupleShape}.CreateTransform{TTupleOutShape}(Func{TTupleShape, TTupleOutShape})"/>.
    /// </summary>
    public abstract class DataInputReconciler : Reconciler
    {
        /// <summary>
        /// Returns an estimator.
        /// </summary>
        /// <param name="toOutput"></param>
        /// <param name="inputNames"></param>
        /// <param name="outputNames"></param>
        /// <returns></returns>
        public abstract IEstimator<ITransformer> Reconcile(
            PipelineColumn[] toOutput,
            Dictionary<PipelineColumn, string> inputNames,
            Dictionary<PipelineColumn, string> outputNames);
    }
}
