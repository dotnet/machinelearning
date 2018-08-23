// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Data.StaticPipe.Runtime
{
    /// <summary>
    /// An object for <see cref="PipelineColumn"/> instances to indicate to the analysis code for static pipelines that
    /// they should be considered a single group of columns (through equality on the reconcilers), as well as how to
    /// actually create the underlying dynamic structures, whether an <see cref="IDataReaderEstimator{TSource, TReader}"/>
    /// (for the <see cref="ReaderReconciler{TREaderIn}"/>) or a <see cref="IEstimator{TTransformer}"/>
    /// (for the <see cref="DataInputReconciler"/>).
    /// </summary>
    public abstract class Reconciler
    {
        private protected Reconciler() { }
    }

    /// <summary>
    /// Reconciler for column groups intended to resolve to a new <see cref="IDataReaderEstimator{TSource, TReader}"/>.
    /// </summary>
    /// <typeparam name="TREaderIn">The type of object returned through the reconciler. This is intended to
    /// be either an <see cref="IEstimator{TTransformer}"/> or <see cref="IDataReaderEstimator{TSource, TReader}"/>
    /// object.</typeparam>
    public abstract class ReaderReconciler<TREaderIn> : Reconciler
    {
        public ReaderReconciler() : base() { }

        /// <summary>
        /// Returns a data-reader estimator. Note that there are no input names because the columns from a data-reader
        /// estimator should have no dependencies.
        /// </summary>
        /// <param name="toOutput">The columns that the reconciler should output</param>
        /// <param name="outputNames"></param>
        /// <returns></returns>
        public abstract IDataReaderEstimator<TREaderIn, IDataReader<TREaderIn>> Reconcile(
            PipelineColumn[] toOutput, Dictionary<PipelineColumn, string> outputNames);
    }

    /// <summary>
    /// Reconciler for column groups intended to resolve to an <see cref="IEstimator{TTransformer}"/>. This type of
    /// reconciler will work with <see cref="BlockMaker{TTupleShape}.CreateTransform{TTupleOutShape}(Func{TTupleShape, TTupleOutShape})"/>
    /// or other functions that involve the creation of transforms.
    /// </summary>
    public abstract class DataInputReconciler : Reconciler
    {
        public DataInputReconciler() : base() { }

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
