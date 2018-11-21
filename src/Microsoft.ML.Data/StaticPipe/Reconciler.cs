// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.StaticPipe.Runtime
{
    /// <summary>
    /// An object for <see cref="PipelineColumn"/> instances to indicate to the analysis code for static pipelines that
    /// they should be considered a single group of columns (through equality on the reconcilers), as well as how to
    /// actually create the underlying dynamic structures, whether an <see cref="IDataReaderEstimator{TSource, TReader}"/>
    /// (for the <see cref="ReaderReconciler{TREaderIn}"/>) or a <see cref="IEstimator{TTransformer}"/>
    /// (for the <see cref="EstimatorReconciler"/>).
    /// </summary>
    public abstract class Reconciler
    {
        private protected Reconciler() { }
    }

    /// <summary>
    /// Reconciler for column groups intended to resolve to a new <see cref="IDataReaderEstimator{TSource, TReader}"/>
    /// or <see cref="IDataReader{TSource}"/>.
    /// </summary>
    /// <typeparam name="TIn">The input type of the <see cref="IDataReaderEstimator{TSource, TReader}"/>
    /// object.</typeparam>
    public abstract class ReaderReconciler<TIn> : Reconciler
    {
        public ReaderReconciler() : base() { }

        /// <summary>
        /// Returns a data-reader estimator. Note that there are no input names because the columns from a data-reader
        /// estimator should have no dependencies.
        /// </summary>
        /// <param name="env">The host environment to use to create the data-reader estimator</param>
        /// <param name="toOutput">The columns that the object created by the reconciler should output</param>
        /// <param name="outputNames">A map containing</param>
        /// <returns></returns>
        public abstract IDataReaderEstimator<TIn, IDataReader<TIn>> Reconcile(
            IHostEnvironment env, PipelineColumn[] toOutput, IReadOnlyDictionary<PipelineColumn, string> outputNames);
    }

    /// <summary>
    /// Reconciler for column groups intended to resolve to an <see cref="IEstimator{TTransformer}"/>. This type of
    /// reconciler will work with <see cref="Estimator{TInShape, TOutShape, TTransformer}.Append{TNewOutShape}(Func{TOutShape, TNewOutShape})"/>
    /// or other methods that involve the creation of estimator chains.
    /// </summary>
    public abstract class EstimatorReconciler : Reconciler
    {
        public EstimatorReconciler() : base() { }

        /// <summary>
        /// Returns an estimator.
        /// </summary>
        /// <param name="env">The host environment to use to create the estimator</param>
        /// <param name="toOutput">The columns that the object created by the reconciler should output</param>
        /// <param name="inputNames">The name mapping that maps dependencies of the output columns to their names</param>
        /// <param name="outputNames">The name mapping that maps the output column to their names</param>
        /// <param name="usedNames">While most estimators allow full control over the names of their outputs, a limited
        /// subset of estimator transforms do not allow this: they produce columns whose names are unconfigurable. For
        /// these, there is this collection which provides the names used by the analysis tool. If the estimator under
        /// construction must use one of the names here, then they are responsible for "saving" the column they will
        /// overwrite using applications of the <see cref="ColumnsCopyingEstimator"/>. Note that if the estimator under
        /// construction has complete control over what columns it produces, there is no need for it to pay this argument
        /// any attention.</param>
        /// <returns>Returns an estimator.</returns>
        public abstract IEstimator<ITransformer> Reconcile(
            IHostEnvironment env,
            PipelineColumn[] toOutput,
            IReadOnlyDictionary<PipelineColumn, string> inputNames,
            IReadOnlyDictionary<PipelineColumn, string> outputNames,
            IReadOnlyCollection<string> usedNames);
    }
}
