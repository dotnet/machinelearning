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
    /// <summary>
    /// A base class for the statically-typed pipeline components, that are marked as producing
    /// data whose schema has a certain shape.
    /// </summary>
    /// <typeparam name="TTupleShape"></typeparam>
    public abstract class SchemaBearing<TTupleShape>
    {
        private protected readonly IHostEnvironment Env;

        /// <summary>
        /// Constructor for a block maker.
        /// </summary>
        /// <param name="env"></param>
        private protected SchemaBearing(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            Env = env;
        }

        /// <summary>
        /// Create an object that can be used as the start of a new pipeline, that assumes it uses
        /// something with the sahape of <typeparamref name="TTupleShape"/> as its input schema shape.
        /// The returned object is an empty estimator.
        /// </summary>
        internal Estimator<TTupleShape, TTupleShape, ITransformer> MakeNewEstimator()
        {
            var est = new EstimatorChain<ITransformer>();
            return new Estimator<TTupleShape, TTupleShape, ITransformer>(Env, est);
        }
    }
}
