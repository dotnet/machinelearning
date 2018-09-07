﻿// Licensed to the .NET Foundation under one or more agreements.
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
        internal readonly StaticSchemaShape Shape;

        /// <summary>
        /// Constructor for a block maker.
        /// </summary>
        /// <param name="env">The host environment, stored with this object</param>
        /// <param name="shape">The item holding the name and types as enumerated within
        /// <typeparamref name="TTupleShape"/></param>
        private protected SchemaBearing(IHostEnvironment env, StaticSchemaShape shape)
        {
            Contracts.AssertValue(env);
            env.AssertValue(shape);

            Env = env;
            Shape = shape;
        }

        /// <summary>
        /// Starts a new pipeline, using the output schema of this object. Note that the returned
        /// estimator does not contain this object, but it has its schema informed by <typeparamref name="TTupleShape"/>.
        /// The returned object is an empty estimator, on which a new segment of the pipeline can be created.
        /// </summary>
        /// <returns>An empty estimator with the same shape as the object on which it was created</returns>
        public Estimator<TTupleShape, TTupleShape, ITransformer> MakeNewEstimator()
        {
            var est = new EstimatorChain<ITransformer>();
            return new Estimator<TTupleShape, TTupleShape, ITransformer>(Env, est, Shape, Shape);
        }
    }
}
