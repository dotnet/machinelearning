// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime
{
    public interface ILossFunction<in TOutput, in TLabel>
    {
        /// <summary>
        /// Computes the loss given the output and the ground truth.
        /// Note that the return value has type Double because the loss is usually accumulated over many instances.
        /// </summary>
        Double Loss(TOutput output, TLabel label);
    }

    public interface IScalarOutputLoss : ILossFunction<Float, Float>
    {
        /// <summary>
        /// Derivative of the loss function with respect to output
        /// </summary>
        Float Derivative(Float output, Float label);
    }

    [TlcModule.ComponentKind("RegressionLossFunction")]
    public interface ISupportRegressionLossFactory : IComponentFactory<IRegressionLoss>
    {
    }

    public interface IRegressionLoss : IScalarOutputLoss
    {
    }

    [TlcModule.ComponentKind("ClassificationLossFunction")]
    public interface ISupportClassificationLossFactory : IComponentFactory<IClassificationLoss>
    {
    }

    public interface IClassificationLoss : IScalarOutputLoss
    {
    }

    /// <summary>
    /// Delegate signature for standardized classification loss functions.
    /// </summary>
    public delegate void SignatureClassificationLoss();

    /// <summary>
    /// Delegate signature for standardized regression loss functions.
    /// </summary>
    public delegate void SignatureRegressionLoss();
}
