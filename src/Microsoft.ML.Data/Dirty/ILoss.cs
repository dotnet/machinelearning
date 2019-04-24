// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    public interface ILossFunction<in TOutput, in TLabel>
    {
        /// <summary>
        /// Computes the loss given the output and the ground truth.
        /// Note that the return value has type Double because the loss is usually accumulated over many instances.
        /// </summary>
        Double Loss(TOutput output, TLabel label);
    }

    public interface IScalarLoss : ILossFunction<float, float>
    {
        /// <summary>
        /// Derivative of the loss function with respect to output
        /// </summary>
        float Derivative(float output, float label);
    }

    [TlcModule.ComponentKind("RegressionLossFunction")]
    [BestFriend]
    internal interface ISupportRegressionLossFactory : IComponentFactory<IRegressionLoss>
    {
    }

    public interface IRegressionLoss : IScalarLoss
    {
    }

    [TlcModule.ComponentKind("ClassificationLossFunction")]
    [BestFriend]
    internal interface ISupportClassificationLossFactory : IComponentFactory<IClassificationLoss>
    {
    }

    public interface IClassificationLoss : IScalarLoss
    {
    }

    /// <summary>
    /// Delegate signature for standardized classification loss functions.
    /// </summary>
    internal delegate void SignatureClassificationLoss();

    /// <summary>
    /// Delegate signature for standardized regression loss functions.
    /// </summary>
    internal delegate void SignatureRegressionLoss();
}
