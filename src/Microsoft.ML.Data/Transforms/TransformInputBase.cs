// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The base class for all transform inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public abstract class TransformInputBase
    {
        private protected TransformInputBase() { }

        /// <summary>
        /// The input dataset. Used only in entry-point methods, since the normal API mechanism for feeding in a dataset to
        /// create an <see cref="ITransformer"/> is to use the <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> method.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.Required, HelpText = "Input dataset", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, SortOrder = 1)]
        internal IDataView Data;
    }
}
