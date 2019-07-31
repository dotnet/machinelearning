// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Torch;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class TorchCatalog
    {
        /// <summary>
        /// Load <a href="https://pytorch.org/docs/stable/jit.html">TorchScript</a> model into memory.
        /// This is the convenience method that allows the model to be loaded once and subsequently used to create a
        /// <see cref="TorchScorerEstimator"/> using <see cref="TorchModel.ScoreTorchModel(string, long[], string)"/>.
        /// </summary>
        /// <remarks>
        /// A standard PyTorch model defined and trained in python needs to be exported to <a href="https://pytorch.org/docs/stable/jit.html">TorchScript</a>
        /// in order to be executed in ML.NET. Exporting a PyTorch model to <a href="https://pytorch.org/docs/stable/jit.html">TorchScript</a> can be done
        /// through scripting or tracing. See <a href="https://pytorch.org/docs/stable/jit.html#creating-torchscript-code">here</a> for more
        /// information about scripting and tracing PyTorch models.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TorchScript model.</param>
        public static TorchModel LoadTorchModel(this ModelOperationsCatalog catalog, string modelLocation)
            => TorchUtils.LoadTorchModel(CatalogUtils.GetEnvironment(catalog), modelLocation);
    }
}
