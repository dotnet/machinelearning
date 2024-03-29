﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Fairlearn
{
    /// <summary>
    /// Class containing AutoML extension methods to <see cref="MLContext"/>
    /// </summary>
    public static class MLContextExtension
    {
        /// <summary>
        /// Returns a catalog of all possible Fairlearn operations.
        /// </summary>
        /// <param name="mlContext"><see cref="MLContext"/> instance.</param>
        /// <returns>A catalog of all possible AutoML operations.</returns>
        public static FairlearnCatalog Fairlearn(this MLContext mlContext)
        {
            return new FairlearnCatalog(mlContext);
        }
    }
}
