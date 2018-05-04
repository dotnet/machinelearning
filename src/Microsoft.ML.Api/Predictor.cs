// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// An opaque 'holder' of the predictor, meant to insulate the user from the internal TLC predictor structure,
    /// which is subject to change.
    /// </summary>
    public sealed class Predictor
    {
        /// <summary>
        /// The actual predictor.
        /// </summary>
        internal readonly IPredictor Pred;

        internal Predictor(IPredictor pred)
        {
            Contracts.AssertValue(pred);
            Pred = pred;
        }

        /// <summary>
        /// A way for the user to extract the predictor object and 'delve into the underworld' of unsupported non-API methods.
        /// This is needed, for instance, to inspect the weights of a predictor programmatically.
        /// The intention is to expose most methods through the API and make usage of this method increasingly unnecessary.
        /// </summary>
        [Obsolete("Welcome adventurous stranger, to the Underdark! By calling the mysterious GetPredictorObject method,"+
            " you have entered a world of shifting realities, where nothing is as it seems. Your code may work today, but"+
            " the churning impermanence of the Underdark means the strong foothold today may be nothing but empty air"+
            " tomorrow. Brace yourself!")]
        public object GetPredictorObject()
        {
            return Pred;
        }
    }
}
