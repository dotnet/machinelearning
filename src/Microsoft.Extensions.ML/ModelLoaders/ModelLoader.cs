// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    public abstract class ModelLoader
    {
        public abstract IChangeToken GetReloadToken();

        public abstract ITransformer GetModel();
    }
}
