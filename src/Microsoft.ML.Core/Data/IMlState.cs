// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// Dummy interface to allow reference to the AutoMlState object in the C# API (since AutoMlState
    /// has things that reference C# API, leading to circular dependency). Makes state object an opaque
    /// black box to the graph. The macro itself will then case to the concrete type.
    /// </summary>
    public interface IMlState
    {}
}