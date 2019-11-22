// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.DotNet.Configurer
{
    public interface IFileSentinel
    {
        bool Exists();

        void Create();
    }
}
