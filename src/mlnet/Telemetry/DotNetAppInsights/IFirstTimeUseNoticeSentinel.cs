// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;

namespace Microsoft.DotNet.Configurer
{
    public interface IFirstTimeUseNoticeSentinel : IDisposable
    {
        bool Exists();

        void CreateIfNotExists();
    }
}
