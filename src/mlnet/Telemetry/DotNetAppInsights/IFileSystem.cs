// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

namespace Microsoft.Extensions.EnvironmentAbstractions
{
    internal interface IFileSystem
    {
        IFile File { get; }
        IDirectory Directory { get; }
    }
}
