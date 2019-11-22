// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.Extensions.EnvironmentAbstractions
{
    internal class FileSystemWrapper : IFileSystem
    {
        public static IFileSystem Default { get; } = new FileSystemWrapper();

        public IFile File { get; }

        public IDirectory Directory { get; }

        public FileSystemWrapper()
        {
            File = new FileWrapper();
            Directory = new DirectoryWrapper();
        }
    }
}
