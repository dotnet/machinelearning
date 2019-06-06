// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

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
