// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.ML.CLI.Utilities.File
{
    internal class SystemFileInfo : IFileInfo
    {
        public string FullName => _fileInfo.FullName;

        private readonly FileInfo _fileInfo;

        public SystemFileInfo(FileInfo fileInfo)
        {
            _fileInfo = fileInfo;
        }
    }
}
