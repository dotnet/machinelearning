// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Extensions.EnvironmentAbstractions;

namespace Microsoft.DotNet.InternalAbstractions
{
    internal class TemporaryDirectory : ITemporaryDirectory
    {
        public string DirectoryPath { get; }

        public TemporaryDirectory()
        {
            DirectoryPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            Directory.CreateDirectory(DirectoryPath);
        }

        public void Dispose()
        {
            try
            {
                Directory.Delete(DirectoryPath, true);
            }
            catch
            {
                // Ignore failures here.
            }
        }
    }
}
