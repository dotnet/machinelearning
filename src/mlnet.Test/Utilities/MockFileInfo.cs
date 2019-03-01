// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CLI.Utilities.File;

namespace mlnet.Test.Utilities
{
    internal class MockFileInfo : IFileInfo
    {
        public string FullName { get; }

        public MockFileInfo(string filePath)
        {
            FullName = filePath;
        }
    }
}
