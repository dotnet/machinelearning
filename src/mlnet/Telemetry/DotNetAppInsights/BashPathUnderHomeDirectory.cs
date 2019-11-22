// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.DotNet.Configurer
{
    public struct BashPathUnderHomeDirectory
    {
        private readonly string _fullHomeDirectoryPath;
        private readonly string _pathRelativeToHome;

        public BashPathUnderHomeDirectory(string fullHomeDirectoryPath, string pathRelativeToHome)
        {
            _fullHomeDirectoryPath =
                fullHomeDirectoryPath ?? throw new ArgumentNullException(nameof(fullHomeDirectoryPath));
            _pathRelativeToHome = pathRelativeToHome ?? throw new ArgumentNullException(nameof(pathRelativeToHome));
        }

        public string PathWithTilde => $"~/{_pathRelativeToHome}";

        public string PathWithDollar => $"$HOME/{_pathRelativeToHome}";

        public string Path => $"{_fullHomeDirectoryPath}/{_pathRelativeToHome}";
    }
}
