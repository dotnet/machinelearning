// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

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
