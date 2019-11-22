// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.Extensions.EnvironmentAbstractions
{
    internal interface IDirectory
    {
        bool Exists(string path);

        ITemporaryDirectory CreateTemporaryDirectory();

        IEnumerable<string> EnumerateFiles(string path);

        IEnumerable<string> EnumerateFileSystemEntries(string path);

        string GetCurrentDirectory();

        void CreateDirectory(string path);

        void Delete(string path, bool recursive);

        void Move(string source, string destination);
    }
}
