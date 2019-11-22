// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.Extensions.EnvironmentAbstractions
{
    internal interface IFile
    {
        bool Exists(string path);

        string ReadAllText(string path);

        Stream OpenRead(string path);

        Stream OpenFile(
            string path,
            FileMode fileMode,
            FileAccess fileAccess,
            FileShare fileShare,
            int bufferSize,
            FileOptions fileOptions);

        void CreateEmptyFile(string path);

        void WriteAllText(string path, string content);

        void Move(string source, string destination);

        void Copy(string source, string destination);

        void Delete(string path);
    }
}
