// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.Extensions.EnvironmentAbstractions
{
    internal class FileWrapper: IFile
    {
        public bool Exists(string path)
        {
            return File.Exists(path);
        }

        public string ReadAllText(string path)
        {
            return File.ReadAllText(path);
        }

        public Stream OpenRead(string path)
        {
            return File.OpenRead(path);
        }

        public Stream OpenFile(
            string path,
            FileMode fileMode,
            FileAccess fileAccess,
            FileShare fileShare,
            int bufferSize,
            FileOptions fileOptions)
        {
            return new FileStream(path, fileMode, fileAccess, fileShare, bufferSize, fileOptions);
        }

        public void CreateEmptyFile(string path)
        {
            using (File.Create(path))
            {
            }
        }

        public void WriteAllText(string path, string content)
        {
            File.WriteAllText(path, content);
        }

        public void Move(string source, string destination)
        {
            File.Move(source, destination);
        }

        public void Copy(string sourceFileName, string destFileName)
        {
            File.Copy(sourceFileName, destFileName);
        }

        public void Delete(string path)
        {
            File.Delete(path);
        }
    }
}
