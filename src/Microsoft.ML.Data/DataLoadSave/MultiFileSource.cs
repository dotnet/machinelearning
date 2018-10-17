// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Wraps a potentially compound path as an IMultiStreamSource. Expands wild cards and supports
    /// multiple paths separated by +.
    /// </summary>
    public sealed class MultiFileSource : IMultiStreamSource
    {
        private readonly string[] _paths;

        public MultiFileSource(params string[] paths)
        {
            Contracts.CheckValueOrNull(paths);

            if (paths == null || paths.Length == 0)
            {
                _paths = new string[0];
                return;
            }

            // in case of usage from Maml, the paths would be wildcard concatenated in the
            // first string of paths.
            string[] concatenated = paths[0] != null ? StreamUtils.ExpandWildCards(paths[0]) : null;

            if (concatenated != null && concatenated.Length > 1)
            {
                if (paths.Length > 1)
                    throw Contracts.Except($"Pass a single string to the {nameof(MultiFileSource)} constructor, if you are using wildcards.");

                _paths = concatenated;
            }
            else
                _paths = paths;
        }

        public int Count
        {
            get { return _paths.Length; }
        }

        public string GetPathOrNull(int index)
        {
            Contracts.CheckParam(0 <= index && index < Count, nameof(index));
            return _paths[index];
        }

        public Stream Open(int index)
        {
            Contracts.CheckParam(0 <= index && index < Count, nameof(index));

            var path = _paths[index];
            try
            {
                return StreamUtils.OpenInStream(path);
            }
            catch (Exception e)
            {
                throw Contracts.ExceptIO(e, "Could not open file '{0}'. Error is: {1}", path, e.Message);
            }
        }

        public TextReader OpenTextReader(int index)
        {
            return new StreamReader(Open(index));
        }
    }

    /// <summary>
    /// Wraps an <see cref="IFileHandle"/> as an IMultiStreamSource.
    /// </summary>
    public sealed class FileHandleSource : IMultiStreamSource
    {
        private readonly IFileHandle _file;

        public FileHandleSource(IFileHandle file)
        {
            Contracts.CheckValue(file, nameof(file));
            Contracts.CheckParam(file.CanRead, nameof(file), "File handle must be readable");
            _file = file;
        }

        public int Count
        {
            get { return 1; }
        }

        public string GetPathOrNull(int index)
        {
            Contracts.CheckParam(0 <= index && index < Count, nameof(index));
            return null;
        }

        public Stream Open(int index)
        {
            Contracts.CheckParam(0 <= index && index < Count, nameof(index));
            return _file.OpenReadStream();
        }

        public TextReader OpenTextReader(int index)
        {
            return new StreamReader(Open(index));
        }
    }
}
