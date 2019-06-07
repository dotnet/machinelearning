// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.IO;
using Microsoft.Extensions.EnvironmentAbstractions;
using Microsoft.ML.CLI.Telemetry;

namespace Microsoft.DotNet.Configurer
{
    public class FirstTimeUseNoticeSentinel : IFirstTimeUseNoticeSentinel
    {
        public static readonly string Sentinel = $"{Product.Version}.MLNET.dotnetFirstUseSentinel";

        private readonly IFile _file;
        private readonly IDirectory _directory;

        private string _dotnetUserProfileFolderPath;

        private string SentinelPath => Path.Combine(_dotnetUserProfileFolderPath, Sentinel);

        public FirstTimeUseNoticeSentinel() :
            this(
                CliFolderPathCalculator.DotnetUserProfileFolderPath,
                FileSystemWrapper.Default.File,
                FileSystemWrapper.Default.Directory)
        {
        }

        internal FirstTimeUseNoticeSentinel(string dotnetUserProfileFolderPath, IFile file, IDirectory directory)
        {
            _file = file;
            _directory = directory;
            _dotnetUserProfileFolderPath = dotnetUserProfileFolderPath;
        }

        public bool Exists()
        {
            return _file.Exists(SentinelPath);
        }

        public void CreateIfNotExists()
        {
            if (!Exists())
            {
                if (!_directory.Exists(_dotnetUserProfileFolderPath))
                {
                    _directory.CreateDirectory(_dotnetUserProfileFolderPath);
                }

                _file.CreateEmptyFile(SentinelPath);
            }
        }

        public void Dispose()
        {
        }
    }
}
