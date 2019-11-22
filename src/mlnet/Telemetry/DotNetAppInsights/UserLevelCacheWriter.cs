// Licensed to the .NET Foundation under one or more agreements.\r
// The .NET Foundation licenses this file to you under the MIT license.\r
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Extensions.EnvironmentAbstractions;
using Microsoft.ML.CLI.Telemetry;

namespace Microsoft.DotNet.Configurer
{
    public class UserLevelCacheWriter : IUserLevelCacheWriter
    {
        private readonly IFile _file;
        private readonly IDirectory _directory;
        private string _dotnetUserProfileFolderPath;

        public UserLevelCacheWriter() :
            this(
                CliFolderPathCalculator.DotnetUserProfileFolderPath,
                FileSystemWrapper.Default.File,
                FileSystemWrapper.Default.Directory)
        {
        }

        public string RunWithCache(string cacheKey, Func<string> getValueToCache)
        {
            var cacheFilepath = GetCacheFilePath(cacheKey);
            try
            {
                if (!_file.Exists(cacheFilepath))
                {
                    if (!_directory.Exists(_dotnetUserProfileFolderPath))
                    {
                        _directory.CreateDirectory(_dotnetUserProfileFolderPath);
                    }

                    var runResult = getValueToCache();

                    _file.WriteAllText(cacheFilepath, runResult);
                    return runResult;
                }
                else
                {
                    return _file.ReadAllText(cacheFilepath);
                }
            }
            catch (Exception ex)
            {
                if (ex is UnauthorizedAccessException
                    || ex is PathTooLongException
                    || ex is IOException)
                {
                    return getValueToCache();
                }

                throw;
            }

        }

        internal UserLevelCacheWriter(string dotnetUserProfileFolderPath, IFile file, IDirectory directory)
        {
            _file = file;
            _directory = directory;
            _dotnetUserProfileFolderPath = dotnetUserProfileFolderPath;
        }

        private string GetCacheFilePath(string cacheKey)
        {
            return Path.Combine(_dotnetUserProfileFolderPath, $"{Product.Version}_{cacheKey}.dotnetUserLevelCache");
        }
    }
}
