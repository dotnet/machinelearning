// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.DotNet.Cli.Telemetry;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.Utilities;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    internal class MLNetCommandEvent
    {
        public NewCommandSettings AutoTrainCommandSettings { get; set; }
        public IEnumerable<string> CommandLineParametersUsed { get; set; }

        public void TrackEvent()
        {
            Telemetry.TrackEvent("mlnet-command",
                new Dictionary<string, string>
                {
                    { "Cache", Utils.GetCacheSettings(AutoTrainCommandSettings.Cache).ToString() },
                    { "CommandLineParametersUsed", string.Join(",", CommandLineParametersUsed) },
                    { "FilenameHash", HashFilename(AutoTrainCommandSettings.Dataset.Name) },
                    { "FileSizeBucket", GetFileSizeBucketStr(AutoTrainCommandSettings.Dataset) },
                    { "HasHeader", AutoTrainCommandSettings.HasHeader.ToString() },
                    { "IgnoredColumnsCount", AutoTrainCommandSettings.IgnoreColumns.Count.ToString() },
                    { "LearningTaskType", AutoTrainCommandSettings.MlTask },
                    { "MaxExplorationTime", AutoTrainCommandSettings.MaxExplorationTime.ToString() },
                    { "ValidFilenameHash", HashFilename(AutoTrainCommandSettings.ValidationDataset?.Name) },
                    { "ValidFileSizeBucket", GetFileSizeBucketStr(AutoTrainCommandSettings.ValidationDataset) },
                    { "TestFilenameHash", HashFilename(AutoTrainCommandSettings.TestDataset?.Name) },
                    { "TestFileSizeBucket", GetFileSizeBucketStr(AutoTrainCommandSettings.TestDataset) },
                });
        }

        private static string HashFilename(string filename)
        {
            return string.IsNullOrEmpty(filename) ? null : Sha256Hasher.Hash(filename);
        }

        private static double CalcFileSizeBucket(FileInfo fileInfo)
        {
            return Math.Pow(2, Math.Ceiling(Math.Log(fileInfo.Length, 2)));
        }

        private static string GetFileSizeBucketStr(FileInfo fileInfo)
        {
            if (fileInfo == null || !fileInfo.Exists)
            {
                return null;
            }
            return CalcFileSizeBucket(fileInfo).ToString();
        }
    }
}
