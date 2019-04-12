// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Microsoft.DotNet.Configurer;

namespace Microsoft.DotNet.Cli.Telemetry
{
    public class MlTelemetry
    {
        private bool _firstTimeUse = false;
        private bool _enabled = false;
        private List<string> _parameters = new List<string>();
        private string _command;

        public void SetCommandAndParameters(string command, IEnumerable<string> parameters)
        {
            if(parameters != null)
            {
                _parameters.AddRange(parameters);
            }

            _command = command;
        }

        public void LogAutoTrainMlCommand(string dataFileName, string task, long dataFileSize)
        {
            CheckFistTimeUse();

            if(!_enabled)
            {
                return;
            }

            var telemetry = new Telemetry();

            var fileSizeBucket = Math.Pow(2, Math.Ceiling(Math.Log(dataFileSize, 2)));

            var fileNameHash = string.IsNullOrEmpty(dataFileName) ? string.Empty : Sha256Hasher.Hash(dataFileName);

            var paramString = string.Join(",", _parameters);

            var propertiesToLog = new Dictionary<string, string>
                {
                    { "Command", _command },
                    { "FileSizeBucket", fileSizeBucket.ToString() },
                    { "FileNameHash", fileNameHash },
                    { "CommandLineParametersUsed", paramString },
                    { "LearningTaskType", task }
                };

            telemetry.TrackEvent("mlnet-command", propertiesToLog, new Dictionary<string, double>());
        }

        private void CheckFistTimeUse()
        {
            using (IFirstTimeUseNoticeSentinel firstTimeUseNoticeSentinel = new FirstTimeUseNoticeSentinel())
            {
                // if we're in first time use invocation and there are repeat telemetry calls, don't send telemetry
                if (_firstTimeUse)
                {
                    return;
                }

                _firstTimeUse = !firstTimeUseNoticeSentinel.Exists();

                if (_firstTimeUse)
                {
                    Console.WriteLine(
@"Welcome to the ML.NET CLI!
--------------------------
Learn more about ML.NET CLI: https://aka.ms/mlnet-cli
Use 'dotnet ml --help' to see available commands or visit: https://aka.ms/mlnet-cli-docs

Telemetry
---------
The ML.NET CLI tool collect usage data in order to help us improve your experience.
The data is anonymous and doesn't include personal information or data from your datasets.
You can opt-out of telemetry by setting the MLDOTNET_CLI_TELEMETRY_OPTOUT environment variable to '1' or 'true' using your favorite shell.

Read more about ML.NET CLI Tool telemetry: https://aka.ms/mlnet-cli-telemetry
");

                    firstTimeUseNoticeSentinel.CreateIfNotExists();

                    // since the user didn't yet have a chance to read the above message and decide to opt out,
                    // don't log any telemetry on the first invocation.

                    return;
                }

                _enabled = true;
            }
        }
    }
}