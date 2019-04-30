// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.DotNet.Cli.Telemetry;
using Microsoft.ML.CLI.CodeGenerator;
using Microsoft.ML.CLI.Data;
using NLog;

namespace Microsoft.ML.CLI.Commands.New
{
    internal class NewCommand : ICommand
    {
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private readonly NewCommandSettings settings;
        private readonly MlTelemetry telemetry;

        internal NewCommand(NewCommandSettings settings, MlTelemetry telemetry)
        {
            this.settings = settings;
            this.telemetry = telemetry;
        }

        public void Execute()
        {
            try
            {
                telemetry.LogAutoTrainMlCommand(settings.Dataset.FullName, settings.MlTask.ToString(), settings.Dataset.Length);
                CodeGenerationHelper codeGenerationHelper = new CodeGenerationHelper(new AutoMLEngine(settings), settings); // Needs to be improved.
                codeGenerationHelper.GenerateCode();
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Debug, e.ToString());
                logger.Log(LogLevel.Info, Strings.LookIntoLogFile);
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }
        }
    }
}
