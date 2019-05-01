// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.DotNet.Cli.Telemetry;
using Microsoft.ML.CLI.CodeGenerator;
using Microsoft.ML.CLI.Data;

namespace Microsoft.ML.CLI.Commands.New
{
    internal class NewCommand : ICommand
    {
        private readonly NewCommandSettings settings;
        private readonly MlTelemetry telemetry;

        internal NewCommand(NewCommandSettings settings, MlTelemetry telemetry)
        {
            this.settings = settings;
            this.telemetry = telemetry;
        }

        public void Execute()
        {
            telemetry.LogAutoTrainMlCommand(settings.Dataset.Name, settings.MlTask.ToString(), settings.Dataset.Length);

            CodeGenerationHelper codeGenerationHelper = new CodeGenerationHelper(new AutoMLEngine(settings), settings); // Needs to be improved.
            codeGenerationHelper.GenerateCode();
        }
    }
}
