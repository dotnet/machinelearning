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
        private readonly NewCommandSettings _settings;
        private readonly MlTelemetry _telemetry;

        internal NewCommand(NewCommandSettings settings, MlTelemetry telemetry)
        {
            _settings = settings;
            _telemetry = telemetry;
        }

        public void Execute()
        {
            _telemetry.LogAutoTrainMlCommand(_settings.Dataset.Name, _settings.MlTask.ToString(), _settings.Dataset.Length);

            CodeGenerationHelper codeGenerationHelper = new CodeGenerationHelper(new AutoMLEngine(_settings), _settings); // Needs to be improved.
            codeGenerationHelper.GenerateCode();
        }
    }
}
