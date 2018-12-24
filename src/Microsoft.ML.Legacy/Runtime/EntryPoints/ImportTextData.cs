// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

[assembly: EntryPointModule(typeof(Microsoft.ML.Legacy.EntryPoints.ImportTextData))]

// The warning #612 is disabled because the following code uses legacy TextLoader.
// Because that dependency will be removed form ML.NET, one needs to rewrite all places where legacy APIs are used.
#pragma warning disable 612
namespace Microsoft.ML.Legacy.EntryPoints
{
    /// <summary>
    /// A component for importing text files as <see cref="IDataView"/>.
    /// </summary>
    public static class ImportTextData
    {
        [TlcModule.EntryPointKind(typeof(ILearningPipelineLoader))]
        public sealed class LoaderInput
        {
            [Argument(ArgumentType.Required, ShortName = "data", HelpText = "Location of the input file", SortOrder = 1)]
            public IFileHandle InputFile;

            [Argument(ArgumentType.Required, ShortName = "args", HelpText = "Arguments", SortOrder = 2)]
            public TextLoader.Arguments Arguments = new TextLoader.Arguments();
        }

        [TlcModule.EntryPoint(Name = "Data.TextLoader", Desc = "Import a dataset from a text file")]
        public static ML.EntryPoints.ImportTextData.Output TextLoader(IHostEnvironment env, LoaderInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ImportTextData");
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var loader = host.CreateLoader(input.Arguments, new FileHandleSource(input.InputFile));
            return new ML.EntryPoints.ImportTextData.Output { Data = loader };
        }
    }
}
#pragma warning restore 612
