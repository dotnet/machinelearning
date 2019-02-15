// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ImportTextData), null, typeof(SignatureEntryPointModule), "ImportTextData")]

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// A component for importing text files as <see cref="IDataView"/>.
    /// </summary>
    internal static class ImportTextData
    {
        public sealed class Input
        {
            [Argument(ArgumentType.Required, ShortName = "data", HelpText = "Location of the input file", SortOrder = 1)]
            public IFileHandle InputFile;

            [Argument(ArgumentType.AtMostOnce, ShortName = "schema", HelpText = "Custom schema to use for parsing", SortOrder = 2)]
            public string CustomSchema = null;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The resulting data view", SortOrder = 1)]
            public IDataView Data;
        }

        [TlcModule.EntryPoint(Name = "Data.CustomTextLoader", Desc = "Import a dataset from a text file")]
        public static Output ImportText(IHostEnvironment env, Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ImportTextData");
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var loader = host.CreateLoader(string.Format("Text{{{0}}}", input.CustomSchema), new FileHandleSource(input.InputFile));
            return new Output { Data = loader };
        }

        public sealed class LoaderInput
        {
            [Argument(ArgumentType.Required, ShortName = "data", HelpText = "Location of the input file", SortOrder = 1)]
            public IFileHandle InputFile;

            [Argument(ArgumentType.Required, ShortName = "args", HelpText = "Arguments", SortOrder = 2)]
            public TextLoader.Options Arguments = new TextLoader.Options();
        }

        [TlcModule.EntryPoint(Name = "Data.TextLoader", Desc = "Import a dataset from a text file")]
        public static Output TextLoader(IHostEnvironment env, LoaderInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ImportTextData");
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var loader = host.CreateLoader(input.Arguments, new FileHandleSource(input.InputFile));
            return new Output { Data = loader };
        }
    }
}
