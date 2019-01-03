// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ImportTextData), null, typeof(SignatureEntryPointModule), "ImportTextData")]

// The warning #612 is disabled because the following code uses legacy TextLoader.
// Because that dependency will be removed form ML.NET, one needs to rewrite all places where legacy APIs are used.
#pragma warning disable 612
namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// A component for importing text files as <see cref="IDataView"/>.
    /// </summary>
    public static class ImportTextData
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
    }
}
#pragma warning restore 612
