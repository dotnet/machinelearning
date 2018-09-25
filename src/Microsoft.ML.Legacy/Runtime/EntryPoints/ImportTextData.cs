// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Legacy;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ImportTextData), null, typeof(SignatureEntryPointModule), "ImportTextData")]

namespace Microsoft.ML.Runtime.EntryPoints
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

        [TlcModule.EntryPointKind(typeof(ILearningPipelineLoader))]
        public sealed class LoaderInput
        {
            [Argument(ArgumentType.Required, ShortName = "data", HelpText = "Location of the input file", SortOrder = 1)]
            public IFileHandle InputFile;

            [Argument(ArgumentType.Required, ShortName = "args", HelpText = "Arguments", SortOrder = 2)]
            public TextLoader.Arguments Arguments = new TextLoader.Arguments();
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The resulting data view", SortOrder = 1)]
            public IDataView Data;
        }

#pragma warning disable 0618
        [Obsolete("Use TextLoader instead.")]
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
#pragma warning restore 0618

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
