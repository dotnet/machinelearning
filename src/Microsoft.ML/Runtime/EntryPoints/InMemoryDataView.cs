// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(InMemoryDataView), null, typeof(SignatureEntryPointModule), "InMemoryDataView")]
namespace Microsoft.ML.Runtime.EntryPoints
{
    public class InMemoryDataView
    {
        public sealed class Input
        {
            [Argument(ArgumentType.Required, HelpText = "Pointer to IDataView in memory", SortOrder = 1)]
            public IDataView Data;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The resulting data view", SortOrder = 1)]
            public IDataView Data;
        }

        [TlcModule.EntryPoint(Name = "Data.DataViewReference", Desc = "Pass dataview from memory to experiment")]
        public static Output ImportData(IHostEnvironment env, Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("DataViewReference");
            env.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            return new Output { Data = input.Data };
        }
    }
}
