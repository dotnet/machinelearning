// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;

namespace Microsoft.ML.CLI
{
    class Program
    {
        public static void Main(string[] args)
        {
            var parser = new CommandLineBuilder()
                         // parser
                         .AddCommand(CommandDefinitions.New())
                         .UseDefaults()
                         .Build();

            parser.InvokeAsync(args).Wait();
        }


    }
}
