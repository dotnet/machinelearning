// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;

[assembly: EntryPointModule(typeof(IterationTerminator.Arguments))]
[assembly: EntryPointModule(typeof(ISupportITerminatorFactory))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    [TlcModule.ComponentKind("SearchTerminator")]
    public interface ISupportITerminatorFactory : IComponentFactory<ITerminator> { }

    public sealed class IterationTerminator : ITerminator
    {
        private readonly int _finalHistoryLength;

        [TlcModule.Component(Name = "IterationLimited", FriendlyName = "Pipeline Sweep Iteration Terminator", Desc = "Terminators a sweep based on total number of iterations.")]
        public sealed class Arguments : ISupportITerminatorFactory
        {
            [Argument(ArgumentType.Required, HelpText = "Total number of iterations.", ShortName = "length")]
            public int FinalHistoryLength;

            public ITerminator CreateComponent(IHostEnvironment env) => new IterationTerminator(FinalHistoryLength);
        }

        public IterationTerminator(int finalHistoryLength)
        {
            _finalHistoryLength = finalHistoryLength;
        }

        public bool ShouldTerminate(IEnumerable<PipelinePattern> history) =>
            history.ToArray().Length >= _finalHistoryLength;

        public int RemainingIterations(IEnumerable<PipelinePattern> history) =>
            _finalHistoryLength - history.ToArray().Length;
    }
}
