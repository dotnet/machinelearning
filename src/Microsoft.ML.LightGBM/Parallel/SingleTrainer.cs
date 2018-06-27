// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(Microsoft.ML.Runtime.LightGBM.SingleTrainer),
    null, typeof(Microsoft.ML.Runtime.LightGBM.SignatureParallelTrainer), "single")]

[assembly: EntryPointModule(typeof(Microsoft.ML.Runtime.LightGBM.SingleTrainerFactory))]

namespace Microsoft.ML.Runtime.LightGBM
{
    public sealed class SingleTrainer : IParallel
    {
        public AllgatherFunction GetAllgatherFunction()
        {
            return null;
        }

        public ReduceScatterFunction GetReduceScatterFunction()
        {
            return null;
        }

        public int NumMachines()
        {
            return 1;
        }

        public string ParallelType()
        {
            return "serial";
        }

        public int Rank()
        {
            return 0;
        }

        public Dictionary<string, string> AdditionalParams()
        {
            return null;
        }
    }

    [TlcModule.Component(Name = "Single", Desc = "Single node machine learning process.")]
    public sealed class SingleTrainerFactory : ISupportParallel
    {
        public IParallel CreateComponent(IHostEnvironment env) => new SingleTrainer();
    }
}
