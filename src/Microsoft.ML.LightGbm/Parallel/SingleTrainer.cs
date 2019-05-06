// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(Microsoft.ML.Trainers.LightGbm.SingleTrainer),
    null, typeof(Microsoft.ML.Trainers.LightGbm.SignatureParallelTrainer), "single")]

[assembly: EntryPointModule(typeof(Microsoft.ML.Trainers.LightGbm.SingleTrainerFactory))]

namespace Microsoft.ML.Trainers.LightGbm
{
    internal sealed class SingleTrainer : IParallel
    {
        AllgatherFunction IParallel.GetAllgatherFunction()
        {
            return null;
        }

        ReduceScatterFunction IParallel.GetReduceScatterFunction()
        {
            return null;
        }

        int IParallel.NumMachines()
        {
            return 1;
        }

        string IParallel.ParallelType()
        {
            return "serial";
        }

        int IParallel.Rank()
        {
            return 0;
        }

        Dictionary<string, string> IParallel.AdditionalParams()
        {
            return null;
        }
    }

    [TlcModule.Component(Name = "Single", Desc = "Single node machine learning process.")]
    internal sealed class SingleTrainerFactory : ISupportParallel
    {
        public IParallel CreateComponent(IHostEnvironment env) => new SingleTrainer();
    }
}
