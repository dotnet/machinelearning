// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.EntryPoints;

[assembly: LoadableClass(typeof(Microsoft.ML.LightGBM.SingleTrainer),
    null, typeof(Microsoft.ML.LightGBM.SignatureParallelTrainer), "single")]

[assembly: EntryPointModule(typeof(Microsoft.ML.LightGBM.SingleTrainerFactory))]

namespace Microsoft.ML.LightGBM
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
