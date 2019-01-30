// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Ensemble" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.StaticPipe" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Core.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Predictor.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "RunTestsMore" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Internal.MetaLinearLearner" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "ParameterMixer" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Runtime.Sar" + InternalPublicKey.Value)]
