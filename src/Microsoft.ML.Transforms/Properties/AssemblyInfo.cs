// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.StaticPipe" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Runtime.NeuralNetworks" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.RServerScoring.NeuralNetworks" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Runtime.TextAnalytics" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.RServerScoring.TextAnalytics" + InternalPublicKey.Value)]

[assembly: WantsToBeBestFriends]
