// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Predictor.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]

[assembly: WantsToBeBestFriends]
