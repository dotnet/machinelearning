// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Core.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Predictor.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.LightGbm" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Sweeper" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.StaticPipe" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Internal.FastTree" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]

[assembly: WantsToBeBestFriends]
