// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Mkl.Components.StaticPipe" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Internal.MetaLinearLearner" + InternalPublicKey.Value)]

[assembly: WantsToBeBestFriends]
