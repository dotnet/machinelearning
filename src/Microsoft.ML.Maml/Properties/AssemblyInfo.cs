// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo("Microsoft.ML.TestFramework" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.PerformanceTests" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.ResultProcessor" + PublicKey.Value)]

[assembly: InternalsVisibleTo("RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo("TLC" + InternalPublicKey.Value)]
