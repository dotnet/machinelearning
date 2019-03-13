// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML.Internal.CpuMath.Core;

[assembly: InternalsVisibleTo("Microsoft.ML.CpuMath.UnitTests.netstandard" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo("Microsoft.ML.CpuMath.UnitTests.netcoreapp" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo("Microsoft.ML.Data" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.FastTree" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.Mkl.Components" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.KMeansClustering" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.PCA" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.StandardTrainers" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.Sweeper" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.TimeSeries" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.Transforms" + PublicKey.Value)]
[assembly: InternalsVisibleTo("Microsoft.ML.Benchmarks.Tests" + PublicKey.TestValue)]

[assembly: WantsToBeBestFriends]
