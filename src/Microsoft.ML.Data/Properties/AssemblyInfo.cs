// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.TestFramework" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Core.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.InferenceTesting" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.OnnxTransformTest" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Predictor.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.TimeSeries.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.EntryPoints" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Legacy" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Maml" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.ResultProcessor" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Api" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Ensemble" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.FastTree" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.HalLearners" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.KMeansClustering" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.LightGBM" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Onnx" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.OnnxTransform" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Parquet" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.PCA" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.PipelineInference" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Recommender" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Runtime.ImageAnalytics" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Scoring" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.StandardLearners" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Sweeper" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.TensorFlow" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.TimeSeries" + PublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Transforms" + PublicKey.Value)]

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.StaticPipe" + PublicKey.Value)]

[assembly: WantsToBeBestFriends]
