// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


//=================================================================================================
// This test can be run either as a compiled test with .NET Core (on any platform) or
// manually in script form (to help debug it and also check that F# scripting works with ML.NET).
// Running as a script requires using F# Interactive on Windows, and the explicit references below.  
// The references would normally be created by a package loader for the scripting 
// environment, for example, see https://github.com/isaacabraham/ml-test-experiment/, but 
// here we list them explicitly to avoid the dependency on a package loader,
//
// You should build Microsoft.ML.FSharp.Tests in Debug mode for framework net461 
// before running this as a script with F# Interactive by editing the project
// file to have:
//    <TargetFrameworks>netcoreapp2.1; net461</TargetFrameworks>

#if INTERACTIVE
#r "netstandard"
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.Core.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Google.Protobuf.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Newtonsoft.Json.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/System.CodeDom.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/System.Threading.Tasks.Dataflow.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.CpuMath.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.Data.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.Transforms.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.ResultProcessor.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.PCA.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.KMeansClustering.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.FastTree.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.Api.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.Sweeper.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.StandardLearners.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/Microsoft.ML.PipelineInference.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/xunit.core.dll" 
#r @"../../bin/AnyCPU.Debug/Microsoft.ML.FSharp.Tests/net461/xunit.assert.dll" 
#r "System" 
#r "System.ComponentModel.Composition" 
#r "System.Core" 
#r "System.Xml.Linq" 

// Later tests will add data import using F# type providers:
//#r @"../../packages/fsharp.data/3.0.0-beta4/lib/netstandard2.0/FSharp.Data.dll" // this must be referenced from its package location

#endif

//================================================================================
// The tests proper start here

#if !INTERACTIVE
namespace Microsoft.ML.FSharp.Tests
#endif

open System
open Microsoft.ML
open Microsoft.ML.Legacy.Data
open Microsoft.ML.Legacy.Transforms
open Microsoft.ML.Legacy.Trainers
open Microsoft.ML.Runtime.Api
open Xunit

module SmokeTest1 = 

    type SentimentData() =
        [<Column(ordinal = "0"); DefaultValue>]
        val mutable SentimentText : string
        [<Column(ordinal = "1", name = "Label"); DefaultValue>]
        val mutable Sentiment : float32

    type SentimentPrediction() =
        [<ColumnName("PredictedLabel"); DefaultValue>]
        val mutable Sentiment : bool

    [<Fact>]
    let ``FSharp-Sentiment-Smoke-Test`` () =

        // See https://github.com/dotnet/machinelearning/issues/401: forces the loading of ML.NET component assemblies
        let _load  =
            [ typeof<Microsoft.ML.Transforms.Text.TextNormalizerEstimator>; 
              typeof<Microsoft.ML.Runtime.FastTree.FastTree> ]

        let testDataPath = __SOURCE_DIRECTORY__ + @"/../data/wikipedia-detox-250-line-data.tsv"

        let pipeline = Legacy.LearningPipeline()

        pipeline.Add(
            TextLoader(testDataPath).CreateFrom<SentimentData>(
                Arguments = 
                    TextLoaderArguments(
                        HasHeader = true,
                        Column = [| TextLoaderColumn(Name = "Label", 
                                                     Source = [| TextLoaderRange(0) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Num))
                                    TextLoaderColumn(Name = "SentimentText", 
                                                     Source = [| TextLoaderRange(1) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Text)) |] 
                    )))

        pipeline.Add(
            TextFeaturizer(
                "Features", [| "SentimentText" |],
                KeepPunctuations = false,
                OutputTokens = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2
            ))

        pipeline.Add(
            FastTreeBinaryClassifier(
                NumLeaves = 5, 
                NumTrees = 5, 
                MinDocumentsInLeafs = 2
            ))

        let model = pipeline.Train<SentimentData, SentimentPrediction>()

        let predictions =
            [ SentimentData(SentimentText = "This is a gross exaggeration. Nobody is setting a kangaroo court. There was a simple addition.")
              SentimentData(SentimentText = "Sort of ok")
              SentimentData(SentimentText = "Joe versus the Volcano Coffee Company is a great film.") ]
            |> model.Predict

        let predictionResults = [ for p in predictions -> p.Sentiment ]
        Assert.Equal<bool list>(predictionResults, [ false; true; true ])

module SmokeTest2 = 
    open System

    [<CLIMutable>]
    type SentimentData =
        { [<Column(ordinal = "0")>] 
          SentimentText : string

          [<Column(ordinal = "1", name = "Label")>] 
          Sentiment : float32 }

    [<CLIMutable>]
    type SentimentPrediction =
        { [<ColumnName("PredictedLabel")>] 
           Sentiment : bool }

    [<Fact>]
    let ``FSharp-Sentiment-Smoke-Test`` () =

        // See https://github.com/dotnet/machinelearning/issues/401: forces the loading of ML.NET component assemblies
        let _load  =
            [ typeof<Microsoft.ML.Transforms.Text.TextNormalizerEstimator>; 
              typeof<Microsoft.ML.Runtime.FastTree.FastTree> ]

        let testDataPath = __SOURCE_DIRECTORY__ + @"/../data/wikipedia-detox-250-line-data.tsv"

        let pipeline = Legacy.LearningPipeline()

        pipeline.Add(
            TextLoader(testDataPath).CreateFrom<SentimentData>(
                Arguments = 
                    TextLoaderArguments(
                        HasHeader = true,
                        Column = [| TextLoaderColumn(Name = "Label", 
                                                     Source = [| TextLoaderRange(0) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Num))
                                    TextLoaderColumn(Name = "SentimentText", 
                                                     Source = [| TextLoaderRange(1) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Text)) |] 
                    )))

        pipeline.Add(
            TextFeaturizer(
                "Features", [| "SentimentText" |],
                KeepPunctuations = false,
                OutputTokens = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2
            ))

        pipeline.Add(
            FastTreeBinaryClassifier(
                NumLeaves = 5, 
                NumTrees = 5, 
                MinDocumentsInLeafs = 2
            ))

        let model = pipeline.Train<SentimentData, SentimentPrediction>()

        let predictions =
            [ { SentimentText = "This is a gross exaggeration. Nobody is setting a kangaroo court. There was a simple addition."; Sentiment = 0.0f }
              { SentimentText = "Sort of ok"; Sentiment = 0.0f }
              { SentimentText = "Joe versus the Volcano Coffee Company is a great film."; Sentiment = 0.0f } ]
            |> model.Predict

        let predictionResults = [ for p in predictions -> p.Sentiment ]
        Assert.Equal<bool list>(predictionResults, [ false; true; true ])

module SmokeTest3 = 

    type SentimentData() =
        [<Column(ordinal = "0")>] 
        member val SentimentText = "".AsMemory() with get, set

        [<Column(ordinal = "1", name = "Label")>] 
        member val Sentiment = 0.0 with get, set

    type SentimentPrediction() =
        [<ColumnName("PredictedLabel")>] 
        member val Sentiment = false with get, set

    [<Fact>]
    let ``FSharp-Sentiment-Smoke-Test`` () =

        // See https://github.com/dotnet/machinelearning/issues/401: forces the loading of ML.NET component assemblies
        let _load  =
            [ typeof<Microsoft.ML.Transforms.Text.TextNormalizerEstimator>; 
              typeof<Microsoft.ML.Runtime.FastTree.FastTree> ]

        let testDataPath = __SOURCE_DIRECTORY__ + @"/../data/wikipedia-detox-250-line-data.tsv"

        let pipeline = Legacy.LearningPipeline()

        pipeline.Add(
            TextLoader(testDataPath).CreateFrom<SentimentData>(
                Arguments = 
                    TextLoaderArguments(
                        HasHeader = true,
                        Column = [| TextLoaderColumn(Name = "Label", 
                                                     Source = [| TextLoaderRange(0) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Num))
                                    TextLoaderColumn(Name = "SentimentText", 
                                                     Source = [| TextLoaderRange(1) |], 
                                                     Type = Nullable (Legacy.Data.DataKind.Text)) |] 
                    )))

        pipeline.Add(
            TextFeaturizer(
                "Features", [| "SentimentText" |],
                KeepPunctuations = false,
                OutputTokens = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2
            ))

        pipeline.Add(
            FastTreeBinaryClassifier(
                NumLeaves = 5, 
                NumTrees = 5, 
                MinDocumentsInLeafs = 2
            ))

        let model = pipeline.Train<SentimentData, SentimentPrediction>()

        let predictions =
            [ SentimentData(SentimentText = "This is a gross exaggeration. Nobody is setting a kangaroo court. There was a simple addition.".AsMemory())
              SentimentData(SentimentText = "Sort of ok".AsMemory())
              SentimentData(SentimentText = "Joe versus the Volcano Coffee Company is a great film.".AsMemory()) ]
            |> model.Predict

        let predictionResults = [ for p in predictions -> p.Sentiment ]
        Assert.Equal<bool list>(predictionResults, [ false; true; true ])

