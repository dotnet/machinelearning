# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)은 .NET 개발자들에게 머신러닝을 이용할 수 있게 해주는 크로스 플랫폼 오픈소스 머신러닝 프레임워크입니다.

ML.NET을 이용해 .NET 개발자는 머신러닝 모델을 개발하거나 튜닝하는 데 전문적인 사전 지식 없이 모델을 개발하고 맞춤형 머신러닝을 애플리케이션에 적용시킬 수 있습니다.

ML.NET은 Microsoft Research에서 처음 개발되어 지난 10년 동안 중요한 프레임워크로 발전했으며 Windows, Bing, PowerPoint, Excel 등과 같은 많은 Microsoft의 제품군에서 사용되고 있습니다.

첫 번째 프리뷰 릴리즈에서는 ML.NET은 분류(예: 텍스트 분류, 감정 분석)과 회귀(예:가격 예측)와 같은 ML 작업을 지원합니다.

이러한 ML 기능들과 함께, ML.NET의 첫 번째 릴리즈는 학습 알고리즘, 트랜스폼 및 ML 데이터 구조와 같은 이 프레임워크의 핵심 구성 요소뿐만 아니라 모델 훈련 및 예측을 하기 위한 .NET API의 초안도 제공합니다.

이 첫 번째 프리뷰 릴리즈에서는 ML.NET은 분류(예: 텍스트 분류, 감정 분석) 및 회귀(예: 가격 예측)와 같은 ML 작업을 지원합니다.

## 설치

[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET은 Windows, Linux 및 macOS - 64 bit 이상의 [.Net Core](https://github.com/dotnet/core를 사용할 수 있는 모든 플랫폼에서 실행 가능합니다.

현재 릴리즈는 0.6입니다. 새로운 내용을 보려면 [release notes](docs/release-notes/0.6/release-0.6.md)를 확인하십시오.

First, ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework. Note that ML.NET currently must run in a 64-bit process.
먼저, .NET Core 2.0 이상을 설치했는지 확인합니다. ML.NET은 .NET Framework에서도 작동합니다. 현재 ML.NET은 64-bit 프로세스에서 실행되어야 합니다.

앱이 설치되어있다면 .NET Core CLI에서 다음을 이용해 ML.NET NuGet 패키지를 설치할 수 있습니다.
```
dotnet add package Microsoft.ML
```

또는 NuGet 패키지 매니저에서 다음을 사용할 수도 있습니다:
```
Install-Package Microsoft.ML
```

대안으로, Visual Studio의 NuGet 패키지 관리자 내에서 또는 [Paket](https://github.com/fsprojects/Paket)을 통해 Microsoft.ML 패키지를 추가할 수 있습니다.

프로젝트의 일일 NuGet 빌드는 MyGet 피드에서도 사용할 수 있습니다.

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## 빌드

소스를 이용해 직접 ML.NET을 빌드하려면 [개발자 가이드](docs/project-docs/developer-guide.md)를 확인하십시오.

|    | x64 Debug | x64 Release |
|:---|----------------:|------------------:|
|**Linux**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**macOS**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**Windows**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|

## 기여하기

우리는 기여를 환영합니다! [기여 가이드](CONTRIBUTING.md)를 확인해주세요.

## 커뮤니티

Glitter의 커뮤니티에 가입하세요 [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

이 프로젝트는 커뮤니티에서 예상되는 행동을 명확히 하기 위해 [기여자 규약](https://contributor-covenant.org/)에 정의된 행동 강령을 채택했습니다.
자세한 내용은 [.NET 재단 행동 강령](https://dotnetfoundation.org/code-of-conduct)을 참조하십시오.

## 예제

다음은 텍스트 샘플에서 감정을 예측하도록 모델을 학습시키는 코드의 예입니다.
(레거시 API의 샘플은 [여기](test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs)에서 찾을 수 있습니다.)

```C#
var env = new LocalEnvironment();
var reader = TextLoader.CreateReader(env, ctx => (
        Target: ctx.LoadFloat(2),
        FeatureVector: ctx.LoadFloat(3, 6)),
        separator: ',',
        hasHeader: true);
var data = reader.Read(new MultiFileSource(dataPath));
var classification = new MulticlassClassificationContext(env);
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
    r.Target,
    Prediction: classification.Trainers.Sdca(r.Target.ToKey(), r.FeatureVector)));
var model = learningPipeline.Fit(data);

```

이제 모델에서 추론(예측)을 할 수 있습니다.

```C#
var predictionFunc = model.MakePredictionFunction<SentimentInput, SentimentPrediction>(env);
var prediction = predictionFunc.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
};
Console.WriteLine("prediction: " + prediction.Sentiment);
```

다양한 기존 시나리오와 새로운 시나리오에서 이러한 API를 사용하는 방법을 보여주는 가이드는 [여기](docs/code/MlNetCookBook.md)에서 찾을 수 있습니다.


## 샘플

확인할 수 있는 [샘플 리포지토리](https://github.com/dotnet/machinelearning-samples)가 있습니다.

## 라이선스

ML.NET은 [MIT 허가서]에 따라 라이선스가 부여됩니다.

## .NET 재단

ML.NET은[.NET 재단](https://www.dotnetfoundation.org/projects) 프로젝트입니다.

GitHub에서 많은 .NET 관련 프로젝트를 확인할 수 있습니다.

- [.NET 리포지토리](https://github.com/Microsoft/dotnet) - Microsoft 및 커뮤니티에서 제공하는 수백 개의 .NET 프로젝트에 대한 링크
