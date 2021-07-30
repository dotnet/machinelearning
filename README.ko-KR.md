# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)은 Power BI, Windows Defender 및 Azure를 비롯한 여러 Microsoft 제품에서 동일한 코드를 이용해 .NET 개발자가 기계 학습에 접근할 수 있도록 하는 크로스 플랫폼 오픈 소스 기계 학습 프레임워크입니다.

ML.NET을 이용해 .NET 개발자는 기계 학습 모델 개발 또는 튜닝에 대한 전문적인 사전 지식 없이도 .NET을 이용해 모델을 개발/훈련하고 맞춤형 기계 학습 모델을 프로그램에 적용시킬 수 있습니다. 파일 및 데이터베이스에서 데이터를 불러오는 것과 데이터의 변환을 지원하며, 다양한 ML 알고리즘을 포함하고 있습니다.

ML.NET은 분류(예: 텍스트 분류, 감정 분석), 회귀(예: 가격 예측)와 같은 기계 학습(ML) 작업과 이상 탐지, 시계열 예측, 클러스터링, 랭킹 등과 같은 많은 다양한 ML 작업을 지원합니다.

## ML.NET을 이용해 기계 학습 시작하기

기계 학습을 처음 접하는 경우 ML.NET을 대상으로 하는 이 리소스 모음에서 기본 사항들을 학습하십시오.

[ML.NET 배우기](https://dotnet.microsoft.com/learn/ml-dotnet)

## ML.NET 설명서, 자습서 및 참조

[문서 및 자습서](https://docs.microsoft.com/en-us/dotnet/machine-learning/)를 확인하십시오.

[API 레퍼런스 문서](https://docs.microsoft.com/en-us/dotnet/api/?view=ml-dotnet)를 확인하십시오.

## 샘플 앱

[ML.NET 샘플 앱](https://github.com/dotnet/machinelearning-samples)을 포함한 GitHub 리포지토리에서 감성 분석, 부정 행위 탐지, 제품 추천, 가격 예측, 이상 탐지, 이미지 분류, 개체 탐지 등과 같은 다양한 시나리오를 제공합니다.

Microsoft에서 제공하는 ML.NET 샘플 외에도 별도의 페이지 [ML.NET 커뮤티니 샘플](https://github.com/dotnet/machinelearning-samples/blob/main/docs/COMMUNITY-SAMPLES.md)에 소개 된 커뮤니티에서 만든 샘플들이 있습니다.

## ML.NET YouTube 비디오 플레이리스트

YouTube의 [ML.NET 비디오 플레이리스트](https://aka.ms/mlnetyoutube)에는 몇 가지 짧은 동영상이 있습니다. 각 비디오는 ML.NET의 특정 주제에 중점을 두고있습니다.

## ML.NET에서 지원하는 운영 체제 및 프로세서 아키텍처

ML.NET은 [.NET Core](https://github.com/dotnet/core)를 사용하여 Windows, Linux 및 macOS에서 실행되거나, .NET Framework를 사용하여 Windows에서 실행될 수 있습니다.

64비트는 모든 플랫폼에서 지원됩니다. 32비트는 Windows에서 지원되며,TensorFlow 및 LightGBM 관련 기능들은 지원되지 않습니다.

## ML.NET NuGet 패키지 상태

[![NuGet Status](https://img.shields.io/nuget/vpre/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

## 릴리즈 노트

새로운 기능을 확인하려면 [릴리즈 노트](docs/release-notes)를 확인하십시오.

## ML.NET 패키지 사용하기

먼저 [.NET Core 2.1](https://www.microsoft.com/net/learn/get-started) 이상을 설치했는지 확인합니다. ML.NET은 .NET Framework 4.6.1 이상에서도 작동하지만 4.7.2 이상이 권장됩니다.

앱이 있는 경우, 다음을 사용하여 .NET Core CLI에서 ML.NET NuGet 패키지를 설치할 수 있습니다.
```
dotnet add package Microsoft.ML
```

또는 NuGet 패키지 매니저에서 설치할 수도 있습니다.
```
Install-Package Microsoft.ML
```

대안으로, Visual Studio의 NuGet 패키지 관리자 내에서 또는 [Paket](https://github.com/fsprojects/Paket)을 통해 Microsoft.ML 패키지를 추가할 수 있습니다.

프로젝트의 일일 NuGet 빌드는 Azure DevOps 피드에서도 사용할 수 있습니다.

> [https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json](https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json)

## ML.NET 빌드하기 (ML.NET 오픈소스 코드를 빌드하려 하는 기여자들을 위해)

소스에서 ML.NET을 빌드하려면 [개발자 가이드](docs/project-docs/developer-guide.md)를 방문하십시오.

[![codecov](https://codecov.io/gh/dotnet/machinelearning/branch/main/graph/badge.svg?flag=production)](https://codecov.io/gh/dotnet/machinelearning)

|    | Debug | Release |
|:---|----------------:|------------------:|
|**CentOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Ubuntu**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**macOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows x64**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows FullFramework**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows x86**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows NetCore3.1**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|

## 릴리즈 프로세스 및 버전 관리

다양한 종류의 ML.NET 릴리즈를 확인하려면 [릴리즈 프로세스 문서](docs/release-notes)를 확인하십시오.

## 기여하기

우리는 기여를 환영합니다! [기여 가이드](CONTRIBUTING.md)를 확인하십시오.

## 커뮤니티

Glitter의 커뮤니티에 가입하세요 [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

이 프로젝트는 커뮤니티에서 예상되는 행동을 명확히 하기 위해 [기여자 규약](https://contributor-covenant.org/)에 정의된 행동 강령을 채택했습니다.
자세한 내용은 [.NET 재단 행동 강령](https://dotnetfoundation.org/code-of-conduct)을 참조하십시오.


## 예제

다음은 텍스트 샘플에서 감정을 예측하도록 모델을 학습시키는 스니펫 코드입니다. [샘플 리포지토리](https://github.com/dotnet/machinelearning-samples)에서 전체 샘플을 확인할 수 있습니다.

```C#
var dataPath = "sentiment.csv";
var mlContext = new MLContext();
var loader = mlContext.Data.CreateTextLoader(new[]
    {
        new TextLoader.Column("SentimentText", DataKind.String, 1),
        new TextLoader.Column("Label", DataKind.Boolean, 0),
    },
    hasHeader: true,
    separatorChar: ',');
var data = loader.Load(dataPath);
var learningPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
        .Append(mlContext.BinaryClassification.Trainers.FastTree());
var model = learningPipeline.Fit(data);
```

이제 모델에서 추론(예측)을 할 수 있습니다.

```C#
var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
var prediction = predictionEngine.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
});
Console.WriteLine("prediction: " + prediction.Prediction);
```
다양한 기존 시나리오와 새로운 시나리오에서 이러한 API를 사용하는 방법을 보여주는 가이드는 [여기](docs/code/MlNetCookBook.md)에서 찾을 수 있습니다.


## 라이선스

ML.NET은 [MIT 허가서](LICENSE)에 따라 라이선스가 부여되며 상업적으로 무료로 사용할 수 있습니다.

## .NET 재단

ML.NET은[.NET 재단](https://www.dotnetfoundation.org/projects) 프로젝트입니다.

GitHub에서 많은 .NET 관련 프로젝트를 확인할 수 있습니다.

- [.NET 홈 리포지토리](https://github.com/Microsoft/dotnet) - Microsoft 및 커뮤니티에서 제공하는 수백 개의 .NET 프로젝트에 대한 링크
