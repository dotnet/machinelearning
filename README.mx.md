# Machine Learning para .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) es un framework de machine learning de código libre y multiplataforma el cual hace el machine learning accesible para los desarrolladores .NET.

ML.NET permite a los desarrolladores .NET desarrollar sus propios modelos y mezclar ML personalizada en sus propias aplicaciones sin tener experiencia previa desarrollando o entrenando modelos de machine learning, todo desde .NET.

ML.NET fue originalmente desarrollada en el Microsoft Research y evolucionó en un framework importante en la ultima década y es usado en multiples productos en todo Microsoft por ejemplo Windows, Bing, PowerPoint, Excel y más.

Con este lanzamiento adelantado, ML.NET activa tareas de ML como clasificación (por ejemplo: support text clasficación, sentiment analysis) y regresiones (por ejemplo: predicciones de precios).

Dentro de todas esas capacidades del ML, este primer lanzamiento de ML.NET  también brinda el primer paso a las APIS de .NET para entrenar modelos, usando modelos para predicciones, así como los componentes fundamentales de este framework tales como algoritmos de aprendizaje, transformaciones y estructuras de datos ML.


## Instalación

[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET corre en Windows, Linux, y macOS - cualquier plataforma de 64 Bits [.NET Core](https://github.com/dotnet/core) o posterior está disponible.

La versión actual es la 0.6. revisa las [notas de la versión](docs/release-notes/0.6/release-0.6.md).

Primero verifica que tengas instalado [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) o posterior. ML.NET también trabaja en el Framework de .NET. nota que ML.NET actualmente debe correr en procesadores de 64 Bits.

Una vez que tengas la app, puedes instalar el gestor de paquetes NuGet desde la .NET Core CLI usando:
```
dotnet add package Microsoft.ML
```

o desde el gestor de paquetes NuGet:
```
Install-Package Microsoft.ML
```

Una alternativa, puedes agregar el paquete Microsoft.ML con Visual Studio desde el gestor de paquetes NuGet ó vía [Paquete](https://github.com/fsprojects/Paket).

Diariamente hay actualizaciones disponibles del projecto en tus noticias en el MyGet:

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## Compilando

Para compilar ML.NET desde la biblioteca origen por favor visita nuestra  [guía de desarrolladores](docs/project-docs/developer-guide.md).

|    | x64 Debug | x64 Release |
|:---|----------------:|------------------:|
|**Linux**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**macOS**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**Windows**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|

## Contribuyendo

Nos encantan la participación de nuestros usuarios! Por favor revisa nuestro [guía para contribuir](CONTRIBUTING.md).

## Comunidad
Por favor únete a nuestra comunidad en Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Este proyecto ha adoptado un código de conducta definido por el [convenido del contribuidor](https://contributor-covenant.org/) para aclarar el comportamiento esperado en nuestra comunidad.
Para mayor información, revisa el [El código de conducta de la fundación .NET](https://dotnetfoundation.org/code-of-conduct).

## Ejemplos

Aquí hay un ejemplo de entrenamiento de un modelo para predecir un sentimiento proveniente de unos textos de ejemplo.
(Puedes encontrar un ejemplo de los fundamentos de la API [Aquí](test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs)):

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

Ahora desde el modelo podemos hacer inferencias (predicciones):

```C#
var predictionFunc = model.MakePredictionFunction<SentimentInput, SentimentPrediction>(env);
var prediction = predictionFunc.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
};
Console.WriteLine("prediction: " + prediction.Sentiment);
```
Un libro especializado que muestra el uso de estas APIs para una variedad de existentes y nuevos escenarios puede ser encontrada
[aquí](docs/code/MlNetCookBook.md).


## Muestras

Tenemos un [repositorio de muestras](https://github.com/dotnet/machinelearning-samples) en donde puedes buscar.

## Licencia

ML.NET está licenciado bajo la [MIT license](LICENSE).

## Fundación .NET

ML.NET es un [proyecto .NET.](https://www.dotnetfoundation.org/projects)

Existen muchos proyectos .NET relacionados en GitHub.

- [.NET repositorio principal](https://github.com/Microsoft/dotnet) - links hacia 100 proyectos de .NET, de Microsoft y la comunidad. 
