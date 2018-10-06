# Machine Learning para .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) es un framework de machine learning de código libre y multiplataforma el cual hace el machine learning accesible para los desarrolladores .NET.

ML.NET permite a los desarrolladores .NET desarrollar sus propios modelos y mezclar ML personalizada en sus propias aplicaciones sin tener experiencia previa desarrollando o entrenando modelos de machine learning, todo desde .NET.

ML.NET fue originalmente desarrollada en el Microsoft Research y evolucionó en un framework importante en la ultima década y es usado en multiples productos en todo Microsoft por ejemplo Windows, Bing, PowerPoint, Excel y más.

Con este lanzamiento previo, ML.NET activa tareas de ML como clasificación (por ejemplo: support text clasficación, sentiment analysis) y regresiones (e.g. price-prediction).

Dentro de todas esas capacidades del ML, este primer lanzamiento de ML.NET  también brinda el primer paso a las APIS de .NET para entrenar modelos, usando modelos para predicciones, así como los componentes fundamentales de este framework tales como algoritmos de aprendizaje, transformaciones y estructuras de datos ML.


## Instalación

[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET runs on Windows, Linux, and macOS - any platform where 64 bit [.NET Core](https://github.com/dotnet/core) or later is available.

The current release is 0.6. Check out the [release notes](docs/release-notes/0.6/release-0.6.md).

First, ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework. Note that ML.NET currently must run in a 64-bit process.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```
