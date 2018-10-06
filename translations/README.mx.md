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

Una vez que tengas la app, puedes instalar ML.NET en el gestor de paquetes NuGet desde la .NET Core CLI usando:
```
dotnet add package Microsoft.ML
```
