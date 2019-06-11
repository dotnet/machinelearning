# The ML.NET Roadmap

The goal of ML.NET project is to provide an easy to use, .NET-friendly ML platform. This document describes the tentative plan for the project in the short and long-term. 

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to open an issue in this repo. It's always a good idea to have a discussion before embarking on a large code change to make sure there is not duplicated effort.
Many of the features listed on the roadmap already exist in the internal version of the code-base.  They are marked with (*).  We plan to release more and more internal features to Github over time.

In the meanwhile, we are looking for contributions.  An easy place to start is to look at _up-for-grabs_ issues on [Github](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs)

## Short Term
### Training Improvements
* Deep Learning Training Support
  * Integrate with leading DNN package(s)
  * Support for transfer learning.
  * Hybrid training of pipelines containing both DNN and non-DNN predictors.
  * Fast.ai like APIs.

### Trained Model Management
* Export models to [ONNX](https://github.com/onnx/models)  (*)

## Longer Term

### Training Improvements
* Add more learners, perhaps, including:  (*)
  * [ProtoNN and Bonsaii](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/) for compact and efficient models.
* Integration with other ML packages
  * Accord.NET
  * etc.
* Additional ML tasks  (*)
  * _Sequence Classification_ - learns from a series of examples in a sequence, and each item is assigned a distinct label, akin to a multiclass classification task
* Additional Data source support
  * Data from SQL Databases, such as SQL Server
  * Data located on the cloud
  * Apache Parquet
  * Native Binary high-performance format
* Distributed Training
  * Easily train models on the cloud
* Whole-pipeline optimizations for both training and inference
* Automation of more data science tasks
* Additional Trainers
* Additional tasks

### Featurization Improvements
* Improved data wrangling support
* Add auto-suggestion of training pipelines. The technology will provide intelligent ```LearningPipeline``` suggestions based on training data attributes  (*)
* Additional natural language text preprocessing
* Time series and forecasting
* Support for Video, audio, and other data types

### Trained Model Management
* Model operationalization in the Cloud
* Model deployment on mobile platforms
* Ability to run [ONNX](https://github.com/onnx/models) models in the ```LearningPipeline```
* Support for the next version of ONNX
* Model deployment to IOT devices

### GUI Improvements
* Usability improvements
* Support of additional ML.NET features
* Improved code generation for training and inference
* Run the pipelines rather than just suggesting them; present to the user the pipelines and the metrics generated from running. 
* Distributed runs, rather than sequential. 

### Other
* Support for additional languages
* Published reproducible benchmarks against industry-leading ML toolkits on a variety of tasks and datasets

