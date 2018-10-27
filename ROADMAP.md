# The ML.NET Roadmap

The goal of ML.NET project is to provide an easy to use, .NET-friendly ML platform. This document describes the tentative plan for the project in the short and long-term. 

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to open an issue in this repo. It's always a good idea to have a discussion before embarking on a large code change to make sure there is not duplicated effort.
Many of the features listed on the roadmap already exist in the internal version of the code-base.  They are marked with (*).  We plan to release more and more internal features to Github over time.

In the meanwhile, we are looking for contributions.  An easy place to start is to look at _up-for-grabs_ issues on [Github](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs)

## Short Term
### Training Improvements
* Improved public API for training and inference
* Enhanced tests and scenarios
* Additional Learners
    * [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) for anomaly detection  (*)
	* [LightGBM](https://github.com/Microsoft/LightGBM) - a high-performance boosted decision tree  (*)
* Additional Learning Tasks  (*)
	* _Ranking_ - problem where the goal is to automatically sort (rank) instances within a group based on ranked examples in training data
	* _Anomaly Detection_ - is also known as _outlier detection_. It is a task to identify items, events or observations which do not conform to an expected pattern in the dataset.
	* _Quantile Regression_ is a type of regression analysis. Whereas regression results in estimates that approximate the conditional mean of the response variable given certain values of the predictor variables, quantile regression aims at estimating either the conditional median or other quantiles of the response variable
* Additional Data Sources support  (*)
	* Apache Parquet
	* Native Binary high-performance format

### Featurization Improvements
We already provide text/NLP and image processing functionalities that will be expanded
* Text  (*)
  * Natural language text preprocessing such as improving tokenization features, adding part-of-speech tagging, and sentence boundary disambiguation
  * Pre-trained text models (beyond current n-gram and pre-trained WordEmbedding text handling) that can further improve the extraction of semantic or sentiment features from text
* Image  (*)
  * Image preprocessing such as loading, resizing, and normalization of images
  * Image featurization, including industry-standard pre-trained ImageNet neural models, such as ResNet and AlexNet

### Trained Model Management
* Export models to [ONNX](https://github.com/onnx/models)  (*)

### GUI
* Release the Model Builder tool to ease model development  (*)
* Design improvements to make the design adhere better to Fluent principles
* Add a view for an easier comparison of several experiments
* Ability to select the best performing pipeline, by sweeping transforms, the same way learners are swept.

## Longer Term

### Training Improvements
* Add more learners, perhaps, including:  (*)
  * Generative Additive Models
  * [SymSGD](https://arxiv.org/pdf/1705.08030.pdf) -a fast linear SGD learner
  * Factorization Machines
  * [ProtoNN and Bonsaii](https://www.microsoft.com/en-us/research/project/resource-efficient-ml-for-the-edge-and-endpoint-iot-devices/) for compact and efficient models
* Integration with other ML packages
  * Accord.NET
  * etc.
* Deep Learning Support
  * Integrate with leading DNN package(s)
  * Support for transfer learning
  * Hybrid training of pipelines containing both DNN and non-DNN predictors
* Additional ML tasks  (*)
  * _Recommendation_ - Is a problem that can be phrased a: "For a given user, predict the ratings this user would give to the items that they have not explicitly rated yet"
  * _Anomaly Detection_, also known as _outlier detection_. It is a task to identify items, events or observations which do not conform to an expected pattern in the dataset. Typical examples are: detecting credit card fraud, medical problems or errors in text. Anomalies are also referred to as outliers, novelties, noise, deviations and exceptions
  * _Sequence Classification_ - learns from a series of examples in a sequence, and each item is assigned a distinct label, akin to a multiclass classification task
* Additional Data source support
  * Data from SQL Databases, such as SQL Server
  * Data located on the cloud
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

