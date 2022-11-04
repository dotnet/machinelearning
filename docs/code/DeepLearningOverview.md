# What is Deep Learning?

Deep Learning is an umbrella term for an approach to Machine Learning
that makes use of "deep" Neural Networks, a kind of models originally
inspired by the function of biological brains.  These days, Deep
Learning is probably the most visible area of Machine Learning, and it
has seen amazing successes in areas like Computer Vision, Natural
Language Processing and, in combination with Reinforcement Learning,
more complicated settings such as game playing, decision making and
simulation.

A crucial element of the success of Deep Learning ("DL" in what
follows) has been the existence of software frameworks and runtimes
that facilitate the creation of Neural Network models and their
execution for inference.  Examples of such frameworks include
Tensorflow, (Py)Torch and onnx.  ML.NET provides access to some of
these frameworks, while maintaining the familiar pipeline interface.
In this way, users of ML.NET can take advantage of some
state-of-the-art models and applications of DL at a lower cost than
the steep learning curve learning that other DL frameworks require.

# Deep Learning vs Machine Learning?

As mentioned above, DL relies on "Neural Network" models, in contrast
with "traditional" Machine Learning techniques (which use a wider
variety of architectures, such as, for example, generalized linear
models, decision trees or Support Vector Machines).  The most
immediate, practical implication of this difference is that DL methods
may be better or worse suited for some kind of data.  The performance
of DL methods on images, on textual and on other non- or
less-structured data has been well documented in the literature.
Traditional Machine Learning methods such as gradient-boosted trees
(XGBoost, LightGBM and CatBoost) seem to still have an edge when it
comes to tabular data.  The best approach is always to experiment with
your particular data source and use case and determine for yourself,
and ML.NET makes this experimentation relatively straightforward and
pain-free.

# Neural Network architectures

A crucial differentiating characteristic of DL from other classes (or
schools) of ML is the use of artificial Neural Networks as models.  At
a high-level, one can think of a Neural Network as a configuration of
"processing units" where the output of each unit constitutes the input
of another.  Each of these units can take one or many inputs, and
essentially carries out a weighted sum of its inputs, applies an
offset (or "bias") and then a non-linear transformation function
(called "activation").  Different arrangements of these relatively
simple components have been proven surprisingly rich to describe
decision boundaries in classification, regression functions and other
structures central to ML tasks.

The past decade has seen an explosion of use cases, applications and
techniques of DL, each more impressive than the last, pushing the
boundaries of what functionalities we thought a computer program could
feature.  This expansion is fueled by an increasing variety of
operations that can be incorporated into Neural Networks, by a richer
set of arrangments that these operations can be configured in and by
improved computational support for these improvements.  In general, we
can categorize these new Neural Architectures, and their use cases
they enable, in (a more complete description can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning#artificial-neural-networks) ):

* Feed-forward Neural Network
* Convolutional Neural Network
* Recurrent Neural Network
* Generative Adversarial Network
* Transformers

# What can I use deep learning for?

As stated above, the scope of application of DL techniques is rapidly
expanding.  DL architectures, however, have shown amazing
(close-to-human in some cases) performance in tasks having to do with
"unstructured data": images, audio, free-form text and the like.  In
this way, DL is constantly featured in image/audio classification and
generation applications.  When it comes to text processing, more
generally Natural Language Processing, DL methods have shown amazing
results in tasks like translation, classification, generation and
similar.  Some of the more spectacular, recent applications of ML,
such as "[Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion)" are powered by sophisticated, large Neural
Network architectures.

# Deep learning in ML.NET

A central concern of DL is what Neural Network architecture (specific configuration of operations) will the model have, and to this end, DL frameworks like Tensorflow and Pytorch feature expressive Domain-Specific Languages to describe in detail such architectures.  ML.NET departs from this practice and concentrates on the consumption of pre-trained models (i.e., architectures that have been specified *and* trained in other frameworks).

# Train custom models

Many Deep Learning frameworks include facilities to build up a Deep Learning architecture out of individual operations, and  to construct a training loop.  Training Deep Learning models usually requires a significant amount of training data and compute, which is why ML.NET has implemented a "separable" training/inference API, that allows training a model (in, for example, [the Cloud](https://devblogs.microsoft.com/dotnet/training-a-ml-dotnet-model-with-azure-ml/) to make use of bigger and more plentiful resources) and consuming the result in a client/edge or other deployment scenarios with more limited resources.

<!--
# Image classification
-->

# Text classification 
<!-- (Needs tutorial) -->

As stated above, Deep Learning has proven to be especially useful in tasks involving non-structured data, like audio, video, images or text.  ML.NET provides a rich API for working with text, powered by state-of-the-art Deep Learning models (specifically, [NAS-BERT](https://dl.acm.org/doi/abs/10.1145/3447548.3467262)).  The API is high-level, though, so you can treat the actual models powering it as an implementation detail, and concentrate of the functionality of your application.

An end-to-end example of this API can be found [here](https://github.com/dotnet/csharp-notebooks/blob/main/machine-learning/E2E-Text-Classification-API-with-Yelp-Dataset.ipynb), but a snippet is shown below as an illustration.   Here, the input (for the training task) is assumed as a `IDataView` that includes phrases with their corresponding label (the strings "positive" or "negative" to describe the sentiment they express):
```
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label","Sentiment")
			.Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "Text"))
			.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            
var model = pipeline.Fit(trainingData);
```
As you can see, the pipeline recodes the sentiment category, and configures the `TextClassification` trainer to use the relevant column in the input.  The call to `Fit` trains the model, that then can be used to predict the labels on unseen data with a call like:
```
var predictedData = model.Transform(newData);
```

# Sentence Similarity 
<!-- (Needs tutorial - P1) -->

Many important tasks in NLP (for example, text summarization or semantic search) rely on the determination of how similar two sentences or passages of text are to each other.  Deep Learning models in general, and Transformers in particular, are especially well-suited to carry out this operation since they represent sentence in a mathematical space that captures the *meaning* of each sentence (in contrast with other ways to measure similarity that concentrate on the *form* -- for example, methods like the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)).   ML.NET provides a high-level API (also powered by NAS-BERT) that takes, for its training input, a set of sentence pairs and their associated similarity, coded as a single-precision floating-point number, in the range 0-1, where closer to 1 indicates higher similarity between the sentences.  With this on hand, an ML.NET pipeline (for training/scoring) would look like:
```
var pipeline = mlContext.Regression.Trainers.SentenceSimilarity(sentence1ColumnName: "Sentence", sentence2ColumnName: "Sentence2");

var model = pipeline.Fit(trainingData);
var score = model.Transform(newData).GetColumn<float>("Score");
```

# Consume pretrained models

ML.NET provides access to many specialized libraries for Machine Learning.  This is evident in many "traditional" Machine Learning tasks, like gradient boosted trees, but is especially useful in Deep Learning, where it allows users to build applications from the great variety of pre-trained models that are available as open-source.  In the following subsections, we describe this workflow for solving Image Classification problems, but this functionality applies to many other scenarios.

## TensorFlow  

[Tensorflow](https://www.tensorflow.org/) is one of best-known and most mature Deep Learning frameworks in existence today.  It has a rich ecosystem, including a great variety of pre-trained models in the "[Tensorflow Hub](https://www.tensorflow.org/hub)".  ML.NET  makes it straightforward to make use of those models, define appropriate pre- and post-processing, and assemble a pipeline that can drive the ML tasks of your app.

You can find a "Getting Started" guide, [here](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_TensorFlow) for example (and there are more references cited below).  At the heart of these examples, you'll find pipelines like:
```
var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageNetData.ImagePath))
    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "input"))
    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean))
    .Append(mlContext.Model.LoadTensorFlowModel(modelLocation)
        .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2" }, inputColumnNames: new[] { "input" },
            addBatchDimensionInput:true));
```
where, taking an input a csv file that lists images (providing the path to their location on disk), the pipeline carries out some preprocessing (here illustrated by the `ResizeImages` and `ExtractPixels` transformations), applies a TensorFlow model downloaded from the Hub (and referenced in the code by the variable `modelLocation`, which contains its path in disk) and outputs a  `IDataView` that provides the probability of the corresponding image being associated with a label (using a "[softmax](https://en.wikipedia.org/wiki/Softmax_function)" operation, a common practice in DL classification tasks).

The pipeline is somewhat verbose, owing to the many configuration knobs that pre-trained models have (which give them their flexibility), but relatively easy to read and understand.  Moreover, the `ImageAnalytics` preprocessing operation ML.NET includes make the very non-trivial task of pre-processing much easier.

### References

* https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/text-classification-tf
* https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_TensorFlow

## ONNX 

[ONNX](https://onnx.ai/), or "Open Neural Network Exchange" is an industry-wide format designed to make Deep Learning frameworks interoperable.  Throughout the years, the scope of ONNX has expanded beyond Neural Network models, and it has recently also acquired support for training (it was originally only for inference).   Also, and very much like TensorFlow, it is at the center of a growing ecosystem, which also includes a repository for pre-trained models.  The [ONNX Model Zoo](https://github.com/onnx/models) hosts many models pre-trained in a great variety of datasets, and optimized for a variety of scenarios (including, for example models that support specialized precisions like BFloat or INT8, or that have been quantized).

Like it does with TensorFlow, ML.NET makes it straightforward to consume pre-trained models in the ONNX format.  The pipeline is very similar (this is a snippet from the sample scenario referenced below):
```
var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
        .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
        .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

```

Note that the pre-processing stages are effectively the same as those in the TensorFlow example, variations only introduced to accommodate the requirements of the model. 

### References

* https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx

