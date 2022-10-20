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

# Image classification

# Text classification (Needs tutorial)
# Sentence Similarity (Needs tutorial - P1)

# Consume pretrained models

# TensorFlow  https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/text-classification-tf
# ONNX https://github.com/dotnet/csharp-notebooks/blob/main/machine-learning/E2E-Text-Classification-API-with-Yelp-Dataset.ipynb
