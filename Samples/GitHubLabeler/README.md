## Goal
This is a simple prototype application to demonstrate how to use [ML.NET](https://www.nuget.org/packages/Microsoft.ML/) APIs. The main focus is on creating, training, and using ML (Machine Learning) model that is implemented in Predictor.cs class.

## Overview
GitHubLabeler is a .NET Core console application that runs from command-line interface (CLI) and allows you to:
* train ML model on your labeled GitHub issues to teach the model what label should be assigned for a new issue. (As an example, you can use `corefx_issues.tsv` file that contains issues from public [corefx](https://github.com/dotnet/corefx) repository)
* label a new issue. The application will get all unlabeled open issues from specified in `App.config` GitHub repository and label them using the trained ML model created on the step above.  

This ML model is using multi-class classification algorithm and text capabilities (`TextFeaturizer`) of [ML.NET](https://www.nuget.org/packages/Microsoft.ML/).

## Enter you data
1. **Provide your GitHub data** in `App.config`:
    ```csharp
        <add key="GitHubToken" value="" />
        <add key="GitHubUserName" value="" />
        <add key="GitHubRepoName" value="" />
    ```
    Your user account (`GitHubToken`) should have write rights to the repository (`GitHubRepoName`).
2. **Provide training file**

    a.  You can use existing `corefx_issues.tsv` data file for experimenting  with the program. In this case the predicted labels will be chosen among labels from [corefx](https://github.com/dotnet/corefx) repository. No changes required.
    
    b. To work with labels from your GitHub repository, you will need to train the model on your data. To do so, export GitHub issues from your repository into `.tsv` file with the following columns:
    * ID - issue's ID
    * Area - issue's label (named this way to avoid confusion with the Label concept in ML.NET)
    * Title - issue's title
    * Description - issue's description
    
    and add the file in `Data` folder. Update `Predictor.DataPath` field to match your file's name:
```csharp
private static string DataPath => Path.Combine(AppPath, "Data", "corefx_issues.tsv");
```

## Training 
Training is a process of running an ML model through known examples (in our case - issues with labels) and teaching it how to label new issues.

To perform the training run the application with `"train"` key:
```
$ dotnet run -- train
```
After the training is completed, the model is saved as a .zip file in `Models\Model.zip`.

## Labeling
When the model is trained, it can be used for predicting new issue's label. To do so, run the application with `"label"` key:
```
$ dotnet run -- label
```
For testing convenience only open and not labeled issues that were created in the past 10 minutes are subject to labeling:
```csharp
Since = DateTime.Now.AddMinutes(-10)
```
You can modify those settings. After predicting the label, the program updates the issue with the predicted label on GitHub.