// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// A catalog of all available AutoML tasks.
    /// </summary>
    public sealed class AutoCatalog
    {
        private readonly MLContext _context;

        internal AutoCatalog(MLContext context)
        {
            _context = context;
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a regression dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML regression experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentResult = mlContext.Auto()
        ///     .CreateRegressionExperiment(60)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public RegressionExperiment CreateRegressionExperiment(uint maxExperimentTimeInSeconds)
        {
            return new RegressionExperiment(_context, new RegressionExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a regression dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML regression experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentSettings = new RegressionExperimentSettings()
        /// {
        ///     MaxExperimentTimeInSeconds = 60
        /// };
        /// var experimentResult = mlContext.Auto()
        ///     .CreateRegressionExperiment(experimentSettings)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public RegressionExperiment CreateRegressionExperiment(RegressionExperimentSettings experimentSettings)
        {
            return new RegressionExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a binary classification dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML binary classification experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentResult = mlContext.Auto()
        ///     .CreateBinaryClassificationExperiment(60)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new BinaryClassificationExperiment(_context, new BinaryExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a binary classification dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML binary classification experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentSettings = new BinaryExperimentSettings()
        /// {
        ///     MaxExperimentTimeInSeconds = 60
        /// };
        /// var experimentResult = mlContext.Auto()
        ///     .CreateBinaryClassificationExperiment(experimentSettings)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(BinaryExperimentSettings experimentSettings)
        {
            return new BinaryClassificationExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a multiclass classification dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML multiclass classification experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentResult = mlContext.Auto()
        ///     .CreateMulticlassClassificationExperiment(60)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new MulticlassClassificationExperiment(_context, new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a binary classification dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML multiclass classification experiment.</returns>
        /// <example>
        /// <code>
        /// MLContext mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInferenceResults = mlContext.Auto().InferColumns(@"C:\dataset\train.csv", "LabelColumn");
        /// 
        /// // Load data
        /// var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// IDataView trainDataView = textLoader.Load(@"C:\dataset\train.csv");
        /// 
        /// // Run an AutoML experiment
        /// var experimentSettings = new MulticlassExperimentSettings()
        /// {
        ///     MaxExperimentTimeInSeconds = 60
        /// };
        /// var experimentResult = mlContext.Auto()
        ///     .CreateMulticlassClassificationExperiment(experimentSettings)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(MulticlassExperimentSettings experimentSettings)
        {
            return new MulticlassClassificationExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. If null, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain column defined by a quoted string. If null, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If null, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to 
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to 
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment, 
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains AutoML's best guess for the purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data,? Should the column be ignored? Etc). 
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and passed to an AutoML experiment for running.
        /// </remarks>
        /// <example>
        /// <code>
        /// var mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(@"C:\dataset\train_data.csv", "LabelColumn");
        /// 
        /// // Load data using inferred columns
        /// TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// 
        /// // Run an AutoML experiment using the inferred columns.
        /// var experiment = new MLContext().Auto()
        ///     .CreateRegressionExperiment(60)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public ColumnInferenceResults InferColumns(string path, string labelColumnName = DefaultColumnNames.Label, char? separatorChar = null, bool? allowQuoting = null, 
            bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, labelColumnName);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnName, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="columnInformation">Column information for the dataset.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. If null, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain column defined by a quoted string. If null, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If null, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to 
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to 
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment, 
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains AutoML's best guess for the purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data,? Should the column be ignored? Etc). 
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and passed to an AutoML experiment for running.
        /// </remarks>
        /// <example>
        /// <code>
        /// var mlContext = new MLContext();
        /// 
        /// // Infer columns
        /// var columnInformation = new ColumnInformation()
        /// {
        ///     LabelColumnName = "LabelColumn"
        /// };
        /// ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(@"C:\dataset\train_data.csv", columnInformation);
        /// 
        /// // Load data using inferred columns
        /// TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        /// 
        /// // Run an AutoML experiment using the inferred columns.
        /// var experiment = new MLContext().Auto()
        ///     .CreateRegressionExperiment(60)
        ///     .Execute(trainDataView, columnInference.ColumnInformation);
        /// </code>
        /// </example>
        public ColumnInferenceResults InferColumns(string path, ColumnInformation columnInformation, char? separatorChar = null, bool? allowQuoting = null,
            bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            columnInformation = columnInformation ?? new ColumnInformation();
            UserInputValidationUtil.ValidateInferColumnsArgs(path, columnInformation);
            return ColumnInferenceApi.InferColumns(_context, path, columnInformation, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="labelColumnIndex">Column index of the label column in the dataset.</param>
        /// <param name="hasHeader">Whether or not the dataset file has a header row.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. If null, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain column defined by a quoted string. If null, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If null, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to 
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to 
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment, 
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains AutoML's best guess for the purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data,? Should the column be ignored? Etc). 
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and passed to an AutoML experiment for running.
        /// </remarks>
        public ColumnInferenceResults InferColumns(string path, uint labelColumnIndex, bool hasHeader = false, char? separatorChar = null, 
            bool? allowQuoting = null, bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnIndex, hasHeader, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }
    }
}
