// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.ObjectModel;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Contains information AutoML inferred about columns in a dataset.
    /// </summary>
    public sealed class ColumnInferenceResults
    {
        /// <summary>
        /// Inferred <see cref="TextLoader.Options" /> for the dataset.
        /// </summary>
        /// <remarks>
        /// Can be used to instantiate a new <see cref="TextLoader" /> to load
        /// data into an <see cref="IDataView" />.
        /// </remarks>
        public TextLoader.Options TextLoaderOptions { get; internal set; }

        /// <summary>
        /// Information about the inferred columns in the dataset.
        /// </summary>
        /// <remarks>
        /// <para>Contains the inferred purposes of each column. See <see cref="AutoML.ColumnInformation"/> for more details.</para>
        /// <para>This can be fed to the AutoML API when running an experiment.
        /// See <typeref cref="ExperimentBase{TMetrics, TExperimentSettings}.Execute(IDataView, ColumnInformation, IEstimator{ITransformer}, System.IProgress{RunDetail{TMetrics}})" />
        /// for example.</para>
        /// </remarks>
        public ColumnInformation ColumnInformation { get; internal set; }
    }

    /// <summary>
    /// Information about the columns in a dataset.
    /// </summary>
    /// <remarks>
    /// <para>Contains information about the purpose of each column in the dataset. For instance,
    /// it enumerates the dataset columns that AutoML should treat as categorical,
    /// the columns AutoML should ignore, which column is the label, etc.</para>
    /// <para><see cref="ColumnInformation"/> can be fed to the AutoML API when running an experiment.
    /// See <typeref cref="ExperimentBase{TMetrics, TExperimentSettings}.Execute(IDataView, ColumnInformation, IEstimator{ITransformer}, System.IProgress{RunDetail{TMetrics}})" />
    /// for example.</para>
    /// </remarks>
    public sealed class ColumnInformation
    {
        /// <summary>
        /// The dataset column to use as the label.
        /// </summary>
        /// <value>The default value is "Label".</value>
        public string LabelColumnName { get; set; }

        /// <summary>
        /// The dataset column to use for example weight.
        /// </summary>
        public string ExampleWeightColumnName { get; set; }

        /// <summary>
        /// The dataset column to use for grouping rows.
        /// If two examples share the same sampling key column name,
        /// they are guaranteed to appear in the same subset (train or test).
        /// This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/>, no row grouping will be performed.
        /// </summary>
        public string SamplingKeyColumnName { get; set; }

        /// <summary>
        /// The dataset columns that are categorical.
        /// </summary>
        /// <value>The default value is a new, empty <see cref="Collection{String}"/>.</value>
        /// <remarks>
        /// Categorical data columns should generally be columns that contain a small number of unique values.
        /// </remarks>
        public ICollection<string> CategoricalColumnNames { get; }

        /// <summary>
        /// The dataset columns that are numeric.
        /// </summary>
        /// <value>The default value is a new, empty <see cref="Collection{String}"/>.</value>
        public ICollection<string> NumericColumnNames { get; }

        /// <summary>
        /// The dataset columns that are text.
        /// </summary>
        /// <value>The default value is a new, empty <see cref="Collection{String}"/>.</value>
        public ICollection<string> TextColumnNames { get; }

        /// <summary>
        /// The dataset columns that AutoML should ignore.
        /// </summary>
        /// <value>The default value is a new, empty <see cref="Collection{String}"/>.</value>
        public ICollection<string> IgnoredColumnNames { get; }

        public ColumnInformation()
        {
            LabelColumnName = DefaultColumnNames.Label;
            CategoricalColumnNames = new Collection<string>();
            NumericColumnNames = new Collection<string>();
            TextColumnNames = new Collection<string>();
            IgnoredColumnNames = new Collection<string>();
        }
    }
}