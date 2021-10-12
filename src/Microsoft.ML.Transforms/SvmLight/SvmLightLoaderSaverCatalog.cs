// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public static class SvmLightLoaderSaverCatalog
    {
        /// <summary>
        /// Creates a loader that loads SVM-light format files. <see cref="SvmLightLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="inputSize">The number of features in the Features column. If 0 is specified, the
        /// loader will determine it by looking at the file sample given in <paramref name="dataSample"/>.</param>
        /// <param name="numberOfRows">The number of rows from the sample to be used for determining the number of features.</param>
        /// <param name="zeroBased">If the file contains zero-based indices, this parameter should be set to true. If they are one-based
        /// it should be set to false.</param>
        /// <param name="dataSample">A data sample to be used for determining the number of features in the Features column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadingSvmLight](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/LoadingSvmLight.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SvmLightLoader CreateSvmLightLoader(this DataOperationsCatalog catalog,
            long? numberOfRows = null,
            int inputSize = 0,
            bool zeroBased = false,
            IMultiStreamSource dataSample = null)
            => new SvmLightLoader(CatalogUtils.GetEnvironment(catalog), new SvmLightLoader.Options()
            {
                InputSize = inputSize,
                NumberOfRows = numberOfRows,
                FeatureIndices = zeroBased ?
                SvmLightLoader.FeatureIndices.ZeroBased : SvmLightLoader.FeatureIndices.OneBased
            }, dataSample);

        /// <summary>
        /// Creates a loader that loads SVM-light like files, where features are specified by their names.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="numberOfRows">The number of rows from the sample to be used for determining the set of feature names.</param>
        /// <param name="dataSample">A data sample to be used for determining the set of features names.</param>
        public static SvmLightLoader CreateSvmLightLoaderWithFeatureNames(this DataOperationsCatalog catalog,
            long? numberOfRows = null,
            IMultiStreamSource dataSample = null)
            => new SvmLightLoader(CatalogUtils.GetEnvironment(catalog), new SvmLightLoader.Options()
            { NumberOfRows = numberOfRows, FeatureIndices = SvmLightLoader.FeatureIndices.Names }, dataSample);

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="SvmLightLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file.</param>
        /// <param name="inputSize">The number of features in the Features column. If 0 is specified, the
        /// loader will determine it by looking at the file given in <paramref name="path"/>.</param>
        /// <param name="zeroBased">If the file contains zero-based indices, this parameter should be set to true. If they are one-based
        /// it should be set to false.</param>
        /// <param name="numberOfRows">The number of rows from the sample to be used for determining the number of features.</param>
        public static IDataView LoadFromSvmLightFile(this DataOperationsCatalog catalog,
            string path,
            long? numberOfRows = null,
            int inputSize = 0,
            bool zeroBased = false)
        {
            Contracts.CheckNonEmpty(path, nameof(path));
            if (!File.Exists(path))
            {
                throw Contracts.ExceptParam(nameof(path), "File does not exist at path: {0}", path);
            }

            var file = new MultiFileSource(path);
            var loader = catalog.CreateSvmLightLoader(numberOfRows, inputSize, zeroBased, file);
            return loader.Load(file);
        }

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file containing features specified by feature names,
        /// using <see cref="SvmLightLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file.</param>
        /// <param name="numberOfRows">The number of rows from the sample to be used for determining the set of feature names.</param>
        public static IDataView LoadFromSvmLightFileWithFeatureNames(this DataOperationsCatalog catalog,
            string path,
            long? numberOfRows = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));
            if (!File.Exists(path))
            {
                throw Contracts.ExceptParam(nameof(path), "File does not exist at path: {0}", path);
            }

            var file = new MultiFileSource(path);
            var loader = catalog.CreateSvmLightLoaderWithFeatureNames(numberOfRows, file);
            return loader.Load(file);
        }

        /// <summary>
        /// Save the <see cref="IDataView"/> in SVM-light format. Four columns can be saved: a label and a features column,
        /// and optionally a group ID column and an example weight column.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="zeroBasedIndexing">Whether to index the features starting at 0 or at 1.</param>
        /// <param name="binaryLabel">If set to true, saves 1 for positive labels, -1 for non-positive labels and 0 for NaN.
        /// Otherwise, saves the value of the label in the data view.</param>
        /// <param name="labelColumnName">The name of the column to be saved as the label column.</param>
        /// <param name="featureColumnName">The name of the column to be saved as the features column.</param>
        /// <param name="rowGroupColumnName">The name of the column to be saved as the group ID column. If null, a group ID column
        /// will not be saved.</param>
        /// <param name="exampleWeightColumnName">The name of the column to be saved as the weight column. If null, a weight column
        /// will not be saved.</param>
        public static void SaveInSvmLightFormat(this DataOperationsCatalog catalog,
            IDataView data,
            Stream stream,
            bool zeroBasedIndexing = false,
            bool binaryLabel = false,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = null,
            string exampleWeightColumnName = null)
        {
            var args = new SvmLightSaver.Arguments()
            {
                Zero = zeroBasedIndexing,
                Binary = binaryLabel,
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                RowGroupColumnName = rowGroupColumnName,
                ExampleWeightColumnName = exampleWeightColumnName
            };

            var saver = new SvmLightSaver(CatalogUtils.GetEnvironment(catalog), args);
            saver.SaveData(stream, data, data.Schema.Select(col => col.Index).ToArray());
        }
    }
}
