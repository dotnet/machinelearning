// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    public static class SvmLightLoaderSaverCatalog
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="inputSize"></param>
        /// <param name="numberOfRows"></param>
        /// <param name="zeroBased"></param>
        /// <param name="dataSample"></param>
        /// <returns></returns>
        public static SvmLightLoader CreateSvmLightLoader(this DataOperationsCatalog catalog,
            int inputSize = 0,
            long? numberOfRows = null,
            bool zeroBased = false,
            IMultiStreamSource dataSample = null)
            => new SvmLightLoader(CatalogUtils.GetEnvironment(catalog), new SvmLightLoader.Options()
            { InputSize = inputSize, NumberOfRows = numberOfRows, FeatureIndices = zeroBased ?
                SvmLightLoader.FeatureIndices.ZeroBased : SvmLightLoader.FeatureIndices.OneBased }, dataSample);

        /// <summary>
        ///
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="numberOfRows"></param>
        /// <param name="dataSample"></param>
        /// <returns></returns>
        public static SvmLightLoader CreateSvmLightLoaderWithFeatureNames(this DataOperationsCatalog catalog,
            long? numberOfRows = null,
            IMultiStreamSource dataSample = null)
            => new SvmLightLoader(CatalogUtils.GetEnvironment(catalog), new SvmLightLoader.Options()
            { NumberOfRows = numberOfRows, FeatureIndices = SvmLightLoader.FeatureIndices.Names }, dataSample);

        /// <summary>
        ///
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="path"></param>
        /// <param name="inputSize"></param>
        /// <param name="zeroBased"></param>
        /// <param name="numberOfRows"></param>
        /// <returns></returns>
        public static IDataView LoadFromSvmLightFile(this DataOperationsCatalog catalog,
            string path,
            int inputSize = 0,
            bool zeroBased = false,
            long? numberOfRows = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var file = new MultiFileSource(path);
            var loader = catalog.CreateSvmLightLoader(inputSize, numberOfRows, zeroBased, file);
            return loader.Load(file);
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="path"></param>
        /// <param name="numberOfRows"></param>
        /// <returns></returns>
        public static IDataView LoadFromSvmLightFileWithFeatureNames(this DataOperationsCatalog catalog,
            string path,
            long? numberOfRows = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var file = new MultiFileSource(path);
            var loader = catalog.CreateSvmLightLoaderWithFeatureNames(numberOfRows, file);
            return loader.Load(file);
        }

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
