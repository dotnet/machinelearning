// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class AutoMLCatalog
    {
        private readonly MLContext _context;

        internal AutoMLCatalog(MLContext context)
        {
            _context = context;
        }

        public RegressionExperiment CreateRegressionExperiment(uint maxExperimentTimeInSeconds)
        {
            return new RegressionExperiment(_context, new RegressionExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        public RegressionExperiment CreateRegressionExperiment(RegressionExperimentSettings experimentSettings)
        {
            return new RegressionExperiment(_context, experimentSettings);
        }

        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new BinaryClassificationExperiment(_context, new BinaryExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(BinaryExperimentSettings experimentSettings)
        {
            return new BinaryClassificationExperiment(_context, experimentSettings);
        }

        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new MulticlassClassificationExperiment(_context, new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(MulticlassExperimentSettings experimentSettings)
        {
            return new MulticlassClassificationExperiment(_context, experimentSettings);
        }

        public ColumnInferenceResults InferColumns(string path, string labelColumn = DefaultColumnNames.Label, char? separatorChar = null, bool? allowQuotedStrings = null, 
            bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, labelColumn);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumn, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public ColumnInferenceResults InferColumns(string path, ColumnInformation columnInformation, char? separatorChar = null, bool? allowQuotedStrings = null,
            bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            columnInformation = columnInformation ?? new ColumnInformation();
            UserInputValidationUtil.ValidateInferColumnsArgs(path, columnInformation);
            return ColumnInferenceApi.InferColumns(_context, path, columnInformation, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public ColumnInferenceResults InferColumns(string path, uint labelColumnIndex, bool hasHeader = false, char? separatorChar = null, 
            bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnIndex, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }
    }
}
