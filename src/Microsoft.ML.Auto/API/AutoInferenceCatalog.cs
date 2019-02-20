// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Auto
{
    public class AutoInferenceCatalog
    {
        private readonly MLContext _context;

        internal AutoInferenceCatalog(MLContext context)
        {
            _context = context;
        }

        public RegressionExperiment CreateRegressionExperiment(uint maxInferenceTimeInSeconds)
        {
            return new RegressionExperiment(_context, new RegressionExperimentSettings()
            {
                MaxInferenceTimeInSeconds = maxInferenceTimeInSeconds
            });
        }

        public RegressionExperiment CreateRegressionExperiment(RegressionExperimentSettings experimentSettings)
        {
            return new RegressionExperiment(_context, experimentSettings);
        }

        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(uint maxInferenceTimeInSeconds)
        {
            return new BinaryClassificationExperiment(_context, new BinaryExperimentSettings()
            {
                MaxInferenceTimeInSeconds = maxInferenceTimeInSeconds
            });
        }

        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(BinaryExperimentSettings experimentSettings)
        {
            return new BinaryClassificationExperiment(_context, experimentSettings);
        }

        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(uint maxInferenceTimeInSeconds)
        {
            return new MulticlassClassificationExperiment(_context, new MulticlassExperimentSettings()
            {
                MaxInferenceTimeInSeconds = maxInferenceTimeInSeconds
            });
        }

        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(MulticlassExperimentSettings experimentSettings)
        {
            return new MulticlassClassificationExperiment(_context, experimentSettings);
        }

        public ColumnInferenceResults InferColumns(string path, string label,char? separatorChar = null, bool? allowQuotedStrings = null, 
            bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            //UserInputValidationUtil.ValidateInferColumnsArgs(path, label);
            return ColumnInferenceApi.InferColumns(_context, path, label, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public ColumnInferenceResults InferColumns(string path, uint labelColumnIndex, bool hasHeader = false, char? separatorChar = null, 
            bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            //UserInputValidationUtil.ValidateInferColumnsArgs(path);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnIndex, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }
    }
}
