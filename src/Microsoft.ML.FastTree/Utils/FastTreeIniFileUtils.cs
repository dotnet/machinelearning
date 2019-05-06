// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    internal static class FastTreeIniFileUtils
    {
        public static string TreeEnsembleToIni(
            IHost host, InternalTreeEnsemble ensemble, RoleMappedSchema schema, ICalibrator calibrator,
            string trainingParams, bool appendFeatureGain, bool includeZeroGainFeatures)
        {
            host.CheckValue(ensemble, nameof(ensemble));
            host.CheckValue(schema, nameof(schema));

            string ensembleIni = ensemble.ToTreeEnsembleIni(new FeaturesToContentMap(schema),
                trainingParams, appendFeatureGain, includeZeroGainFeatures);
            ensembleIni = AddCalibrationToIni(host, ensembleIni, calibrator);
            return ensembleIni;
        }

        /// <summary>
        /// Get the calibration summary in INI format
        /// </summary>
        private static string AddCalibrationToIni(IHost host, string ini, ICalibrator calibrator)
        {
            host.AssertValue(ini);
            host.AssertValueOrNull(calibrator);

            if (calibrator == null)
                return ini;

            if (calibrator is PlattCalibrator)
            {
                string calibratorEvaluatorIni = IniFileUtils.GetCalibratorEvaluatorIni(ini, calibrator as PlattCalibrator);
                return IniFileUtils.AddEvaluator(ini, calibratorEvaluatorIni);
            }
            else
            {
                StringBuilder newSection = new StringBuilder();
                newSection.AppendLine();
                newSection.AppendLine();
                newSection.AppendLine("[TLCCalibration]");
                newSection.AppendLine("Type=" + calibrator.GetType().Name);
                return ini + newSection;
            }
        }
    }
}
