// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.RunTests
{
    using TestLearners = TestLearnersBase;

#if OLD_TESTS // REVIEW: Need to port the INI stuff.
    /// <summary>
    /// Summary description for TestIniModels
    /// </summary>
    public sealed class TestIniModels : BaseTestPredictorsOld
    {
        private const string IniSubDirectory = @"Ini";
        private const string EvaluationExecutorDir = @"RankerEval2";
        private const string EvaluationCommandLineFormat = "NeuralNetRankerEval2.exe /InputFile:\"{0}\" /OutputDir:\"{1}\" /DataFile:\"{2}\" /OutputDocRanking";

        /// <summary>
        /// Get a list of datasets for INI model test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForIniTest()
        {
            return new TestDataset[] {
                TestDatasets.breastCancerBing,
                TestDatasets.rankingExtract
            };
        }

        /// <summary>
        /// Run a train unit test
        /// </summary>
        public InternalLearnRunParameters TrainForIniModel(
            PredictorAndArgs predictor,
            string trainDataset,
            string outName,
            string[] extra = null,
            ModelType.ModelKind modelKind = ModelType.ModelKind.Model)
        {
            InternalLearnRunParameters runParams = new InternalLearnRunParameters
            {
                Command = "Train",
                Trainer = predictor.Trainer,
                Dataset = trainDataset,
                BaselineDir = IniSubDirectory,
                BaselineFilename = outName,
                ModelFilename = outName + ModelType.GetModelType(modelKind).ModelExtension,
                ModelType = ModelType.GetModelType(modelKind),
                extraArgs = extra,
                NoTest = true,
            };
            RunPredictor(runParams);
            return runParams;
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI")]
        public void TestPerceptronNotNormIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.perceptronNotNorm },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */
            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI")]
        public void TestLinearSVMNotNormIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.linearSVMNotNorm },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */
            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI")]
        public void TestLogisticRegressionIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.logisticRegression_tlOld },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */
            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), Priority(2)]
        // REVIEW : This test fails when run, but when debugged it succeeds
        public void TestLogisticRegressionSGDIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.logisticRegressionSGD },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */
            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), TestCategory("FastRank"), Priority(2)]
        public void TestFastRankClassificationIniModels()
        {
            // Inconsistent baseline comparison among different hardware
            using (var ctx = new MismatchContext(this))
            {
                RunMTAThread(() =>
                {
                    RunAllIniFileEvaluationTests(
                        new PredictorAndArgs[] { TestLearners.fastRankClassification },
                        GetDatasetsForIniTest());
                });
            }

            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), TestCategory("FastRank"), Priority(2)]
        public void TestFastRankRegressionIniModels()
        {
            // Inconsistent baseline comparison among different hardware
            using (var ctx = new MismatchContext(this))
            {
                RunMTAThread(() =>
                {
                    RunAllIniFileEvaluationTests(
                        new PredictorAndArgs[] { TestLearners.fastRankRegression },
                        GetDatasetsForIniTest());
                });
            }

            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), TestCategory("FastRank"), Priority(2)]
        public void TestFastRankRankingIniModels()
        {
            // Inconsistent baseline comparison among different hardware
            using (var ctx = new MismatchContext(this))
            {
                RunMTAThread(() =>
                {
                    RunAllIniFileEvaluationTests(
                        new PredictorAndArgs[] { TestLearners.fastRankRanking },
                        GetDatasetsForIniTest());
                });
            }

            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), TestCategory("Neural Nets"), Priority(2)]
        public void TestNnMultiDefaultIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.NnMultiDefault(5) },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */

            Done();
        }

        /// <summary>
        /// The main entry to test INI models.
        /// </summary>
        [Fact, TestCategory("Test INI"), TestCategory("Neural Nets"), Priority(2)]
        public void TestNnMultiMomentumIniModels()
        {
            RunAllIniFileEvaluationTests(
                new PredictorAndArgs[] { TestLearners.NnMultiMomentum(5) },
                GetDatasetsForIniTest());
            /* NOTE: 1. NeuralNetRankerEval2.exe cannot process breast-cancer dataset because it "could not find query id column :-1"
             *       2. BinaryNeuralNetwork requires two outputs, so that we cannot use TestDatasets.ranking dataset for
             *          BinaryNeuralNetwork.
                RunAllIniFileEvaluationTests(
                    GetPredictorsForNnTestBinary(),
                    GetDatasetsForIniTest()
                    );
            }
            */

            Done();
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        /// <summary>
        /// Data structure for storing test information on Ini Models.
        /// </summary>
        public class IniModelTestInformation
        {
            public IniModelTestInformation(
                string modelFilePath,
                string trainDatasetPath,
                string evaluationOutputDir,
                string evaluationCommandLine,
                InternalLearnRunParameters runParameters,
                ProcessDebugInformation processDebugInformation,
                KeyValuePair<Exception, List<string>> baselineDebugInformation
                )
            {
                this.ModelFilePath = modelFilePath;
                this.TrainDatasetPath = trainDatasetPath;
                this.EvaluationOutputDir = evaluationOutputDir;
                this.EvaluationCommandLine = evaluationCommandLine;
                this.RunParameters = runParameters;
                this.ProcessDebugInformation = processDebugInformation;
                this.BaselineDebugInformation = baselineDebugInformation;
            }
            public readonly string ModelFilePath;
            public readonly string TrainDatasetPath;
            public readonly string EvaluationOutputDir;
            public readonly string EvaluationCommandLine;
            public readonly InternalLearnRunParameters RunParameters;
            public readonly ProcessDebugInformation ProcessDebugInformation;
            public readonly KeyValuePair<Exception, List<string>> BaselineDebugInformation;
        }

        /// <summary>
        /// Run INI test for a collection of combinationss of predictors and datasets.
        /// </summary>
        /// <param name="predictors"></param>
        /// <param name="datasets"></param>
        /// <param name="extraSettings"></param>
        /// <param name="extraTag"></param>
        public void RunAllIniFileEvaluationTests(
            IList<PredictorAndArgs> predictors,
            IList<TestDataset> datasets,
            string[] extraSettings = null,
            string extraTag = "")
        {
            Contracts.Assert(IsActive);
            string evaluationOutputDirRoot = GetOutputDir(IniSubDirectory);
            List<IniModelTestInformation> successTestInformation = new List<IniModelTestInformation>();
            List<IniModelTestInformation> failureTestInformation = new List<IniModelTestInformation>();
            foreach (TestDataset dataset in datasets)
            {
                foreach (PredictorAndArgs predictor in predictors)
                {
                    RunIniFileEvaluationTest(
                        successTestInformation,
                        failureTestInformation,
                        predictor,
                        dataset,
                        IniSubDirectory,
                        extraSettings,
                        extraTag
                        );
                }
            }
            Assert.IsTrue(failureTestInformation.Count <= 0);
        }

        /// <summary>
        /// Run INI test for a pair of predictor and dataset.
        /// </summary>
        /// <param name="debugInformation"></param>
        /// <param name="predictor"></param>
        /// <param name="dataset"></param>
        /// <param name="evaluationOutputDirPrefix"></param>
        /// <param name="extraSettings"></param>
        /// <param name="extraTag"></param>
        public void RunIniFileEvaluationTest(
            List<IniModelTestInformation> successTestInformation,
            List<IniModelTestInformation> failureTestInformation,
            PredictorAndArgs predictor,
            TestDataset dataset,
            string evaluationOutputDirPrefix,
            string[] extraSettings = null,
            string extraTag = ""
            )
        {
            string outName = ExpectedFilename("Train", predictor, dataset, extraTag);
            string[] extraTrainingSettings = JoinOptions(GetInstancesSettings(dataset), extraSettings);
            string trainDataset = dataset.testFilename;
            InternalLearnRunParameters runParameters = TrainForIniModel(
                predictor,
                trainDataset,
                outName,
                extraTrainingSettings,
                ModelType.ModelKind.Ini);

            CheckEqualityNormalized(runParameters.BaselineDir, runParameters.ModelFilename);
            string modelFilePath = GetOutputPath(runParameters.BaselineDir, runParameters.ModelFilename);
            string trainDatasetPath = GetDataPath(trainDataset);
            string evaluationOutputDir = GetOutputDir(evaluationOutputDirPrefix + @"\Dirs\" + outName);
            Assert.IsNull(EnsureEmptyDirectory(evaluationOutputDir));

            string cmd = string.Format(EvaluationCommandLineFormat, modelFilePath, evaluationOutputDir, trainDatasetPath);
            string dir = Path.GetFullPath(EvaluationExecutorDir);
            Log("Working directory for evaluation: {0}", dir);
            Log("Evaluation command line: {0}", cmd);
            ProcessDebugInformation processDebugInformation = RunCommandLine(cmd, dir);

            if (processDebugInformation.ExitCode == 0)
            {
                KeyValuePair<Exception, List<string>> baselineCheckDebugInformation =
                    DirectoryBaselineCheck(evaluationOutputDir);
                IniModelTestInformation iniModelTestInformation =
                    new IniModelTestInformation(modelFilePath, trainDatasetPath, evaluationOutputDir, cmd, runParameters, processDebugInformation, baselineCheckDebugInformation);
                if (baselineCheckDebugInformation.Key == null)
                {
                    successTestInformation.Add(iniModelTestInformation);
                }
                else
                {
                    failureTestInformation.Add(iniModelTestInformation);
                }
            }
            else
            {
                IniModelTestInformation iniModelTestInformation =
                    new IniModelTestInformation(modelFilePath, trainDatasetPath, evaluationOutputDir, cmd, runParameters, processDebugInformation, new KeyValuePair<Exception, List<string>>(null, null));
                failureTestInformation.Add(iniModelTestInformation);
            }
        }

        /// <summary>
        /// Do a baseline check for and INI test directory
        /// </summary>
        /// <param name="outputDirectory"></param>
        public KeyValuePair<Exception, List<string>> DirectoryBaselineCheck(string outputDirectory)
        {
            List<string> baselineCheckDebugInformation = new List<string>();
            try
            {
                foreach (string file in Directory.EnumerateFiles(outputDirectory))
                {
                    FileInfo fileInfo = new FileInfo(file);
                    string fileName = fileInfo.Name;
                    string firstLevelDirectoryName = fileInfo.Directory.Name;
                    Contracts.Assert(fileInfo.Directory.Parent.Name == "Dirs");
                    string secondLevelDirectoryName = fileInfo.Directory.Parent.Parent.Name;
                    string subDirectory = secondLevelDirectoryName + @"\Dirs\" + firstLevelDirectoryName;
                    baselineCheckDebugInformation.Add(file);
                    baselineCheckDebugInformation.Add(fileName);
                    baselineCheckDebugInformation.Add(firstLevelDirectoryName);
                    baselineCheckDebugInformation.Add(secondLevelDirectoryName);
                    baselineCheckDebugInformation.Add(subDirectory);
                    CheckEqualityNormalized(subDirectory, fileName);
                }
            }
            catch (Exception e)
            {
                return new KeyValuePair<Exception, List<string>>(e, baselineCheckDebugInformation);
            }
            return new KeyValuePair<Exception, List<string>>(null, baselineCheckDebugInformation);
        }
        /// <summary>
        /// Ensure a directory has been recreated and is empty.
        /// </summary>
        /// <param name="directory"></param>
        /// <param name="isRecursive"></param>
        /// <returns>null for a successful operation, an exception object otherwise</returns>
        public static Exception EnsureEmptyDirectory(string directory, bool isRecursive = true)
        {
            int count = 0;
            for (; ; )
            {
                try
                {
                    Directory.Delete(directory, isRecursive);
                }
                catch
                {
                }

                // Directory.Delete doesn't appear to be entirely blocking. If we call CreateDirectory
                // before the Delete is complete, the create call does nothing, and when the delete
                // completes we're left with no directory!
                if (!Directory.Exists(directory))
                    break;

                if (++count >= 100)
                    throw Contracts.Except("Can't delete the directory!");
                Thread.Sleep(100 * count);
            }

            try
            {
                Directory.CreateDirectory(directory);
            }
            catch (Exception e)
            {
                return e;
            }
            return null;
        }
        /// <summary>
        /// Data structure to store debugging information about a failed process execution.
        /// </summary>
        public class ProcessDebugInformation
        {
            public ProcessDebugInformation(
                int exitCode,
                string standardOutput,
                string standardError
                )
            {
                this.ExitCode = exitCode;
                this.StandardOutput = standardOutput;
                this.StandardError = standardError;
            }
            public readonly int ExitCode;
            public readonly string StandardOutput;
            public readonly string StandardError;
        }
        /// <summary>
        /// Run a command-line script.
        /// </summary>
        /// <param name="commandLine"></param>
        public ProcessDebugInformation RunCommandLine(string commandLine, string dir)
        {
            // Create the process.
            var proc = new System.Diagnostics.Process();
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.FileName = "cmd.exe";
            proc.StartInfo.CreateNoWindow = true;
            proc.StartInfo.Arguments = "/C" + " " + commandLine;
            proc.StartInfo.RedirectStandardOutput = true;
            proc.StartInfo.RedirectStandardError = true;
            proc.StartInfo.WorkingDirectory = dir;

            // Run the process.
            proc.Start();
            string standardOutput = proc.StandardOutput.ReadToEnd();
            string standardError = proc.StandardError.ReadToEnd();
            Log("---- execution standard output: " + standardOutput);
            Log("---- execution standard error: " + standardError);

            // Wait for the process to finish.
            proc.WaitForExit();

            ProcessDebugInformation info = new ProcessDebugInformation(proc.ExitCode, standardError, standardOutput);
            proc.Close();
            return info;
        }
    }
#endif
}
