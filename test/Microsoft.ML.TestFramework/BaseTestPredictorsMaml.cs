// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.RunTests
{
    using ResultProcessor = Microsoft.ML.Runtime.Internal.Internallearn.ResultProcessor.ResultProcessor;

    /// <summary>
    /// This is a base test class designed to support running trainings and related
    /// commands, and comparing the results against baselines.
    /// </summary>
    public abstract partial class BaseTestPredictors : TestDmCommandBase
    {
        public enum Cmd
        {
            TrainTest,
            Train,
            Test,
            CV
        }

        /// <summary>
        /// A generic class for a test run.
        /// </summary>
        protected sealed class RunContext : RunContextBase
        {
            public readonly Cmd Command;
            public readonly PredictorAndArgs Predictor;
            public readonly TestDataset Dataset;

            public readonly string[] ExtraArgs;
            public readonly string ExtraTag;

            public readonly bool ExpectedToFail;
            public readonly bool Summary;
            public readonly bool SaveAsIni;

            public readonly OutputPath ModelOverride;

            public override bool NoComparisons { get { return true; } }

            public RunContext(TestCommandBase test, Cmd cmd, PredictorAndArgs predictor, TestDataset dataset,
                string[] extraArgs = null, string extraTag = "",
                bool expectFailure = false, OutputPath modelOverride = null, bool summary = false, bool saveAsIni = false)
                : base(test, predictor.Trainer.Kind, GetNamePrefix(cmd.ToString(), predictor, dataset, extraTag), predictor.BaselineProgress)
            {
                Command = cmd;
                Predictor = predictor;
                Dataset = dataset;

                ExtraArgs = extraArgs;
                ExtraTag = extraTag;

                ExpectedToFail = expectFailure;
                Summary = summary;

                ModelOverride = modelOverride;
                SaveAsIni = saveAsIni;
            }

            public override OutputPath ModelPath()
            {
                return ModelOverride ?? base.ModelPath();
            }

            public RunContextBase TestCtx()
            {
                return new TestImpl(this);
            }

            private sealed class TestImpl : RunContextBase
            {
                public override bool NoComparisons { get { return true; } }

                public TestImpl(RunContextBase ctx) :
                    base(ctx.Test, ctx.BaselineDir, ctx.BaselineNamePrefix + "-test", ctx.BaselineProgress)
                {
                }
            }
        }

        public delegate bool Equal<T1, T2>(ref T1 a, ref T2 b, out int nonEqualIdx);

        /// <summary>
        /// Run the predictor with given args and check if it adds up
        /// </summary>
        protected void Run(RunContext ctx)
        {
            Contracts.Assert(IsActive);
            List<string> args = new List<string>();
            if (ctx.Command != Cmd.Test)
                AddIfNotEmpty(args, ctx.Predictor.Trainer, "tr");
            string dataName = ctx.Command == Cmd.Test ? ctx.Dataset.testFilename : ctx.Dataset.trainFilename;
            AddIfNotEmpty(args, GetDataPath(dataName), "data");
            AddIfNotEmpty(args, 1, "seed");
            //AddIfNotEmpty(args, false, "threads");

            Log("Running '{0}' on '{1}'", ctx.Predictor.Trainer.Kind, ctx.Dataset.name);

            string dir = ctx.BaselineDir;
            if (ctx.Command == Cmd.TrainTest)
                AddIfNotEmpty(args, GetDataPath(ctx.Dataset.testFilename), "test");
            if (ctx.Command == Cmd.TrainTest || ctx.Command == Cmd.Train)
                AddIfNotEmpty(args, GetDataPath(ctx.Dataset.validFilename), "valid");

            // Add in the loader args, and keep a location so we can backtrack and remove it later.
            int loaderArgIndex = -1;
            string loaderArgs = GetLoaderTransformSettings(ctx.Dataset);
            if (!string.IsNullOrWhiteSpace(loaderArgs))
            {
                loaderArgIndex = args.Count;
                args.Add(loaderArgs);
            }
            // Add in the dataset transforms. These need to come before the predictor imposed transforms.
            if (ctx.Dataset.mamlExtraSettings != null)
                args.AddRange(ctx.Dataset.mamlExtraSettings);

            // Model file output, used only for train/traintest.
            var modelPath = ctx.Command == Cmd.Train || ctx.Command == Cmd.TrainTest ? ctx.ModelPath() : null;
            AddIfNotEmpty(args, modelPath, "out");

            string basePrefix = ctx.BaselineNamePrefix;

            // Predictions output, for all types of commands except train.
            OutputPath predOutPath = ctx.Command == Cmd.Train ? null : ctx.InitPath(".txt");
            AddIfNotEmpty(args, predOutPath, "dout");

            if (ctx.Predictor.MamlArgs != null)
                args.AddRange(ctx.Predictor.MamlArgs);

            // If CV, do not run the CV in multiple threads.
            if (ctx.Command == Cmd.CV)
                args.Add("threads-");

            if (ctx.ExtraArgs != null)
            {
                foreach (string arg in ctx.ExtraArgs)
                    args.Add(arg);
            }

            AddIfNotEmpty(args, ctx.Predictor.Scorer, "scorer");
            if (ctx.Command != Cmd.Test)
                AddIfNotEmpty(args, ctx.Predictor.Tester, "eval");
            else
                AddIfNotEmpty(args, ctx.ModelOverride.Path, "in");

            string runcmd = string.Join(" ", args.Where(a => !string.IsNullOrWhiteSpace(a)));
            Log("  Running as: {0} {1}", ctx.Command, runcmd);

            int res;
            if (basePrefix == null)
            {
                // Not capturing into a specific log.
                Log("*** Start raw predictor output");
                res = MainForTest(Env, LogWriter, string.Join(" ", ctx.Command, runcmd), ctx.BaselineProgress);
                Log("*** End raw predictor output, return={0}", res);
                return;
            }
            var consOutPath = ctx.StdoutPath();
            TestCore(ctx, ctx.Command.ToString(), runcmd);
            bool matched = consOutPath.CheckEqualityNormalized();

            if (modelPath != null && (ctx.Summary || ctx.SaveAsIni))
            {
                // Save the predictor summary and compare it to baseline.
                string str = string.Format("SavePredictorAs in={{{0}}}", modelPath.Path);
                List<string> files = new List<string>();
                if (ctx.Summary)
                {
                    var summaryName = basePrefix + "-summary.txt";
                    files.Add(summaryName);
                    var summaryPath = DeleteOutputPath(dir, summaryName);
                    str += string.Format(" sum={{{0}}}", summaryPath);
                    Log("  Saving summary with: {0}", str);
                }

                if (ctx.SaveAsIni)
                {
                    var iniName = basePrefix + ".ini";
                    files.Add(iniName);
                    var iniPath = DeleteOutputPath(dir, iniName);
                    str += string.Format(" ini={{{0}}}", iniPath);
                    Log("  Saving ini file: {0}", str);
                }

                MainForTest(Env, LogWriter, str);
                files.ForEach(file => CheckEqualityNormalized(dir, file));
            }

            if (ctx.Command == Cmd.Train || ctx.Command == Cmd.Test || ctx.ExpectedToFail)
                return;

            // ResultProcessor output
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) // -rp.txt files are not getting generated for Non-Windows Os
            {
                string rpName = basePrefix + "-rp.txt";
                string rpOutPath = DeleteOutputPath(dir, rpName);

                string[] rpArgs = null;
                if (ctx.Command == Cmd.CV && ctx.ExtraArgs != null && ctx.ExtraArgs.Any(arg => arg.Contains("opf+")))
                    rpArgs = new string[] { "opf+" };

                // Run result processor on the console output.
                RunResultProcessorTest(Env, new string[] { consOutPath.Path }, rpOutPath, rpArgs);
                CheckEqualityNormalized(dir, rpName);
            }

            // Check the prediction output against its baseline.
            Contracts.Assert(predOutPath != null);
            predOutPath.CheckEquality();

            if (ctx.Command == Cmd.TrainTest)
            {
                // Adjust the args so that we no longer have the loader and transform
                // arguments in there.
                if (loaderArgIndex >= 0)
                    args.RemoveAt(loaderArgIndex);
                bool foundOut = false;
                List<int> toRemove = new List<int>();
                HashSet<string> removeArgs = new HashSet<string>();
                removeArgs.Add("tr=");
                removeArgs.Add("data=");
                removeArgs.Add("valid=");
                removeArgs.Add("norm=");
                removeArgs.Add("cali=");
                removeArgs.Add("numcali=");
                removeArgs.Add("xf=");
                removeArgs.Add("cache-");
                removeArgs.Add("sf=");

                for (int i = 0; i < args.Count; ++i)
                {
                    if (string.IsNullOrWhiteSpace(args[i]))
                        continue;
                    if (removeArgs.Any(x => args[i].StartsWith(x)))
                        toRemove.Add(i);
                    if (args[i].StartsWith("out="))
                        args[i] = string.Format("in={0}", args[i].Substring(4));
                    if (args[i].StartsWith("test="))
                        args[i] = string.Format("data={0}", args[i].Substring(5));
                    foundOut = true;
                }
                Contracts.Assert(foundOut);
                toRemove.Reverse();
                foreach (int i in toRemove)
                    args.RemoveAt(i);
                runcmd = string.Join(" ", args.Where(a => !string.IsNullOrWhiteSpace(a)));

                // Redirect output to the individual log and run the test.
                var ctx2 = ctx.TestCtx();
                OutputPath consOutPath2 = ctx2.StdoutPath();
                TestCore(ctx2, "Test", runcmd);

                if (CheckTestOutputMatchesTrainTest(consOutPath.Path, consOutPath2.Path, 1))
                    File.Delete(consOutPath2.Path);
                else if (matched)
                {
                    // The TrainTest output matched the baseline, but the SaveLoadTest output did not, so
                    // append some stuff to the .txt output so comparing output to baselines in BeyondCompare
                    // will show the issue.
                    using (var writer = OpenWriter(consOutPath.Path, true))
                    {
                        writer.WriteLine("*** Unit Test Failure! ***");
                        writer.WriteLine("Loaded predictor test results differ! Compare baseline with {0}", consOutPath2.Path);
                        writer.WriteLine("*** Unit Test Failure! ***");
                    }
                }
                // REVIEW: There is nothing analogous to the old predictor output comparison here.
                // The MAML command does not "export" the result of its training programmatically, that would
                // allow us to compare it to the loaded model. To verify that the result of the trained model
                // is the same as its programmatic 
            }
        }

        protected void RunResultProcessorTest(IHostEnvironment env, string[] dataFiles, string outPath, string[] extraArgs)
        {
            Contracts.Assert(IsActive);

            File.Delete(outPath);

            List<string> args = new List<string>();
            for (int i = 0; i < dataFiles.Length; i++)
            {
                args.Add("\"" + dataFiles[i] + "\"");
            }
            args.Add("/o");
            args.Add(outPath);
            args.Add("/calledFromUnitTestSuite+");

            if (extraArgs != null)
                args.AddRange(extraArgs);
            ResultProcessor.Main(env, args.ToArray());
        }

        private static string GetNamePrefix(string testType, PredictorAndArgs predictor, TestDataset dataset, string extraTag = "")
        {
            // REVIEW: Once we finish the TL->MAML conversion effort, please make the output/baseline
            // names take some form that someone could actually tell what test generated that file.

            string datasetSuffix = dataset.name;
            if (!string.IsNullOrEmpty(extraTag))
            {
                if (char.IsLetterOrDigit(extraTag[0]))
                    datasetSuffix += "." + extraTag;
                else
                    datasetSuffix += extraTag;
            }
            string filePrefix = (string.IsNullOrEmpty(predictor.Tag) ? predictor.Trainer.Kind : predictor.Tag);
            return filePrefix + "-" + testType + "-" + datasetSuffix;
        }

        /// <summary>
        /// Create a string for specifying the loader and transform.
        /// </summary>
        public string GetLoaderTransformSettings(TestDataset dataset)
        {
            List<string> settings = new List<string>();

            Contracts.Check(dataset.testSettings == null, "Separate test loader pipeline is not supported");

            if (!string.IsNullOrEmpty(dataset.loaderSettings))
                settings.Add(dataset.loaderSettings);
            if (!string.IsNullOrEmpty(dataset.labelFilename))
                settings.Add(string.Format("xf=lookup{{col=Label data={{{0}}}}}", GetDataPath(dataset.labelFilename)));

            return settings.Count > 0 ? string.Join(" ", settings) : null;
        }

        /// <summary>
        /// Run TrainTest and CV for a set of predictors on a set of datasets.
        /// </summary>
        protected void RunAllTests(
            IList<PredictorAndArgs> predictors, IList<TestDataset> datasets,
            string[] extraSettings = null, string extraTag = "", bool summary = false)
        {
            Contracts.Assert(IsActive);
            foreach (TestDataset dataset in datasets)
            {
                foreach (PredictorAndArgs predictor in predictors)
                    RunOneAllTests(predictor, dataset, extraSettings, extraTag, summary);
            }
        }

        /// <summary>
        /// Run TrainTest, CV, and TrainSaveTest for a single predictor on a single dataset.
        /// </summary>
        protected void RunOneAllTests(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "", bool summary = false)
        {
            Contracts.Assert(IsActive);
            Run_TrainTest(predictor, dataset, extraSettings, extraTag, summary: summary);
            Run_CV(predictor, dataset, extraSettings, extraTag, useTest: true);
        }

        /// <summary>
        /// Run Train for a single predictor on a single dataset.
        /// </summary>
        protected RunContext RunOneTrain(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "")
        {
            Contracts.Assert(IsActive);
            return Run_Train(predictor, dataset, extraSettings, extraTag);
        }

        /// <summary>
        /// Run a train unit test
        /// </summary>
        protected RunContext Run_Train(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "")
        {
            RunContext ctx = new RunContext(this, Cmd.Train, predictor, dataset, extraSettings, extraTag);
            Run(ctx);
            return ctx;
        }

        /// <summary>
        /// Run a train-test unit test
        /// </summary>
        protected void Run_TrainTest(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "", bool expectFailure = false, bool summary = false, bool saveAsIni = false)
        {
            RunContext ctx = new RunContext(this, Cmd.TrainTest, predictor, dataset, extraSettings, extraTag, expectFailure: expectFailure, summary: summary, saveAsIni: saveAsIni);
            Run(ctx);
        }

        // REVIEW: Remove TrainSaveTest and supporting code.

        /// <summary>
        /// Run a unit test which does training, saves the model, and then tests
        /// after loading the model
        /// </summary>
        protected void Run_TrainSaveTest(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "")
        {
            // Train and save the model.
            RunContext trainCtx = new RunContext(this, Cmd.Train, predictor, dataset, extraSettings, extraTag);
            Run(trainCtx);
            // Load the model and test.
            RunContext testCtx = new RunContext(this, Cmd.Test, predictor, dataset, extraSettings, extraTag,
                modelOverride: trainCtx.ModelPath());
            Run(testCtx);
        }

        protected void Run_Test(PredictorAndArgs predictor, TestDataset dataset, string modelPath,
            string[] extraSettings = null, string extraTag = "")
        {
            OutputPath path = new OutputPath(modelPath);
            RunContext testCtx = new RunContext(this, Cmd.Test, predictor, dataset,
                extraSettings, extraTag, modelOverride: path);
            Run(testCtx);
        }

        /// <summary>
        /// Run a cross-validation unit test, over the training set, unless
        /// <paramref name="useTest"/> is set.
        /// </summary>
        protected void Run_CV(PredictorAndArgs predictor, TestDataset dataset,
            string[] extraSettings = null, string extraTag = "", bool useTest = false)
        {
            if (useTest)
            {
                // REVIEW: It is very strange to use the *test* set in
                // cross validation. Should this just be deprecated outright?
                dataset = dataset.Clone();
                dataset.trainFilename = dataset.testFilename;
            }
            RunContext cvCtx = new RunContext(this, Cmd.CV, predictor, dataset, extraSettings, extraTag);
            Run(cvCtx);
        }

        /// <summary>
        /// Add a /arg value pair if value is not null/empty
        /// </summary>
        private static void AddIfNotEmpty(List<string> list, OutputPath val, string name)
        {
            if (val != null && !string.IsNullOrWhiteSpace(val.Path))
                list.Add(val.ArgStr(name));
        }

        /// <summary>
        /// Add a /arg value pair if value is not null/empty
        /// </summary>
        private static void AddIfNotEmpty(List<string> list, object val, string name)
        {
            string sval = val as string;
            if (!string.IsNullOrWhiteSpace(sval) || ((sval == null) != (val == null)))
                list.Add(string.Format("{0}={1}", name, val));
        }

        /// <summary>
        /// Combine all sets of options
        /// </summary>
        public static string[] JoinOptions(params string[][] options)
        {
            List<string> optionsList = new List<string>();
            foreach (string[] o in options)
                if (o != null)
                    optionsList.AddRange(o);
            return optionsList.ToArray();
        }
    }
}
