// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Tools;

#if TLCFULLBUILD
using Microsoft.ML.Runtime.ExperimentVisualization;
#endif

namespace Microsoft.ML.Runtime.Internal.Internallearn.ResultProcessor
{
    using Float = System.Single;
    /// <summary>
    /// The processed Results of a particular Learner
    /// </summary>
    [Serializable]
    public class PredictorResult
    {
        /// <summary>
        /// list of ExperimentItemResult object belonging to a particular Learner
        /// </summary>
        public List<ExperimentItemResult> PredictorList;

        /// <summary>
        /// Name of the Learner for which the rest of the properties are defined in this object
        /// </summary>
        public string LearnerName;

        /// <summary>
        /// Names of all the Settings which have been modified in the list of ExperimentItemResult
        /// </summary>
        public HashSet<string> SettingHeaderNames;

        /// <summary>
        /// Names of all the Result metrices which have been modified in the list of ExperimentItemResult
        /// </summary>
        public HashSet<string> ResultHeaderNames;

        /// <summary>
        /// The default value of all the settigs specified in the SettingHeaderNames field
        /// </summary>
        public Dictionary<string, string> DefaultSettings;

        /// <summary>
        /// List of all the Field names and values which are the same throughout the DataGrid
        /// </summary>
        public Dictionary<string, object> SameHeaderValues;

        public static Dictionary<string, Dictionary<string, string>> MapDefaultSettingToLearner = new Dictionary<string, Dictionary<string, string>>();

        /// <summary>
        /// Checks which all fields of the Predictor Result object would be having the same values
        /// </summary>
        public void CheckForSameValues()
        {
            if (SameHeaderValues == null)
                SameHeaderValues = new Dictionary<string, object>();
            else
                SameHeaderValues.Clear();

            string testFile = PredictorList[0].TestDatafile ?? "";
            string trainFile = PredictorList[0].Datafile ?? "";

            SameHeaderValues.Add(ResultProcessor.LearnerName, LearnerName);
            SameHeaderValues.Add(ResultProcessor.TestDataset, testFile);
            SameHeaderValues.Add(ResultProcessor.TrainDataset, trainFile);

            foreach (ExperimentItemResult res in PredictorList)
            {
                if (testFile != (res.TestDatafile ?? ""))
                    SameHeaderValues.Remove(ResultProcessor.TestDataset);
                if (trainFile != (res.Datafile ?? ""))
                    SameHeaderValues.Remove(ResultProcessor.TrainDataset);
            }
        }

        /// <summary>
        /// Add the new Setting name in the ExperimentItemResult object to SettingHeaderNames field
        /// </summary>
        /// <param name="result">New ExperimentItemResult Object computed</param>
        public bool AllignSettingHeaderNames(ExperimentItemResult result)
        {
            if (SettingHeaderNames == null)
                SettingHeaderNames = new HashSet<string>();
            int initial = SettingHeaderNames.Count;
            foreach (KeyValuePair<string, string> setting in result.Settings)
                SettingHeaderNames.Add(setting.Key.StartsWith("/") ? setting.Key : "/" + setting.Key);
            if (SettingHeaderNames.Count != initial)
                return false;

            return true;
        }

        /// <summary>
        /// Add the new Result name in the ExperimentItemResult object to ResultHeaderNames field
        /// </summary>
        /// <param name="result">New ExperimentItemResult Object computed</param>
        public bool AllignResultHeaderNames(ExperimentItemResult result)
        {
            if (ResultHeaderNames == null)
                ResultHeaderNames = new HashSet<string>();
            int initial = ResultHeaderNames.Count;

            foreach (KeyValuePair<string, ResultMetric> resultEntity in result.Results)
                ResultHeaderNames.Add(resultEntity.Key);

            if (ResultHeaderNames.Count != initial)
                return false;

            return true;
        }

        /// <summary>
        /// Get all the default settings for a particular learner(public method which calls the private method and sets the Defaultsetting field)
        /// </summary>
        /// <param name="env"></param>
        /// <param name="predictorName">Learner name</param>
        /// <param name="extraAssemblies"></param>
        public void GetDefaultSettingValues(IHostEnvironment env, string predictorName, string[] extraAssemblies = null)
        {
            lock (MapDefaultSettingToLearner)
            {
                Dictionary<string, string> temp;
                if (!MapDefaultSettingToLearner.TryGetValue(predictorName, out temp))
                {
                    temp = GetDefaultSettings(env, predictorName, extraAssemblies);
                    MapDefaultSettingToLearner.Add(predictorName, temp);
                }
                DefaultSettings = temp;
            }
        }

        /// <summary>
        /// Get all the default settings for a particular learner
        /// </summary>
        /// <param name="env"></param>
        /// <param name="predictorName">Learner name</param>
        /// <param name="extraAssemblies"></param>
        private Dictionary<string, string> GetDefaultSettings(IHostEnvironment env, string predictorName, string[] extraAssemblies = null)
        {
            ComponentCatalog.CacheClassesExtra(extraAssemblies);

            var cls = ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(predictorName);
            if (cls == null)
            {
                Console.Error.WriteLine("Can't load trainer '{0}'", predictorName);
                return new Dictionary<string, string>();
            }

            var defs = cls.CreateArguments();
            if (defs == null)
            {
                // No arguments for this trainer.
                return new Dictionary<string, string>(0);
            }
            return
                CmdParser.GetSettingPairs(env, defs, SettingsFlags.ShortNames)
                .GroupBy(kvp => kvp.Key, kvp => kvp.Value)
                .ToDictionary(g => "/" + g.Key, g => string.Join(",", g));
        }

        /// <summary>
        /// Initialize all the public fields of the predictorResult object
        /// </summary>
        /// <param name="result">ExperimentItemResult object</param>
        public void Initialize(ExperimentItemResult result)
        {
            LearnerName = result.Trainer.Kind;
            AllignResultHeaderNames(result);
            AllignSettingHeaderNames(result);
        }
    }

    /// <summary>
    /// All the members which define a particular result metric
    /// </summary>
    [Serializable]
    public class ResultMetric
    {
        public Float MetricValue { get; set; }
        public Float Deviation { get; set; }
        public Float[] AllValues { get; set; }

        /// <summary>
        /// Constructor initializing the object.
        /// </summary>
        /// <param name="metricValue">metric value</param>
        /// <param name="deviation">Deviation, 0.0 if not passed</param>
        public ResultMetric(Float metricValue, Float deviation = 0)
        {
            MetricValue = metricValue;
            Deviation = deviation;
        }
    }

    /// <summary>
    /// A structure summarizing experiment results
    /// </summary>
    [Serializable]
    public struct ExperimentItemResult
    {
        /// <summary>
        /// used in TLC GUI for mapping result to experimental run
        /// </summary>
        public int Key;

        /// <summary>
        /// the path to data file from the command.
        /// </summary>
        public string Datafile;

        /// <summary>
        /// the path to the test data file from the command.
        /// </summary>
        public string TestDatafile;

        /// <summary>
        /// the trainer SubComponent from the command.
        /// </summary>
        public SubComponent<ITrainer, SignatureTrainer> Trainer;

        /// <summary>
        /// The name of the output file produced by the Experiment Run
        /// </summary>
        public string InputFile;

        /// <summary>
        /// the settings for the Experiment Run
        /// </summary>
        public Dictionary<string, string> Settings;

        /// <summary>
        /// List of Result metrices for the particular Experiment Run
        /// </summary>
        public Dictionary<string, ResultMetric> Results;

        /// <summary>
        /// Metrics for individual folds -- each metric will contain data in AllValues field.
        /// </summary>
        public Dictionary<string, ResultMetric> PerFoldResults;

        /// <summary>
        /// commandLine string
        /// </summary>
        public string Commandline;

        /// <summary>
        /// Time taken for the Experiment run to complete
        /// </summary>
        public double Time;

        /// <summary>
        /// Physical memory usage in MB for the Experiment run to complete
        /// </summary>
        public long PhysicalMemory;

        /// <summary>
        /// Virtual memory usage in MB for the Experiment run to complete
        /// </summary>
        public long VirtualMemory;

        /// <summary>
        /// Date and time that the Experiement run completed
        /// </summary>
        public string ExecutionDate;

        /// <summary>
        /// A user defined tag used in visualization report.
        /// </summary>
        public string CustomizedTag;

        /// <summary>
        /// gets the list of settings in a List&lt;string&gt; form
        /// </summary>
        /// <returns>List of settings which are not default</returns>
        public List<string> GetSettings()
        {
            List<string> settings = new List<string>();
            foreach (KeyValuePair<string, string> entity in Settings)
            {
                settings.Add(entity.Key + ":" + entity.Value);
            }
            return settings;
        }
    }

    /// <summary>
    /// Command-line arguments
    /// </summary>
    public class ResultProcessorArguments
    {
        // input data
        [DefaultArgument(ArgumentType.Multiple, HelpText = "Result file pattern")]
        public string[] ResultFiles;

        // output data
        [Argument(ArgumentType.AtMostOnce, HelpText = "Output file name", ShortName = "o")]
        public string OutputFile;

        // output to a visualization HTML too?
        [Argument(ArgumentType.AtMostOnce, HelpText = "Output to a visualization HTML", ShortName = "html")]
        public string VisualizationHtml;

        // include all metrics?
        [Argument(ArgumentType.Multiple, HelpText = "Which metrics should be processed (default=all)?", ShortName = "a")]
        public string[] Metrics;

        // include standard deviations?
        [Argument(ArgumentType.AtMostOnce, HelpText = "Include columns for standard deviations?", ShortName = "stdev")]
        public bool IncludeStandardDeviations = false;

        // print metrics for individual folds/bootstrap rounds?
        [Argument(ArgumentType.AtMostOnce, HelpText = "Output per-fold results", ShortName = "opf")]
        public bool IncludePerFoldResults = false;

        // separator for per-fold results
        [Argument(ArgumentType.AtMostOnce, HelpText = "Separator for per-fold results. Can be: actual char, 'tab', 'colon', 'space','comma'", ShortName = "opfsep")]
        public string PerFoldResultSeparator = ",";

        // extra DLLs for dynamic loading
        [Argument(ArgumentType.Multiple, HelpText = "Extra DLLs", ShortName = "dll")]
        public string[] ExtraAssemblies = null;
        [Argument(ArgumentType.AtMostOnce, HelpText = "Internal setting set if called from unit test suite")]
        public bool CalledFromUnitTestSuite = false;

        [Argument(ArgumentType.Multiple, HelpText = "Result file pattern with customized tag", ShortName = "in")]
        public KeyValuePair<string, string>[] ResultFilesWithTags;
    }

    /// <summary>
    /// Given a pattern of output files, parse them!
    /// </summary>
    public class ResultProcessor
    {
        public const string SectionBreakSeparator = "---------------------------------------";
        public const string TestDataset = "Test Dataset";
        public const string TrainDataset = "Train Dataset";
        public const string LearnerName = "Learner Name";
        public const string Runtime = "Run Time";
        public const string PhysicalMemoryUsage = "Physical Memory";
        public const string VirturalMemoryUsage = "Virtual Memory";
        public const string ResultsFile = "Results File";
        public const string CommandLineArgument = "Command Line";
        public const string Settings = "Settings";

        private const string FoldSeparatorString =
            "----------------------------------------------------------------------------------------";

        private readonly static Regex _rxNameValue = new Regex(@"(?<name>.+)\s*:\s*(?<value>\S+)", RegexOptions.Compiled);
        private readonly static Regex _rxNameValueDeviation = new Regex(@"(?<name>.+)\s*:\s*(?<value>\S+)\s*\((?<deviation>\S+)\)", RegexOptions.Compiled);
        private readonly static Regex _rxTimeElapsed = new Regex(@"(?<executionDate>.*)\t Time elapsed\(s\): (?<timeElapsed>[\d\.]*)", RegexOptions.Compiled);
        private readonly static Regex _rxMemoryUsage = new Regex(@"(?<memoryType>[\w]+) memory usage\(MB\): (?<memoryUsage>[\d]*)", RegexOptions.Compiled);

        public static bool CheckEndOfFileReached(string[] lines)
        {
            int i = 0;
            while (i < lines.Length && !lines[i].Contains("Time elapsed(s):"))
                i++;
            if (i == lines.Length)
                return false;
            else
                return true;
        }

        private static bool ValidateMamlOutput(string filename, string[] rawLines, out List<string> lines)
        {
            if (!TryParseFileToLines(filename, out lines, rawLines))
            {
                Console.Error.WriteLine("ResultProcessor was asked to process results from file {0} which does not exist.", filename);
                return false;
            }

            //return if file is empty
            if (Utils.Size(lines) == 0)
            {
                Console.Error.WriteLine("Empty file {0}", filename);
                return false;
            }
            return true;
        }

        // Temporary hack until we had the new and shiny ResultProcessor
        public static ExperimentItemResult? ProcessMamlOutputLines(IHostEnvironment env, string filename, ResultProcessorArguments cmd = null,
            string[] rawLines = null, string commandline = null)
        {
            List<string> lines;
            if (!ValidateMamlOutput(filename, rawLines, out lines))
            {
                return null;
            }

            bool trimExe = false;
            if (commandline == null)
            {
                trimExe = true;
                // REVIEW : We need update this code to corretly extract commandline in case
                // it has newline character in it, for example CSharp transform.
                if (!TryParseMamlCommand(lines, out commandline))
                {
                    Console.Error.WriteLine("Results file {0} does not contain the MAML executable command:", filename);
                    Console.Error.WriteLine("\t{0}", commandline);
                    return null;
                }
            }

            // REVIEW: This whole mechanism is buggy and convoluted. Fix it!
            List<string> fileTimeMemoryResults;
            List<string> fileResults;
            ComponentCatalog.LoadableClassInfo command;
            object commandArgs;
            if (!TryParseLines(lines, cmd, out fileResults, out fileTimeMemoryResults)
                || !ParseCommandArguments(env, commandline, out commandArgs, out command, trimExe))
            {
                return null;
            }

            var chainArgs = commandArgs as ChainCommand.Arguments;
            if (chainArgs != null)
            {
                if (Utils.Size(chainArgs.Command) == 0)
                    return null;
                var acceptableCommand = chainArgs.Command.FirstOrDefault(x =>
                    string.Equals(x.Kind, "CV", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(x.Kind, "TrainTest", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(x.Kind, "Test", StringComparison.OrdinalIgnoreCase));
                if (acceptableCommand == null || !ParseCommandArguments(env,
                    acceptableCommand.Kind + " " + acceptableCommand.SubComponentSettings, out commandArgs, out command, trimExe))
                {
                    return null;
                }
            }

            object trainerArgs;
            ComponentCatalog.LoadableClassInfo trainerClass;
            string datafile = string.Empty;
            string testDatafile = string.Empty;
            SubComponent<ITrainer, SignatureTrainer> trainer;
            var trainTestArgs = commandArgs as TrainTestCommand.Arguments;
            if (trainTestArgs != null)
            {
                trainer = trainTestArgs.Trainer;
                datafile = trainTestArgs.DataFile;
                testDatafile = trainTestArgs.TestFile;
            }
            else
            {
                var testArgs = commandArgs as TestCommand.Arguments;
                if (testArgs != null)
                {
                    Contracts.AssertNonEmpty(testArgs.InputModelFile);
                    string savedTrainCmd;
                    using (Stream strm = new FileStream(testArgs.InputModelFile, FileMode.Open, FileAccess.Read))
                    using (var rep = RepositoryReader.Open(strm))
                    {
                        var ent = rep.OpenEntryOrNull(ModelFileUtils.DirTrainingInfo, "Command.txt");
                        if (ent == null)
                            return null;

                        using (ent)
                        using (StreamReader sr = new StreamReader(ent.Stream))
                            savedTrainCmd = sr.ReadToEnd();
                    }

                    // Parse train command
                    if (!ParseCommandArguments(env, savedTrainCmd, out trainerArgs, out trainerClass))
                        return null;

                    testDatafile = testArgs.DataFile;
                    var specificTrainArgs = trainerArgs as TrainCommand.Arguments;
                    if (specificTrainArgs != null)
                    {
                        trainer = specificTrainArgs.Trainer;
                        datafile = specificTrainArgs.DataFile;
                    }
                    else
                    {
                        var specificTrainTestArgs = trainerArgs as TrainTestCommand.Arguments;
                        if (specificTrainTestArgs != null)
                        {
                            datafile = specificTrainTestArgs.DataFile;
                            trainer = specificTrainTestArgs.Trainer;
                        }
                        else
                        {
                            var specificCVArgs = trainerArgs as CrossValidationCommand.Arguments;
                            Contracts.AssertValue(specificCVArgs);
                            datafile = specificCVArgs.DataFile;
                            trainer = specificCVArgs.Trainer;
                        }
                    }
                }
                else
                {
                    var cvArgs = commandArgs as CrossValidationCommand.Arguments;
                    // Only train-test, test and cross-validation mode have results in their output.
                    // The other modes should have returned null after TryParseLines() above.
                    // REVIEW: There is one exception: When running Ensembles in Train mode, it still does evaluation
                    // of individual models, so until evaluators are fixed to not print anything to the console,
                    // we need to explicitly take care of this case.
                    if (cvArgs == null)
                        return null;
                    datafile = cvArgs.DataFile;
                    trainer = cvArgs.Trainer;
                }
            }
            Contracts.AssertValue(trainer);
            trainerClass = ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(trainer.Kind);
            trainerArgs = trainerClass.CreateArguments();
            Dictionary<string, string> predictorSettings;
            if (trainerArgs == null)
            {
                // The trainer had no arguments.
                predictorSettings = new Dictionary<string, string>(0);
            }
            else
            {
                CmdParser.ParseArguments(env, PredictionUtil.CombineSettings(trainer.Settings), trainerArgs);
                predictorSettings = CmdParser.GetSettingPairs(env, trainerArgs, trainerClass.CreateArguments(), SettingsFlags.ShortNames).
                    GroupBy(kvp => kvp.Key, kvp => kvp.Value).ToDictionary(g => "/" + g.Key, g => string.Join(",", g));
            }
            var result = GetMetrics(filename, cmd, lines, fileTimeMemoryResults, fileResults);
            if (result.HasValue)
            {
                return new ExperimentItemResult()
                {
                    InputFile = result.Value.InputFile,
                    Results = result.Value.Results,
                    PerFoldResults = result.Value.PerFoldResults,
                    Time = result.Value.Time,
                    VirtualMemory = result.Value.VirtualMemory,
                    PhysicalMemory = result.Value.PhysicalMemory,
                    ExecutionDate = result.Value.ExecutionDate,
                    Commandline = commandline,
                    Datafile = datafile,
                    TestDatafile = testDatafile,
                    Trainer = trainer,
                    Settings = predictorSettings,
                };

            }
            else
                return null;
        }

        public static ExperimentItemResult? ProcessMetricOutputLines(string filename, ResultProcessorArguments cmd = null,
           string[] rawLines = null)
        {
            List<string> lines;
            if (!ValidateMamlOutput(filename, rawLines, out lines))
            {
                return null;
            }

            // REVIEW: This whole mechanism is buggy and convoluted. Fix it!
            List<string> fileTimeMemoryResults;
            List<string> fileResults;
            if (!TryParseLines(lines, cmd, out fileResults, out fileTimeMemoryResults))
            {
                return null;
            }
            // parse result lines
            Double metricValue;
            var runResults = ParseResultLines(fileResults, cmd, out metricValue, null);
            // if per-fold results requested, create them
            var foldResults = (cmd != null && cmd.IncludePerFoldResults ? GetPerFoldResults(lines) : null);

            return new ExperimentItemResult
            {
                InputFile = filename,
                Results = runResults,
                PerFoldResults = foldResults,
                Time = 0,
                ExecutionDate = DateTimeOffset.Now.UtcDateTime.ToString()
            };
        }

        private static ExperimentItemResult? GetMetrics(string filename, ResultProcessorArguments cmd, List<string> lines, List<string> fileTimeMemoryResults, List<string> fileResults)
        {
            // parse result lines
            Double metricValue;
            var runResults = ParseResultLines(fileResults, cmd, out metricValue, null);

            string timeElapsed = string.Empty;
            string executionDate = string.Empty;
            string physicalMemory = string.Empty;
            string virtualMemory = string.Empty;
            bool matchedTimeElapsed = false;

            foreach (string line in fileTimeMemoryResults)
            {
                Match mc = _rxTimeElapsed.Match(line);
                if (mc.Success)
                {
                    timeElapsed = mc.Groups["timeElapsed"].Value;
                    executionDate = mc.Groups["executionDate"].Value;
                    matchedTimeElapsed = true;
                    continue;
                }

                mc = _rxMemoryUsage.Match(line);
                if (mc.Success)
                {
                    if (mc.Groups["memoryType"].Value == "Virtual")
                        virtualMemory = mc.Groups["memoryUsage"].Value;
                    else if (mc.Groups["memoryType"].Value == "Physical")
                        physicalMemory = mc.Groups["memoryUsage"].Value;
                }
            }

            if (!matchedTimeElapsed)
            {
                Console.Error.WriteLine("Invalid file format.");
                return null;
            }

            //Fix the runtime and memory fields in case we do UnitTesting
            if (cmd != null && cmd.CalledFromUnitTestSuite)
            {
                timeElapsed = "99";
                virtualMemory = "0";
                physicalMemory = "0";
            }

            // if per-fold results requested, create them
            var foldResults = (cmd != null && cmd.IncludePerFoldResults ? GetPerFoldResults(lines) : null);

            double time;
            bool timeParsed = double.TryParse(timeElapsed, out time);
            long physical;
            bool physicalParsed = long.TryParse(physicalMemory, out physical);
            long virtualMem;
            bool virtualParsed = long.TryParse(virtualMemory, out virtualMem);

            return new ExperimentItemResult
            {
                InputFile = filename,
                Results = runResults,
                PerFoldResults = foldResults,
                Time = timeParsed ? time : 0,
                PhysicalMemory = physicalParsed ? physical : 0,
                VirtualMemory = virtualParsed ? virtualMem : 0,
                ExecutionDate = executionDate
            };
        }

        public static bool ParseCommandArguments(IHostEnvironment env, string commandline, out object commandArgs, out ComponentCatalog.LoadableClassInfo commandClass, bool trimExe = true)
        {
            string args = commandline;
            if (trimExe)
            {
                string exec;
                args = CmdParser.TrimExePath(commandline, out exec);
            }

            string kind;
            string settings;
            if (!CmdParser.TryGetFirstToken(args, out kind, out settings))
            {
                commandClass = null;
                commandArgs = null;
                return false;
            }

            commandClass = ComponentCatalog.GetLoadableClassInfo<SignatureCommand>(kind);
            commandArgs = commandClass.CreateArguments();
            CmdParser.ParseArguments(env, settings, commandArgs);
            return true;
        }

        public static void ProcessResultLines(string filename, string metricName, out Double metricValue)
        {
            metricValue = 0;

            // get lines
            List<string> lines;
            if (!TryParseFileToLines(filename, out lines, null) || Utils.Size(lines) == 0)
                return;

            List<string> fileTimeMemoryResults;
            List<string> fileResults;
            if (!TryParseLines(lines, null, out fileResults, out fileTimeMemoryResults))
                return;

            ParseResultLines(fileResults, null, out metricValue, metricName);
        }

        private static bool TryParseFileToLines(string filename, out List<string> lines, string[] rawLines = null)
        {
            lines = new List<string>();

            // if lines weren't passed in, read them
            if (rawLines == null)
            {
                if (!File.Exists(filename))
                    return false;

                rawLines = File.ReadAllLines(filename);
            }

            for (int j = 0; j < rawLines.Length; j++)
            {
                // Remove empty lines
                if (!string.IsNullOrWhiteSpace(rawLines[j]))
                    lines.Add(rawLines[j]);
            }

            return true;
        }

        private static bool TryParseMamlCommand(List<string> lines, out string commandline)
        {
            Contracts.AssertValue(lines);
            for (int i = 0; i < lines.Count; i++)
            {
                if (lines[i] != null && lines[i].ToLower().StartsWith("maml.exe"))
                {
                    commandline = lines[i];
                    return true;
                }
            }

            commandline = null;
            return false;
        }

        private static bool TryParseLines(List<string> lines, ResultProcessorArguments cmd, out List<string> fileResults, out List<string> fileTimeMemoryResult)
        {
            fileResults = new List<string>();
            fileTimeMemoryResult = new List<string>();

            // skip until the results section
            int i = lines.Count - 1;
            while (i > 0 && lines[i] != "OVERALL RESULTS")
                i--;
            if (i == 0)
                return false;
            i = i + 2; // skip separator
            // record all results
            while (i < lines.Count && lines[i] != SectionBreakSeparator)
            {
                if (cmd == null || cmd.Metrics == null || cmd.Metrics.Length == 0)
                {
                    fileResults.Add(lines[i++]);
                    continue;
                }
                foreach (string metric in cmd.Metrics)
                {
                    if (lines[i].Contains(metric))
                    {
                        fileResults.Add(lines[i++]);
                        break;
                    }
                }
            }
            // Find the time elapsed line and memory usage lines.
            for (; i < lines.Count; i++)
            {
                if (lines[i].Contains("Time elapsed(s):") ||
                    lines[i].Contains("memory usage(MB):"))
                    fileTimeMemoryResult.Add(lines[i]);
            }
            return true;
        }

        private static Dictionary<string, ResultMetric> ParseResultLines(List<string> fileResults, ResultProcessorArguments cmd, out Double metricValue, string metricName)
        {
            metricValue = 0;

            Dictionary<string, ResultMetric> runResults = new Dictionary<string, ResultMetric>();
            foreach (string resLine in fileResults)
            {
                Match matchNameValueDeviation = _rxNameValueDeviation.Match(resLine);
                if (matchNameValueDeviation.Success)
                {
                    string name = matchNameValueDeviation.Groups["name"].Value;
                    Double doubleValue = Double.Parse(matchNameValueDeviation.Groups["value"].Value, CultureInfo.InvariantCulture);
                    Float value = (Float)doubleValue;
                    Float deviation = (Float)Double.Parse(matchNameValueDeviation.Groups["deviation"].Value, CultureInfo.InvariantCulture);

                    if (name == metricName)
                        metricValue = value;

                    runResults[name] = new ResultMetric(value, deviation);

                    if (cmd != null && cmd.IncludeStandardDeviations)
                    {
                        runResults[name + "_STDEV"] = new ResultMetric(deviation);
                    }
                    continue;
                }

                Match matchNameValue = _rxNameValue.Match(resLine);
                if (matchNameValue.Success)
                {
                    string name = matchNameValue.Groups["name"].Value;
                    Float value = Float.Parse(matchNameValue.Groups["value"].Value, CultureInfo.InvariantCulture);

                    runResults[name] = new ResultMetric(value);
                    continue;
                }

                // If failed to match these two patterns, skip this "resLine".
            }
            return runResults;
        }

        /// <summary>
        /// Takes an output file as input and processes it to return a ExperimentItemResult object
        /// </summary>
        /// <returns>ExperimentItemResult object obtained after processing the output file</returns>
        public static ExperimentItemResult? ProcessOutputFile(IHostEnvironment env, string filename, ResultProcessorArguments cmd)
        {
            var lines = File.ReadAllLines(filename);
            return ProcessMamlOutputLines(env, filename, cmd, lines);
        }

        /// <summary>
        /// Takes an output file as input and processes it to return a ExperimentItemResult object
        /// </summary>
        /// <returns>ExperimentItemResult object obtained after processing the output file</returns>
        public static IEnumerable<ExperimentItemResult?> ProcessOutputFiles(IHostEnvironment env, IEnumerable<string> filePatterns,
                                                              ResultProcessorArguments cmd = null)
        {
            List<ExperimentItemResult?> results = new List<ExperimentItemResult?>();
            foreach (string filePattern in filePatterns)
                foreach (string filename in StreamUtils.ExpandWildCards(filePattern))
                    results.Add(ProcessOutputFile(env, filename, cmd));

            return results;
        }

        /// <summary>
        /// Returns the PredictorResult object from the list corresponding to the Learname specified in the argument
        /// </summary>
        /// <param name="tempPredictorList">the list of PredictorResult objects</param>
        /// <param name="learnerName">The learner name whose PredictorResult object is required</param>
        /// <param name="newLearner"></param>
        /// <returns></returns>
        public static PredictorResult GetPredictorObject(List<PredictorResult> tempPredictorList, string learnerName, out bool newLearner)
        {
            if (tempPredictorList == null)
                tempPredictorList = new List<PredictorResult>();

            //if predictor found for that learner return else create a new one
            if (tempPredictorList.Count != 0)
            {
                foreach (PredictorResult predictor in tempPredictorList)
                {
                    if (predictor.LearnerName.Equals(learnerName))
                    {
                        newLearner = false;
                        return predictor;
                    }
                }
            }
            newLearner = true;
            PredictorResult predictorItem = new PredictorResult();
            tempPredictorList.Add(predictorItem);
            return predictorItem;
        }

        /// <summary>
        /// Extract per-fold results
        /// </summary>
        protected static Dictionary<string, ResultMetric> GetPerFoldResults(IList<string> lines)
        {
            Dictionary<string, ResultMetric> perFoldMetrics = new Dictionary<string, ResultMetric>();

            Dictionary<int, Dictionary<string, Float>> foldResults = new Dictionary<int, Dictionary<string, Float>>();
            int i = 0;
            while (i < lines.Count)
            {
                if (lines[i] == FoldSeparatorString && (i + 1) < lines.Count && lines[i + 1].StartsWith("FOLD"))
                {
                    int startLineIdx = i + 1;
                    int endLineIdx = i + 2;
                    while (endLineIdx < lines.Count && lines[endLineIdx] != FoldSeparatorString)
                        ++endLineIdx;
                    if (endLineIdx == lines.Count)
                    {
                        Console.Error.WriteLine("ResultProcessor tried to extract fold starting at line {0}, couldn't find end-of-fold separator before end of file.", startLineIdx);
                        break; // reached end of file
                    }
                    // if we're here, endLineIdx is the closing separator.
                    var foldLines = new List<string>(lines.Where((s, i1) => i1 >= startLineIdx && i1 < endLineIdx));
                    var thisFoldResults = AddFoldResults(foldLines);
                    if (thisFoldResults.Key < 0 || thisFoldResults.Value == null)
                        Console.Error.WriteLine("ResultProcessor failed to parse fold starting at line {0} ending at line {1}.", startLineIdx, endLineIdx);
                    else
                    {
                        if (foldResults.ContainsKey(thisFoldResults.Key))
                        {
                            Console.Error.WriteLine("Fold {0} results have already been added, not adding.", thisFoldResults.Key);
                        }
                        foldResults[thisFoldResults.Key] = thisFoldResults.Value;
                    }
                    i = endLineIdx + 1;
                }
                else
                    i++;
            }

            // pivot foldResults to be indexed by metric
            var metricToFoldValuesDict = new Dictionary<string, Dictionary<int, Float>>();
            List<int> allFoldIndices = new List<int>(foldResults.Keys);
            allFoldIndices.Sort();
            foreach (var kvp in foldResults)
            {
                int foldIdx = kvp.Key;
                foreach (var kvp1 in kvp.Value)
                {
                    Dictionary<int, Float> metricDict = null;
                    if (!metricToFoldValuesDict.TryGetValue(kvp1.Key, out metricDict))
                    {
                        metricDict = new Dictionary<int, Float>();
                        metricToFoldValuesDict[kvp1.Key] = metricDict;
                    }
                    metricDict[foldIdx] = kvp1.Value;
                }
            }

            foreach (var metricValues in metricToFoldValuesDict)
            {
                perFoldMetrics[metricValues.Key] = new ResultMetric(Float.NaN)
                {
                    AllValues = new List<Float>(from kvp in metricValues.Value
                                                orderby kvp.Key ascending
                                                select kvp.Value).ToArray()
                };
            }

            return perFoldMetrics;
        }

        /// <summary>
        /// Given output for a single fold, add its results
        /// </summary>
        protected static KeyValuePair<int, Dictionary<string, Float>> AddFoldResults(IList<string> lines)
        {
            int foldIdx = -1;
            string[] foldLineCols = lines[0].Split();
            if (foldLineCols.Length < 2)
            {
                Console.Error.WriteLine("Couldn't parse fold index line: " + lines[0]);
                return new KeyValuePair<int, Dictionary<string, Float>>(-1, null);
            }

            if (!int.TryParse(foldLineCols[foldLineCols.Length - 1], out foldIdx))
            {
                Console.Error.WriteLine("Couldn't parse fold index line: " + lines[0]);
                return new KeyValuePair<int, Dictionary<string, Float>>(-1, null);
            }

            // if run index is in front of fold index, account for it
            for (int j = foldLineCols.Length - 2; j > 0; j--)
            {
                int foldIdxExtra = 0;
                if (int.TryParse(foldLineCols[j], out foldIdxExtra))
                    foldIdx += (int)(foldIdxExtra * Math.Pow(1000, j));
            }

            Dictionary<string, Float> valuesDict = new Dictionary<string, Float>();
            for (int i = 1; i < lines.Count; i++)
            {
                if (lines[i].IndexOf(':') < 0)
                    continue;
                string[] nameValCols = lines[i].Split(':');
                if (nameValCols.Length != 2)
                    continue;
                if (nameValCols[1].EndsWith("%"))
                    nameValCols[1] = nameValCols[1].Substring(0, nameValCols[1].Length - 1);
                Float value = 0;
                if (!Float.TryParse(nameValCols[1], out value))
                    continue;
                valuesDict[nameValCols[0]] = value;
            }
            return new KeyValuePair<int, Dictionary<string, Float>>(foldIdx, valuesDict);
        }

        /// <summary>
        /// Makes a deep clone of the list of PredictorResultList Object
        /// </summary>
        /// <param name="predictorResultList">List of PredictorResult Object</param>
        /// <returns>A new instance of List of PredictorResult</returns>
        public static List<PredictorResult> ClonePredictorResultList(List<PredictorResult> predictorResultList)
        {
            MemoryStream ms = new MemoryStream();
            Save(predictorResultList, ms);              //save the object in Memory stream
            ms.Seek(0, SeekOrigin.Begin);
            return Load(ms) as List<PredictorResult>;   // load the object from memory stream
        }

        /// <summary>
        /// Makes a deep clone of the list of PredictorResult Object
        /// </summary>
        /// <param name="predictorItem"></param>
        /// <returns></returns>
        public static PredictorResult ClonePredictorResult(PredictorResult predictorItem)
        {
            MemoryStream ms = new MemoryStream();
            Save(predictorItem, ms);              //save the object in Memory stream
            ms.Seek(0, SeekOrigin.Begin);
            return Load(ms) as PredictorResult;
        }

        /// <summary>
        /// Save the List of Predictor object in Memory
        /// </summary>
        /// <param name="predictor">List of PredictorResult Object</param>
        /// <param name="stream">Memory stream object</param>
        private static void Save(List<PredictorResult> predictor, Stream stream)
        {
            BinaryFormatter bf = new BinaryFormatter();
            bf.Serialize(stream, predictor);
            if (stream is MemoryStream)
                stream.Flush();
            else
                stream.Close();
        }

        /// <summary>
        /// Save the List of Predictor object in Memory
        /// </summary>
        /// <param name="predictor">List of PredictorResult Object</param>
        /// <param name="stream">Memory stream object</param>
        private static void Save(PredictorResult predictor, Stream stream)
        {
            BinaryFormatter bf = new BinaryFormatter();
            bf.Serialize(stream, predictor);
            if (stream is MemoryStream)
                stream.Flush();
            else
                stream.Close();
        }

#if TLCFULLBUILD
        /// <summary>
        /// Create a experiment visualization object from experiment result.
        /// </summary>
        private static Experiment CreateVisualizationExperiment(ExperimentItemResult result, int index)
        {
            var experiment = new ML.Runtime.ExperimentVisualization.Experiment
            {
                Key = index.ToString(),
                CompareGroup = string.IsNullOrEmpty(result.CustomizedTag) ? result.Trainer.Kind : result.CustomizedTag,
                Trainer = new ML.Runtime.ExperimentVisualization.Trainer
                {
                    Name = result.Trainer.Kind,
                    ParameterSets = new List<ML.Runtime.ExperimentVisualization.Item>()
                },
                DataSet = new ML.Runtime.ExperimentVisualization.DataSet { File = result.Datafile },
                TestDataSet = new ML.Runtime.ExperimentVisualization.DataSet { File = result.TestDatafile },
                Tool = "TLC",
                RawCommandLine = result.Commandline,
                Results = new List<ML.Runtime.ExperimentVisualization.ExperimentResult>()
            };

            // Propagate metrics to the report. 
            ML.Runtime.ExperimentVisualization.ExperimentResult metrics = new ML.Runtime.ExperimentVisualization.ExperimentResult
            {
                Metrics = new List<ML.Runtime.ExperimentVisualization.MetricValue>(),
                Build = "TLC"
            };
            foreach (KeyValuePair<string, ResultMetric> resultEntity in result.Results)
            {
                metrics.Metrics.Add(new ML.Runtime.ExperimentVisualization.MetricValue
                {
                    Name = resultEntity.Key,
                    Value = resultEntity.Value.MetricValue,
                    StandardDeviation = resultEntity.Value.Deviation
                });
            }

            metrics.Metrics.Add(new ML.Runtime.ExperimentVisualization.MetricValue
            {
                Name = "Time Elapsed(s)",
                Value = result.Time
            });

            metrics.Metrics.Add(new ML.Runtime.ExperimentVisualization.MetricValue
            {
                Name = "Physical Memory Usage(MB)",
                Value = result.PhysicalMemory
            });

            metrics.Metrics.Add(new ML.Runtime.ExperimentVisualization.MetricValue
            {
                Name = "Virtual Memory Usage(MB)",
                Value = result.VirtualMemory
            });

            // Propagate experiment arguments to the report. 
            foreach (KeyValuePair<string, string> setting in result.Settings)
            {
                string val;
                if (result.Settings.TryGetValue(setting.Key, out val))
                {
                    experiment.Trainer.ParameterSets.Add(new ML.Runtime.ExperimentVisualization.Item
                    {
                        Name = setting.Key.Substring(1),
                        Value = val
                    });
                    double doubleVal;
                    if (Double.TryParse(val, out doubleVal))
                    {
                        metrics.Metrics.Add(new ML.Runtime.ExperimentVisualization.MetricValue
                        {
                            Name = setting.Key,
                            Value = doubleVal
                        });
                    }
                }
            }

            experiment.Results.Add(metrics);

            return experiment;
        }
#endif

        /// <summary>
        /// Deserialize a predictor, returning as an object
        /// </summary>		
        private static object Load(Stream stream)
        {
            BinaryFormatter bf = new BinaryFormatter();
            object o = bf.Deserialize(stream);
            stream.Close();
            return o;
        }

        public static int Main(string[] args)
        {
            try
            {
                Run(args);
                return 0;
            }
            catch (Exception e)
            {
                if (e.IsMarked())
                {
                    Console.Error.WriteLine(e.Message);
                    // Return a non-zero error code to indicate an error
                    //   a negative error code indicates to Aether that there was a failure (a positive
                    //   error code is still considered to have run correctly in Aether).
                    // Note, currently we don't use this executible in Aether, but this was done to be
                    //  consistent with TL.exe.
                    return -1;
                }
                else
                    throw;
            }
        }

        protected static void Run(string[] args)
        {
            ResultProcessorArguments cmd = new ResultProcessorArguments();
            TlcEnvironment env = new TlcEnvironment(42);
            List<PredictorResult> predictorResultsList = new List<PredictorResult>();
            PredictionUtil.ParseArguments(env, cmd, PredictionUtil.CombineSettings(args));

#if TLCFULLBUILD
            Report vizReport = null;
            if (!String.IsNullOrEmpty(cmd.VisualizationHtml))
                vizReport = new Report();
#endif

            if (cmd.IncludePerFoldResults)
                cmd.PerFoldResultSeparator = "" + PredictionUtil.SepCharFromString(cmd.PerFoldResultSeparator);
            foreach (var dll in cmd.ExtraAssemblies)
                ComponentCatalog.LoadAssembly(dll);

            if (cmd.Metrics.Length == 0)
                cmd.Metrics = null;
            if (cmd.Metrics != null)
            {
                List<string> m = new List<string>();
                foreach (string metric in cmd.Metrics)
                    m.AddRange(metric.Split(new char[] { ';', ',' }));
                cmd.Metrics = m.ToArray();
            }

            // <Tag, Pattern>
            List<KeyValuePair<string, string>> srcFiles = new List<KeyValuePair<string, string>>();

            foreach (string pattern in cmd.ResultFiles)
            {
                foreach (string src in StreamUtils.ExpandWildCards(pattern))
                    srcFiles.Add(new KeyValuePair<string, string>(null, src));
            }

            foreach (var taggedPattern in cmd.ResultFilesWithTags)
            {
                foreach (string src in StreamUtils.ExpandWildCards(taggedPattern.Value))
                    srcFiles.Add(new KeyValuePair<string, string>(taggedPattern.Key, src));
            }

            TextWriter outStream = (cmd.OutputFile != null && cmd.OutputFile.Length > 1
                ? File.CreateText(cmd.OutputFile) : Console.Out);

            //iterate each file to process it and extract the ExperimentItemResult object from it
            foreach (var fileWithTag in srcFiles)
            {
                ExperimentItemResult? result = ProcessOutputFile(env, fileWithTag.Value, cmd);
                if (result == null)
                    continue;

                ExperimentItemResult resultValue = result.Value;

                resultValue.CustomizedTag = fileWithTag.Key;

                bool newLearner;
                PredictorResult predictorItem = GetPredictorObject(predictorResultsList,
                    resultValue.Trainer.Kind, out newLearner);

                if (predictorItem.PredictorList == null)
                    predictorItem.PredictorList = new List<ExperimentItemResult>();

                predictorItem.PredictorList.Add(resultValue);
                predictorItem.Initialize(resultValue);
            }

            bool first = true;
            foreach (PredictorResult predictor in predictorResultsList)
            {
                string predictorName = predictor.LearnerName;
                first = true;
                outStream.WriteLine(predictorName);

                // print header for current predictor
                if (first)
                {
                    //outStream.Write( LearnerName +"\t" + ResultsFile);

                    foreach (string metric in predictor.ResultHeaderNames)
                        outStream.Write(metric + "\t");
                    foreach (string arg in predictor.SettingHeaderNames)
                        outStream.Write(arg + "\t");
                    outStream.Write(LearnerName + "\t" + TrainDataset + "\t" + TestDataset + "\t" + ResultsFile + "\t" + Runtime + "\t" + PhysicalMemoryUsage + "\t" + VirturalMemoryUsage + "\t" + CommandLineArgument + "\t" + Settings + "\t");
                    outStream.WriteLine();
                    first = false;
                }

                predictor.GetDefaultSettingValues(env, predictorName, cmd.ExtraAssemblies);

                // print each result
                foreach (ExperimentItemResult result in predictor.PredictorList)
                {
                    //print the result metrices
                    foreach (string name in predictor.ResultHeaderNames)
                    {
                        ResultMetric val;
                        if (result.Results.TryGetValue(name, out val))
                            outStream.Write(val.MetricValue);
                        outStream.Write("\t");
                    }

                    //print the settings for the run
                    foreach (string name in predictor.SettingHeaderNames)
                    {
                        string val = null;
                        if (result.Settings.TryGetValue(name, out val) || predictor.DefaultSettings.TryGetValue(name, out val))
                            outStream.Write(val);
                        outStream.Write("\t");
                    }

                    outStream.Write(predictorName + "\t" + result.Datafile + "\t" + result.TestDatafile + "\t" + result.InputFile + "\t" + result.Time + "\t" + result.PhysicalMemory + "\t" + result.VirtualMemory + "\t" + result.Commandline + "\t" + String.Join(";", result.GetSettings().ToArray()) + "\t");
                    // print per-metric results
                    if (cmd.IncludePerFoldResults && result.PerFoldResults != null)
                    {
                        foreach (var kvp in result.PerFoldResults)
                        {
                            if (Float.IsNaN(kvp.Value.MetricValue) && kvp.Value.AllValues != null)
                                outStream.Write("\t" + kvp.Key + ":"
                                    + string.Join(cmd.PerFoldResultSeparator, new List<string>(new List<Float>(kvp.Value.AllValues).Select(d => "" + d))));
                        }
                    }

                    outStream.WriteLine();

#if TLCFULLBUILD
                    if (vizReport != null)
                        vizReport.Experiments.Add(CreateVisualizationExperiment(result, vizReport.Experiments.Count));
#endif
                }
                outStream.WriteLine();
            }

            outStream.Close();

#if TLCFULLBUILD
            if (vizReport != null)
                ReportGenerator.SaveHtmlReport(cmd.VisualizationHtml.Trim(), vizReport, @"Html\Report_TLC.html");
#endif
        }

    }
}