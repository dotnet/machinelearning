// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    /// <summary>
    /// This is a base test class designed to support baseline comparison.
    /// </summary>
    public abstract partial class BaseTestBaseline : BaseTestClass
    {
        public const int DigitsOfPrecision = 7;

        protected BaseTestBaseline(ITestOutputHelper output) : base(output)
        {
        }

        internal const string RawSuffix = ".raw";
        private const string LogSuffix = ".log";
        private readonly string _logRootRelPath = Path.Combine("Logs", BuildString); // Relative to OutDir.

        private const string TestDir = @"test";

        private const string DataRootRegExp = @"[a-z]:\\[^/\t ]+\\test\\data" + @"\\[^/\t ]+";
        private const string SamplesRootRegExp = @"[a-z]:\\[^/\t ]+\\Samples\\";
        private const string SourceRootRegExp = @"[a-z]:\\[^/\t ]+\\source\\";
        private const string TestsRootRegExp = @"[a-z]:\\[^/\t ]+\\Tests\\";

        private const string DataRootUnixRegExp = @"\/[^\\\t ]+\/test\/data" + @"\/[^\\\t ]+";
        private const string SamplesRootUnixRegExp = @"\/[^\\\t ]+\/Samples\/[^\\\t ]+";
        private const string SourceRootUnixRegExp = @"\/[^\\\t ]+\/source\/[^\\\t ]+";
        private const string TestsRootUnixRegExp = @"\/[^\\\t ]+\/Tests\/[^\\\t ]+";

#if DEBUG
        private const string BuildString = "SingleDebug";
        private const string Mode = "Debug";
#else
        private const string BuildString = "SingleRelease";
        private const string Mode = "Release";
#endif

        private const string OutputRootRegExp = @"[a-z]:\\[^/\t ]+\\TestOutput" + @"\\[^/\t ]+";
        private static readonly string BinRegExp = @"[a-z]:\\[^\t ]+\\bin\\" + Mode;
        private static readonly string Bin64RegExp = @"[a-z]:\\[^/\t ]+\\bin\\x64\\" + Mode;

        private const string OutputRootUnixRegExp = @"\/[^\\\t ]+\/TestOutput" + @"\/[^\\\t ]+";
        private static readonly string BinRegUnixExp = @"\/[^\\\t ]+\/bin\/" + Mode;
        private static readonly string Bin64RegUnixExp = @"\/[^\\\t ]+\/bin\/x64\/" + Mode;
        // The Regex matches both positive and negative decimal point numbers present in a string.
        // The numbers could be a part of a word. They can also be in Exponential form eg. 3E-9 or 4E+07
        private static readonly Regex MatchNumbers = new Regex(@"-?\b[0-9]+\.?[0-9]*(E[-+][0-9]*)?\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        /// <summary>
        /// When the progress log is appended to the end of output (in test runs), this line precedes the progress log.
        /// </summary>
        protected const string ProgressLogLine = "--- Progress log ---";

        // Full paths to the baseline directories.
        private string _baselineCommonDir;
        private string _baselineBuildStringDir;

        // The writer to write to test log files.
        protected TestLogger TestLogger;
        protected StreamWriter LogWriter;
        private protected ConsoleEnvironment _env;
        protected IHostEnvironment Env => _env;
        protected MLContext ML;
        private bool _normal;
        private readonly List<Exception> _failures = new List<Exception>();

        protected override void Initialize()
        {
            base.Initialize();

            // Create the output and log directories.
            string baselineRootDir = Path.Combine(RootDir, TestDir, "BaselineOutput");
            Contracts.Check(Directory.Exists(baselineRootDir));

            _baselineCommonDir = Path.Combine(baselineRootDir, "Common");
            _baselineBuildStringDir = Path.Combine(baselineRootDir, BuildString);

            string logDir = Path.Combine(OutDir, _logRootRelPath);
            Directory.CreateDirectory(logDir);

            string logPath = Path.Combine(logDir, FullTestName + LogSuffix);
            LogWriter = OpenWriter(logPath);

            TestLogger = new TestLogger(Output);
            _env = new ConsoleEnvironment(42, outWriter: LogWriter, errWriter: LogWriter, testWriter: TestLogger)
                .AddStandardComponents();
            ML = new MLContext(42);
            ML.Log += LogTestOutput;
            ML.AddStandardComponents();
        }

        private void LogTestOutput(object sender, LoggingEventArgs e)
        {
            if (e.Kind >= MessageKindToLog)
                Output.WriteLine(e.Message);
        }

        // This method is used by subclass to dispose of disposable objects
        // such as LocalEnvironment.
        // It is called as a first step in test clean up.
        protected override void Cleanup()
        {
            _env = null;

            Contracts.Assert(IsActive);
            Log("Test {0}: {1}: {2}", TestName,
                _normal ? "completed normally" : "aborted",
                IsPassing ? "passed" : "failed");

            Contracts.AssertValue(LogWriter);
            LogWriter.Dispose();
            LogWriter = null;

            base.Cleanup();
        }

        protected bool IsActive { get { return LogWriter != null; } }

        protected bool IsPassing { get { return _failures.Count == 0; } }

        // Called by a test to signal normal completion. If this is not called before the
        // TestScope is disposed, we assume the test was aborted.
        protected void Done()
        {
            Contracts.Assert(IsActive);
            Contracts.Assert(!_normal, "Done() should only be called once!");
            _normal = true;

            switch (_failures.Count)
            {
                case 0:
                    break;

                case 1:
                    ExceptionDispatchInfo.Capture(_failures[0]).Throw();
                    break;

                default:
                    throw new AggregateException(_failures.ToArray());
            }
        }

        protected bool Check(bool f, string msg)
        {
            if (!f)
                Fail(msg);
            return f;
        }

        protected bool Check(bool f, string msg, params object[] args)
        {
            if (!f)
                Fail(msg, args);
            return f;
        }

        protected void Fail(string fmt, params object[] args)
        {
            Contracts.Assert(IsActive);
            try
            {
                throw new InvalidOperationException(string.Format(fmt, args));
            }
            catch (Exception ex)
            {
                _failures.Add(ex);
            }

            Log("*** Failure: " + fmt, args);
        }

        protected void Log(string msg)
        {
            Contracts.Assert(IsActive);
            Contracts.AssertValue(LogWriter);
            LogWriter.WriteLine(msg);
            Output.WriteLine(msg);
        }

        protected void Log(string fmt, params object[] args)
        {
            Contracts.Assert(IsActive);
            Contracts.AssertValue(LogWriter);
            LogWriter.WriteLine(fmt, args);
            Output.WriteLine(fmt, args);
        }

        protected string GetBaselinePath(string name)
        {
            Contracts.Assert(IsActive);
            if (string.IsNullOrWhiteSpace(name))
                return null;

            return GetBaselinePath(string.Empty, name);
        }

        protected string GetBaselinePath(string subDir, string name)
        {
            Contracts.Assert(IsActive);
            subDir = subDir ?? string.Empty;

            // first check the Common folder, and use it if it exists
            string commonBaselinePath = Path.GetFullPath(Path.Combine(_baselineCommonDir, subDir, name));
            if (File.Exists(commonBaselinePath))
            {
                return commonBaselinePath;
            }

            return Path.GetFullPath(Path.Combine(_baselineBuildStringDir, subDir, name));
        }

        // These are used to normalize output.
        private static readonly Regex _matchDataRoot = new Regex(DataRootRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchDataUnixRoot = new Regex(DataRootUnixRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchSamplesRoot = new Regex(SamplesRootRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchSamplesUnixRoot = new Regex(SamplesRootUnixRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchSourceRoot = new Regex(SourceRootRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchSourceUnixRoot = new Regex(SourceRootUnixRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTestsRoot = new Regex(TestsRootRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchOutputRoot = new Regex(OutputRootRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchOutputUnixRoot = new Regex(OutputRootUnixRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTL = new Regex(@"[a-z]:\\[a-z0-9_\.\\]+\\TL.exe", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTempFile = new Regex(@"[a-z]:\\users\\[a-z0-9_\.]+\\appdata\\local\\temp\\[a-z0-9_\.\\]*\.tmp", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTempDir = new Regex(@"[a-z]:\\users\\[a-z0-9_\.]+\\appdata\\local\\temp\\[a-z0-9_\.\\]+\\", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTempUnixDir = new Regex(@"\/(var\/)?tmp" + @"\/[^\\\t ]+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTempDirServiceProfile = new Regex(@"[a-z]:\\Windows\\ServiceProfiles\\[a-z0-9_\.]+\\appdata\\local\\temp\\[a-z0-9_\.\\]+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTempWindows = new Regex(@"[a-z]:\\Windows\\Temp\\[a-z0-9_\.]+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchDateTime = new Regex(@"[0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,4}(\.[0-9]+)?( [AP]M)?", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchTime = new Regex(@"[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?", RegexOptions.Compiled);
        private static readonly Regex _matchShortTime = new Regex(@"\([0-9]{2}:[0-9]{2}(\.[0-9]+)?\)", RegexOptions.Compiled);
        private static readonly Regex _matchMemory = new Regex(@"memory usage\(MB\): [0-9]+", RegexOptions.Compiled);
        private static readonly Regex _matchReservedMemory = new Regex(@": [0-9]+ bytes", RegexOptions.Compiled);
        private static readonly Regex _matchElapsed = new Regex(@"Time elapsed\(s\): [0-9.]+", RegexOptions.Compiled);
        private static readonly Regex _matchTimes = new Regex(@"Instances caching time\(s\): [0-9\.]+", RegexOptions.Compiled);
        private static readonly Regex _matchUpdatesPerSec = new Regex(@", ([0-9\.]+|Infinity)M WeightUpdates/sec", RegexOptions.Compiled);
        private static readonly Regex _matchParameterT = new Regex(@"=PARAM:/t:[0-9]+", RegexOptions.Compiled);
        private static readonly Regex _matchInfinity = new Regex(@"\u221E", RegexOptions.Compiled);
        private static readonly Regex _matchErrorLog = new Regex(@"Error_[\w-]+\.log", RegexOptions.Compiled);
        private static readonly Regex _matchGuid = new Regex(@"[A-F0-9]{8}(?:-[A-F0-9]{4}){3}-[A-F0-9]{12}", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchBin = new Regex(BinRegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchUnixBin = new Regex(BinRegUnixExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchBin64 = new Regex(Bin64RegExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex _matchUnixBin64 = new Regex(Bin64RegUnixExp, RegexOptions.IgnoreCase | RegexOptions.Compiled);

        protected void Normalize(string path)
        {
            string rawPath = path + RawSuffix;
            File.Delete(rawPath);
            File.Move(path, rawPath);

            using (StreamReader src = OpenReader(rawPath))
            using (TextWriter dst = File.CreateText(path))
            {
                string line;
                while ((line = src.ReadLine()) != null)
                {
                    line = _matchDataRoot.Replace(line, "%Data%");
                    line = _matchDataUnixRoot.Replace(line, "%Data%");
                    line = _matchOutputRoot.Replace(line, "%Output%");
                    line = _matchOutputUnixRoot.Replace(line, "%Output%");
                    line = _matchSamplesRoot.Replace(line, "%Samples%\\");
                    line = _matchSamplesUnixRoot.Replace(line, "%Samples%\\");
                    line = _matchSourceRoot.Replace(line, "%Source%\\");
                    line = _matchSourceUnixRoot.Replace(line, "%Source%\\");
                    line = _matchTestsRoot.Replace(line, "%Tests%\\");
                    line = _matchBin.Replace(line, "%Bin%\\");
                    line = _matchUnixBin.Replace(line, "%Bin%\\");
                    line = _matchBin64.Replace(line, "%Bin%\\");
                    line = _matchUnixBin64.Replace(line, "%Bin%\\");
                    line = _matchTL.Replace(line, "%TL%");
                    line = _matchTempFile.Replace(line, "%Temp%");
                    line = _matchTempDir.Replace(line, "%Temp%\\");
                    line = _matchTempUnixDir.Replace(line, "%Temp%\\");
                    line = _matchTempDirServiceProfile.Replace(line, "%Temp%");
                    line = _matchTempWindows.Replace(line, "%Temp%");
                    line = _matchDateTime.Replace(line, "%DateTime%");
                    line = _matchTime.Replace(line, "%Time%");
                    line = _matchShortTime.Replace(line, "(%Time%)");
                    line = _matchElapsed.Replace(line, "Time elapsed(s): %Number%");
                    line = _matchMemory.Replace(line, "memory usage(MB): %Number%");
                    line = _matchReservedMemory.Replace(line, ": %Number% bytes");
                    line = _matchTimes.Replace(line, "Instances caching time(s): %Number%");
                    line = _matchUpdatesPerSec.Replace(line, ", %Number%M WeightUpdates/sec");
                    line = _matchParameterT.Replace(line, "=PARAM:/t:%Number%");
                    line = _matchInfinity.Replace(line, "Infinity");
                    line = _matchErrorLog.Replace(line, "%ErrorLog%");
                    line = _matchGuid.Replace(line, "%Guid%");
                    dst.WriteLine(line);
                }
            }
        }

        /// <summary>
        /// Compare the contents of an output file with its baseline.
        /// </summary>
        protected bool CheckEquality(string dir, string name, string nameBase = null,
            int digitsOfPrecision = DigitsOfPrecision, NumberParseOption parseOption = NumberParseOption.Default)
        {
            return CheckEqualityCore(dir, name, nameBase ?? name, false, digitsOfPrecision, parseOption);
        }

        /// <summary>
        /// Check whether two files are same ignoring volatile differences (path, dates, times, etc).
        /// Returns true if the check passes.
        /// </summary>
        protected bool CheckEqualityNormalized(string dir, string name, string nameBase = null,
            int digitsOfPrecision = DigitsOfPrecision, NumberParseOption parseOption = NumberParseOption.Default)
        {
            return CheckEqualityCore(dir, name, nameBase ?? name, true, digitsOfPrecision, parseOption);
        }

        protected bool CheckEqualityCore(string dir, string name, string nameBase, bool normalize,
            int digitsOfPrecision = DigitsOfPrecision, NumberParseOption parseOption = NumberParseOption.Default)
        {
            Contracts.Assert(IsActive);
            Contracts.AssertValue(dir); // Can be empty.
            Contracts.AssertNonEmpty(name);
            Contracts.AssertNonEmpty(nameBase);

            // The following assert is necessary since some tests were attempting to
            // combine the ZBasline directory with an absolute path, leading to an output
            // file being compared with itself.
            Contracts.Assert(!Path.IsPathRooted(name), "file name should not be a full path");
            Contracts.Assert(!Path.IsPathRooted(nameBase), "file nameBase should not be a full path");

            string relPath = Path.Combine(dir, name);
            string basePath = GetBaselinePath(dir, nameBase);
            string outPath = GetOutputPath(dir, name);

            if (!CheckOutFile(outPath))
                return false;

            // Normalize the output file.
            if (normalize)
                Normalize(outPath);

            if (!CheckBaseFile(basePath))
                return false;

            bool res = CheckEqualityFromPathsCore(relPath, basePath, outPath, digitsOfPrecision: digitsOfPrecision, parseOption: parseOption);

            // No need to keep the raw (unnormalized) output file.
            if (normalize && res)
                File.Delete(outPath + RawSuffix);

            return res;
        }

        private bool FirstIsSuffix<T>(IEnumerator<T> suffix, IEnumerator<T> seq, Func<T, T, bool> equalFunc = null)
        {
            Contracts.AssertValue(suffix);
            Contracts.AssertValue(seq);
            Contracts.AssertValueOrNull(equalFunc);

            // It is possible to compose something that only stores part of suffix, and none of sequence,
            // but this is relatively harder to code.
            if (equalFunc == null)
                equalFunc = EqualityComparer<T>.Default.Equals;
            List<T> suffixList = new List<T>();
            List<T> seqList = new List<T>();
            while (suffix.MoveNext())
            {
                if (!seq.MoveNext())
                {
                    Fail("Baseline sequence had {0} items, but the suffix seems to have more", suffixList.Count);
                    return false;
                }
                suffixList.Add(suffix.Current);
                seqList.Add(seq.Current);
            }
            if (suffixList.Count == 0) // Empty suffix is trivially a suffix of anything.
                return true;
            Contracts.Assert(suffixList.Count == seqList.Count);
            int idx = 0;
            while (seq.MoveNext())
                seqList[idx++ % seqList.Count] = seq.Current;
            Log("Suffix of length {0} compared against sequence of length {1}", suffixList.Count, seqList.Count + idx);
            for (int i = 0; i < suffixList.Count; ++i)
            {
                if (!equalFunc(suffixList[i], seqList[(idx + i) % seqList.Count]))
                {
                    Fail("Baseline sequence mismatched {0} length suffix {1} item from the end",
                        suffixList.Count, suffixList.Count - i - 1);
                    return false;
                }
            }
            return true;
        }

        private IEnumerator<string> LineEnumerator(TextReader reader, Func<string, bool> stop)
        {
            string result;
            while ((result = reader.ReadLine()) != null && !stop(result))
                yield return result;
        }

        /// <summary>
        /// Checks that <paramref name="outPath"/>'s contents are a suffix of <paramref name="basePath"/>'s
        /// contents, assuming one skips <paramref name="skip"/> lines from <paramref name="outPath"/>, and 
        /// the file is read up to the <paramref name="tailSignature"/> line (or to the end, if it's not provided).
        /// </summary>
        protected bool CheckOutputIsSuffix(string basePath, string outPath, int skip = 0, string tailSignature = null)
        {
            Contracts.Assert(skip >= 0);
            Contracts.AssertValueOrNull(tailSignature);

            using (StreamReader baseline = OpenReader(basePath))
            using (StreamReader result = OpenReader(outPath))
            {
                while (--skip >= 0)
                    result.ReadLine();
                Func<string, bool> stop = x => x == tailSignature;
                return FirstIsSuffix(LineEnumerator(result, stop), LineEnumerator(baseline, stop), (a, b) => a == b);
            }
        }

        protected bool CheckEqualityFromPathsCore(string relPath, string basePath, string outPath, int skip = 0,
            int digitsOfPrecision = DigitsOfPrecision, NumberParseOption parseOption = NumberParseOption.Default)
        {
            Contracts.Assert(skip >= 0);

            using (StreamReader baseline = OpenReader(basePath))
            using (StreamReader result = OpenReader(outPath))
            {
                int count = 0;
                if (skip > 0)
                {
                    string line2;
                    do
                    {
                        line2 = result.ReadLine();
                        if (line2 == null)
                        {
                            Fail("Output is shorter than the skip value of {0}!", skip);
                            return false;
                        }
                        count++;
                    } while (count <= skip);

                    string line1;
                    do
                    {
                        line1 = baseline.ReadLine();
                        if (line1 == null)
                        {
                            Fail("Couldn't match output file line to a line in the baseline!");
                            return false;
                        }
                    } while (line1 != line2);
                }

                for (; ; )
                {
                    // read lines while we can
                    string line1 = baseline.ReadLine();
                    string line2 = result.ReadLine();

                    if (line1 == null && line2 == null)
                    {
                        Log("Output matches baseline: '{0}'", relPath);
                        return true;
                    }

                    count++;
                    var inRange = GetNumbersFromFile(ref line1, ref line2, digitsOfPrecision, parseOption);

                    if (!inRange || line1 != line2)
                    {
                        if (line1 == null || line2 == null)
                            Fail("Output and baseline different lengths: '{0}'", relPath);
                        else
                            Fail("Output and baseline mismatch at line {1}, expected '{2}' but got '{3}' : '{0}'", relPath, count, line1, line2);
                        return false;
                    }
                }
            }
        }

        private bool GetNumbersFromFile(ref string firstString, ref string secondString,
            int digitsOfPrecision, NumberParseOption parseOption)
        {
            MatchCollection firstCollection = MatchNumbers.Matches(firstString);
            MatchCollection secondCollection = MatchNumbers.Matches(secondString);

            if (firstCollection.Count == secondCollection.Count)
            {
                if (!MatchNumberWithTolerance(firstCollection, secondCollection, digitsOfPrecision, parseOption))
                {
                    return false;
                }
            }

            firstString = MatchNumbers.Replace(firstString, "%Number%");
            secondString = MatchNumbers.Replace(secondString, "%Number%");
            return true;
        }

        private bool MatchNumberWithTolerance(MatchCollection firstCollection, MatchCollection secondCollection,
            int digitsOfPrecision, NumberParseOption parseOption)
        {
            if (parseOption == NumberParseOption.UseSingle)
            {
                for (int i = 0; i < firstCollection.Count; i++)
                {
                    float f1 = float.Parse(firstCollection[i].ToString());
                    float f2 = float.Parse(secondCollection[i].ToString());

                    if (!CompareNumbersWithTolerance(f1, f2, i, digitsOfPrecision))
                    {
                        return false;
                    }
                }
            }
            else if (parseOption == NumberParseOption.UseDouble)
            {
                for (int i = 0; i < firstCollection.Count; i++)
                {
                    double f1 = double.Parse(firstCollection[i].ToString());
                    double f2 = double.Parse(secondCollection[i].ToString());

                    if (!CompareNumbersWithTolerance(f1, f2, i, digitsOfPrecision))
                    {
                        return false;
                    }
                }
            }
            else
            {
                throw new ArgumentException($"Invalid {nameof(NumberParseOption)}", nameof(parseOption));
            }

            return true;
        }

        public bool CompareNumbersWithTolerance(double expected, double actual, int? iterationOnCollection = null, 
            int digitsOfPrecision = DigitsOfPrecision, bool logFailure = true)
        {
            if (double.IsNaN(expected) && double.IsNaN(actual))
                return true;

            // this follows the IEEE recommendations for how to compare floating point numbers
            double allowedVariance = Math.Pow(10, -digitsOfPrecision);
            double delta = Round(expected, digitsOfPrecision) - Round(actual, digitsOfPrecision);
            // limitting to the digits we care about. 
            delta = Math.Round(delta, digitsOfPrecision);

            bool inRange = delta > -allowedVariance && delta < allowedVariance;

            // for some cases, rounding up is not beneficial
            // so checking on whether the difference is significant prior to rounding, and failing only then. 
            // example, for 5 digits of precision. 
            // F1 = 1.82844949 Rounds to 1.8284
            // F2 = 1.8284502  Rounds to 1.8285
            // would fail the inRange == true check, but would suceed the following, and we doconsider those two numbers 
            // (1.82844949 - 1.8284502) = -0.00000071

            double delta2 = 0;
            if (!inRange)
            {
                delta2 = Math.Round(expected - actual, digitsOfPrecision);
                inRange = delta2 >= -allowedVariance && delta2 <= allowedVariance;
            }

            if (!inRange)
            {
                var message = iterationOnCollection != null ? "" : $"Output and baseline mismatch at line {iterationOnCollection}." + Environment.NewLine;

                if(logFailure)
                    Fail(message +
                            $"Values to compare are {expected} and {actual}" + Environment.NewLine +
                            $"\t AllowedVariance: {allowedVariance}" + Environment.NewLine +
                            $"\t delta: {delta}" + Environment.NewLine +
                            $"\t delta2: {delta2}" + Environment.NewLine);
            }

            return inRange;
        }

        private static double Round(double value, int digitsOfPrecision)
        {
            if ((value == 0) || double.IsInfinity(value) || double.IsNaN(value))
            {
                return value;
            }

            double absValue = Math.Abs(value);
            double integralDigitCount = Math.Floor(Math.Log10(absValue) + 1);

            double scale = Math.Pow(10, integralDigitCount);
            return scale * Math.Round(value / scale, digitsOfPrecision);
        }

#if TOLERANCE_ENABLED
        // This corresponds to how much relative error is tolerable for a value of 0.
        const Float RelativeToleranceStepSize = (Float)0.001;
        const Float AbsoluteTolerance = (Float)0.00001;

        /// <summary>
        /// Check whether two files are same, allowing for some tolerance in comparing numeric values
        /// </summary>
        protected void CheckEqualityApprox(string dir, string name, bool normalize)
        {
            Contracts.Assert(IsActive);
            Contracts.AssertValue(dir); // Can be empty.
            Contracts.AssertNonEmpty(name);
            //Contracts.Assert(0 <= tolerance && tolerance < Float.PositiveInfinity);

            string relPath = Path.Combine(dir, name);
            string basePath = GetBaselinePath(dir, name);
            string outPath = GetOutputPath(dir, name);

            if (!CheckOutFile(outPath))
                return;

            // Normalize the output file.
            if (normalize)
                Normalize(outPath);

            if (!CheckBaseFile(basePath))
                return;

            CheckEqualityApproxCore(relPath, basePath, outPath, normalize);
        }

        private void CheckEqualityApproxCore(string relPath, string basePath, string outPath, bool normalize)
        {
            string line1, line2;
            using (StreamReader perfect = new StreamReader(basePath))
            using (StreamReader result = new StreamReader(outPath))
            {
                int count = 0;
                for (; ; )
                {
                    line1 = perfect.ReadLine();
                    line2 = result.ReadLine();
                    if (line1 == null && line2 == null)
                    {
                        Log("Output matches baseline: '{0}'", relPath);
                        // No need to keep the raw (unnormalized) output file.
                        if (normalize)
                            File.Delete(outPath + RawSuffix);
                        return;
                    }
                    if (line1 == null || line2 == null)
                    {
                        Fail("Output and baseline different lengths: '{0}'", relPath);
                        return;
                    }

                    count++;
                    if (line1 != line2)
                    {
                        String[] Line1 = line1.Split(Seperators);
                        String[] Line2 = line2.Split(Seperators);
                        if (Line1.Length != Line2.Length)
                        {
                            Fail("Output and baseline mismatch at line {1}: '{0}'", relPath, count);
                            return;
                        }

                        // string are same length, go through and try matching.
                        for (int i = 0; i < Line1.Length; i++)
                        {
                            Float first = Float.MinValue, second = Float.MinValue;
                            // couldn't parse either string. TODO: fix this bug!
                            // REVIEW: What should this do if parsing fails?
                            bool firstParseResult = Float.TryParse(Line1[i], out first);
                            bool secondParseResult = Float.TryParse(Line2[i], out second);
                            bool firstIsNan = Float.IsNaN(first);
                            bool secondIsNan = Float.IsNaN(second);
                            bool hasError = false;
                            hasError |= firstParseResult ^ secondParseResult;
                            hasError |= !firstParseResult && !secondParseResult && Line1[i] != Line2[i];
                            hasError |= firstIsNan ^ secondIsNan;
                            hasError |= !firstIsNan && !secondIsNan &&
                                        firstParseResult && secondParseResult &&
                                        Math.Abs(first) > 0.00001 ?
                                           (Math.Abs(first - second) / Math.Abs(first) > GetTolerance(first)) :
                                           Math.Abs(first - second) > AbsoluteTolerance;

                            if (hasError)
                            {
                                Fail("Output and baseline mismatch at line {1}, field {2}: '{0}'", relPath, count, i + 1);
                                Log("  Baseline: {0}", line1);
                                Log("  Output  : {0}", line2);
                                return;
                            }
                        }
                    }
                }
            }
        }

        // REVIEW: Maybe need revamping. As values get closer to 0 , the tolerance needs to increase.
        // REVIEW: Every test should be able to override the RelativeToleranceStepSize.
        private double GetTolerance(Float value)
        {
            double stepSizeMultiplier;
            if (value == 0)
                stepSizeMultiplier = 1;
            else
                stepSizeMultiplier = Math.Log((double)Math.Abs(value));

            if (stepSizeMultiplier <= 0)
            {
                stepSizeMultiplier = Math.Max(1, Math.Abs(stepSizeMultiplier));
                // tolerance needs to increase faster as we start going towards smaller values.
                stepSizeMultiplier *= stepSizeMultiplier;
            }
            else
                stepSizeMultiplier = Math.Min(1, 1 / stepSizeMultiplier);

            return stepSizeMultiplier * RelativeToleranceStepSize;
        }
#endif
        /// <summary>
        /// Make sure the baseline and output files exists. Optionally creates the baseline if it is missing.
        /// </summary>
        private bool CheckOutFile(string outPath)
        {
            if (!File.Exists(outPath))
            {
                Fail("Output file not found: {0}", outPath);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Make sure the baseline and output files exists. Optionally creates the baseline if it is missing.
        /// </summary>
        private bool CheckBaseFile(string basePath)
        {
            if (!File.Exists(basePath))
            {
                Fail("Baseline file not found: {0}", basePath);
                return false;
            }

            return true;
        }

        public void RunMTAThread(ThreadStart fn)
        {
            Exception inner = null;
            var t = new Thread(() =>
            {
                try
                {
                    fn();
                }
                catch (Exception e)
                {
                    inner = e;
                    Fail("The test threw an exception - {0}", e);
                }
            });
            t.IsBackground = true;
            t.Start();
            t.Join();
            if (inner != null)
                throw Contracts.Except(inner, "MTA thread failed");
        }

        /// <summary>
        /// Opens a stream writer for the specified file using the specified encoding and buffer size.
        /// If the file exists, it can be either overwritten or appended to. 
        /// If the file does not exist, a new file is created.
        /// </summary>
        /// <param name="path">The complete file path to write to.</param>
        /// <param name="append">
        /// true to append data to the file; false to overwrite the file. 
        /// If the specified file does not exist, this parameter has no effect and a new file is created.
        /// </param>
        /// <param name="encoding">The character encoding to use.</param>
        /// <param name="bufferSize">The buffer size, in bytes.</param>
        /// <returns>The stream writer to write to the specified file.</returns>
        protected static StreamWriter OpenWriter(string path, bool append = false, Encoding encoding = null, int bufferSize = 1024)
        {
            Contracts.CheckNonWhiteSpace(path, nameof(path));

            return Utils.OpenWriter(File.Open(path, append ? FileMode.Append : FileMode.OpenOrCreate), encoding, bufferSize, false);
        }

        /// <summary>
        /// Opens a stream reader for the specified file name, with the specified character encoding, byte order mark detection option, and buffer size.
        /// </summary>
        /// <param name="path">The complete file path to be read.</param>
        /// <returns>The stream reader to read from the specified file.</returns>
        protected static StreamReader OpenReader(string path)
        {
            Contracts.CheckNonWhiteSpace(path, nameof(path));

            return new StreamReader(File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read));
        }

        /// <summary>
        /// Invoke MAML with specified arguments without output baseline. 
        /// This method is used in unit tests when the output is not baselined.
        /// If the output is to be baselined and compared, the other overload should be used.
        /// </summary>
        protected int MainForTest(string args)
        {
            return Maml.MainCore(ML, args, false);
        }

        public enum NumberParseOption
        {
            Default = UseDouble,
            UseSingle = 1,
            UseDouble = 2,
        }
    }

    public partial class TestBaselineNormalize : BaseTestBaseline
    {
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Baselines")]
        public void TestTimeRegex()
        {
            const string outDir = @"..\Common\BaselineNormalize";
            const string name = "TestTimeRegex.txt";

            string path = DeleteOutputPath(outDir, name);
            File.WriteAllText(path,
@"00:04:58:              Starting to train ...
00:04:58.3572:              Starting to train ...
00:04:58.3xy7z3:              Starting to train ...
"
                );
            CheckEqualityNormalized(outDir, name);

            Done();
        }
    }
}
