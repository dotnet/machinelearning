// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper;

using ResultProcessorInternal = Microsoft.ML.Runtime.Internal.Internallearn.ResultProcessor;

[assembly: LoadableClass(typeof(LocalExeConfigRunner), typeof(LocalExeConfigRunner.Arguments), typeof(SignatureConfigRunner),
    "Local Sweep Config Runner", "Local")]

namespace Microsoft.ML.Runtime.Sweeper
{
    public delegate void SignatureConfigRunner();

    public interface IConfigRunner
    {
        IEnumerable<IRunResult> RunConfigs(ParameterSet[] sweeps, int min);
        void Finish();
        string GetOutputFolderPath(string folderName);
    }

    public abstract class ExeConfigRunnerBase : IConfigRunner
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Command pattern for the sweeps", ShortName = "pattern")]
            public string ArgsPattern;

            [Argument(ArgumentType.AtMostOnce, HelpText = "output folder for the outputs of the sweeps", ShortName = "outfolder")]
            public string OutputFolderName;

            [Argument(ArgumentType.AtMostOnce, HelpText = "prefix to add to the output file names", ShortName = "pre")]
            public string Prefix;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The executable name, including the path (the default is MAML.exe)")]
            public string Exe;

            [Argument(ArgumentType.Multiple, HelpText = "Specify how to extract the metrics from the result file.", ShortName = "ev", SignatureType = typeof(SignatureSweepResultEvaluator))]
            public IComponentFactory<ISweepResultEvaluator<string>> ResultProcessor = ComponentFactoryUtils.CreateFromFunction(
                env => new InternalSweepResultEvaluator(env, new InternalSweepResultEvaluator.Arguments()));

            [Argument(ArgumentType.AtMostOnce, Hide = true)]
            public bool CalledFromUnitTestSuite;
        }

        protected string Exe;
        protected readonly string ArgsPattern;
        protected readonly string OutputFolder;
        protected readonly string Prefix;
        protected readonly ISweepResultEvaluator<string> ResultProcessor;
        protected readonly List<int> RunNums;

        protected readonly IHost Host;

        private readonly bool _calledFromUnitTestSuite;

        protected ExeConfigRunnerBase(ArgumentsBase args, IHostEnvironment env, string registrationName)
        {
            Contracts.AssertValue(env);
            Host = env.Register(registrationName);
            Host.CheckUserArg(!string.IsNullOrEmpty(args.ArgsPattern), nameof(args.ArgsPattern), "The command pattern is missing");
            Host.CheckUserArg(!string.IsNullOrEmpty(args.OutputFolderName), nameof(args.OutputFolderName), "Please specify an output folder");
            ArgsPattern = args.ArgsPattern;
            OutputFolder = GetOutputFolderPath(args.OutputFolderName);
            Prefix = string.IsNullOrEmpty(args.Prefix) ? "" : args.Prefix;
            ResultProcessor = args.ResultProcessor.CreateComponent(Host);
            _calledFromUnitTestSuite = args.CalledFromUnitTestSuite;
            RunNums = new List<int>();
        }

        protected virtual void ProcessFullExePath(string exe)
        {
            Exe = GetFullExePath(exe);

            if (!File.Exists(Exe) && !File.Exists(Exe + ".exe"))
                throw Host.ExceptUserArg(nameof(ArgumentsBase.Exe), "Executable {0} not found", Exe);
        }

        protected virtual string GetFullExePath(string exe)
        {
            if (!string.IsNullOrWhiteSpace(exe))
                return exe;
#if CORECLR
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
                return Path.Combine(SweepCommand.LocalExePath, "../Win/maml.exe");
            //REVIEW: Need mac support
            return Path.Combine(SweepCommand.LocalExePath, "../Linux/maml");
#else
            return Path.Combine(SweepCommand.LocalExePath, "maml.exe");
#endif
        }

        public virtual void Finish()
        {
            if (Exe == null || Exe.EndsWith("maml", StringComparison.OrdinalIgnoreCase) ||
                Exe.EndsWith("maml.exe", StringComparison.OrdinalIgnoreCase))
            {
                using (var ch = Host.Start("Finish"))
                {
                    var runs = RunNums.ToArray();
                    var args = Utils.BuildArray(RunNums.Count + 2,
                        i =>
                        {
                            if (i == RunNums.Count)
                                return string.Format(@"o={{{0}\{1}.summary.txt}}", OutputFolder, Prefix);
                            if (i == RunNums.Count + 1)
                                return string.Format("calledFromUnitTestSuite{0}", _calledFromUnitTestSuite ? "+" : "-");
                            return string.Format("{{{0}}}", GetFilePath(runs[i], "out"));
                        });

                    ResultProcessorInternal.ResultProcessor.Main (args);

                    ch.Info(@"The summary of the run results has been saved to the file {0}\{1}.summary.txt", OutputFolder, Prefix);
                }
            }
        }

        public virtual string GetOutputFolderPath(string folderName)
        {
            var folderPath = Path.GetFullPath(folderName);

            try
            {
                if (!Directory.Exists(folderName))
                    Directory.CreateDirectory(folderName);
                return folderPath;
            }
            catch (Exception e)
            {
                throw Host.Except(e, e.Message);
            }
        }

        // REVIEW: in case we want to use sweep command on linux we need to reconsider our syntax.
        // $something get treated in bash as variable something and if you have command line which looks like:
        // lr=$LR$
        // you get lr=$ only as argument because $LR is variable and empty.
        protected string GetCommandLine(ParameterSet sweep)
        {
            var arguments = ArgsPattern;
            foreach (var parameterValue in sweep)
                arguments = arguments.Replace("$" + parameterValue.Name + "$", parameterValue.ValueText);
            return arguments;
        }

        public IEnumerable<IRunResult> RunConfigs(ParameterSet[] sweeps, int min)
        {
            RunNums.AddRange(Enumerable.Range(min, sweeps.Length));

            using (var ch = Host.Start("Evaluate"))
            {
                for (int i = 0; i < sweeps.Length; i++)
                    ch.Info("Parameter set: {0}", string.Join(", ", sweeps[i].Select(p => string.Format("{0}:{1}", p.Name, p.ValueText))));

               return RunConfigsCore(sweeps, ch, min);
            }
        }

        protected string GetFilePath(int i, string kind)
        {
            return string.Format(@"{0}\{1}{2}.{3}.txt", OutputFolder, Prefix, i, kind);
        }

        protected abstract IEnumerable<IRunResult> RunConfigsCore(ParameterSet[] sweeps, IChannel ch, int min);
    }

    public sealed class LocalExeConfigRunner : ExeConfigRunnerBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of threads to use for the sweep (default auto determined by the number of cores)", ShortName = "t")]
            public int? NumThreads;
        }

        private readonly ParallelOptions _parallelOptions;

        public LocalExeConfigRunner(IHostEnvironment env, Arguments args)
            : base(args, env, "LocalExeSweepEvaluator")
        {
            Contracts.CheckParam(args.NumThreads == null || args.NumThreads.Value > 0, nameof(args.NumThreads), "Cannot be 0 or negative");
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = args.NumThreads ?? -1 };
            Contracts.AssertNonEmpty(args.OutputFolderName);
            ProcessFullExePath(args.Exe);
        }

        protected override IEnumerable<IRunResult> RunConfigsCore(ParameterSet[] sweeps, IChannel ch, int min)
        {
            Parallel.For(0, sweeps.Length, _parallelOptions, j =>
            {
                var outFile = GetFilePath(min + j, "out");
                var errorFile = GetFilePath(min + j, "err");
                var arguments = GetCommandLine(sweeps[j]);
                RunProcess(Exe, new string[] { arguments }, Environment.CurrentDirectory,
                    new StreamWriter(outFile),
                    new StreamWriter(errorFile));

                if (File.Exists(errorFile) && new FileInfo(errorFile).Length == 0)
                {
                    File.Delete(errorFile);
                }
            });
            return sweeps.Select((sweep, j) =>
                ResultProcessor.GetRunResult(sweep, string.Format(@"{0}\{1}.out.txt", OutputFolder, min + j)));
        }

        /// <summary>
        /// Run specified EXE with given arguments
        /// </summary>
        private void RunProcess(string exeFilename, string[] args, string workingDir,
            TextWriter standardOutputWriter = null, TextWriter standardErrorWriter = null)
        {
            var p = new System.Diagnostics.Process
            {
                StartInfo =
                {
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    FileName = exeFilename,
                    Arguments = (args == null ? "" : string.Join(" ", args)),
                }
            };

            if (workingDir != null)
                p.StartInfo.WorkingDirectory = workingDir;

            if (standardOutputWriter != null)
            {
                p.StartInfo.RedirectStandardOutput = true;
                p.OutputDataReceived += (s, a) => { if (a.Data != null) standardOutputWriter.WriteLine(a.Data); };
            }

            if (standardErrorWriter != null)
            {
                p.StartInfo.RedirectStandardError = true;
                p.ErrorDataReceived += (s, a) => { if (a.Data != null) standardErrorWriter.WriteLine(a.Data); };
            }

            p.Start();
            //p.EnableRaisingEvents = true; // REVIEW: Why would you claim you wanted to
            // use the async exit handler, only to just use WaitForExit downstream?
            if (standardOutputWriter != null)
                p.BeginOutputReadLine();
            if (standardErrorWriter != null)
                p.BeginErrorReadLine();
            p.WaitForExit();

            if (standardOutputWriter != null)
            {
                standardOutputWriter.Flush();
                standardOutputWriter.Close();
            }

            if (standardErrorWriter != null)
            {
                standardErrorWriter.Flush();
                standardErrorWriter.Close();
            }
        }
    }
}
