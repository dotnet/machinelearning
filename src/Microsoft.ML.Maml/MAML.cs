// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;

#if CORECLR
using Microsoft.ML.Runtime.Internal.Utilities;
#endif

#if !CORECLR
using System.Configuration;
#endif

namespace Microsoft.ML.Runtime.Tools
{
    public static class Maml
    {
        /// <summary>
        /// Main command line entry point.
        /// </summary>
#if CORECLR
        public static int Main(string[] args)
        {
            string all = string.Join(" ", args);
            return MainAll(all);
        }

        public static unsafe int MainRaw(char* psz)
        {
            string args = new string(psz);
            return MainAll(args);
        }

        public static int MainAll(string args)
        {
            return MainWithProgress(args);
        }
#else
        public static int Main()
        {
            string args = CmdParser.TrimExePath(Environment.CommandLine, out string exe);
            return MainWithProgress(args);
        }
#endif

        private static int MainWithProgress(string args)
        {
            using (var env = CreateEnvironment())
            using (var progressCancel = new CancellationTokenSource())
            {
                var progressTrackerTask = Task.Run(() => TrackProgress(env, progressCancel.Token));
                try
                {
                    return MainCore(env, args, ShouldAlwaysPrintStacktrace());
                }
                finally
                {
                    progressCancel.Cancel();
                    progressTrackerTask.Wait();
                    // If the run completed so quickly that the progress task was cancelled before it even got a chance to start,
                    // we need to gather the checkpoints.
                    env.PrintProgress();
                }
            }
        }

        private static bool ShouldAlwaysPrintStacktrace() => false;

        private static TlcEnvironment CreateEnvironment()
        {
            string sensitivityString = null;
            MessageSensitivity sensitivity = MessageSensitivity.All;
            if (!string.IsNullOrWhiteSpace(sensitivityString))
            {
                // Cannot use host or channels since the environment isn't even
                // created yet.
                if (!Enum.TryParse(sensitivityString, out sensitivity))
                {
                    Console.Error.WriteLine("Cannot parse '{0}' as {1}", sensitivityString, nameof(MessageSensitivity));
                    sensitivity = MessageSensitivity.All;
                }
            }
            return new TlcEnvironment(sensitivity: sensitivity);
        }

        /// <summary>
        /// The main method to invoke TLC, with some high level configuration options set.
        /// </summary>
        /// <param name="env">The environment used in this run of TLC, for the purpose of returning outputs.</param>
        /// <param name="args">The command line arguments.</param>
        /// <param name="alwaysPrintStacktrace">"Marked" exceptions are assumed to be sufficiently descriptive, so we
        /// do not print stack traces for them to the console, and instead print these only to a log file.
        /// However, throwing unmarked exceptions is considered a bug in TLC (even if due to bad user input),
        /// so we always write . If set to true though, this executable will also print stack traces from the
        /// marked exceptions as well.</param>
        /// <returns></returns>
        internal static int MainCore(TlcEnvironment env, string args, bool alwaysPrintStacktrace)
        {
            // REVIEW: How should extra dlls, tracking, etc be handled? Should the args objects for
            // all commands derive from a common base?
            var mainHost = env.Register("Main");
            using (var telemetryPipe = mainHost.StartPipe<TelemetryMessage>("TelemetryPipe"))
            using (var ch = mainHost.Start("Main"))
            {
                int result;
                try
                {
                    if (!CmdParser.TryGetFirstToken(args, out string kind, out string settings))
                    {
                        telemetryPipe.Send(TelemetryMessage.CreateCommand("ArgumentParsingFailure", args));
                        Usage();
                        return -1;
                    }

                    var cmdDef = new SubComponent<ICommand, SignatureCommand>(kind, settings);

                    if (!ComponentCatalog.TryCreateInstance(mainHost, out ICommand cmd, cmdDef))
                    {
                        // Telemetry: Log
                        telemetryPipe.Send(TelemetryMessage.CreateCommand("UnknownCommand", settings));
                        ch.Error("Unknown command: '{0}'", kind);
                        Usage();
                        return -1;
                    }

                    // Telemetry: Log the command and settings.
                    telemetryPipe.Send(TelemetryMessage.CreateCommand(kind.ToUpperInvariant(), settings));
                    cmd.Run();

                    result = 0;
                }
                catch (Exception ex)
                {
                    var dumpFileDir = Path.Combine(
                        Path.GetTempPath(),
                        "TLC");
                    var dumpFilePath = Path.Combine(dumpFileDir,
                        string.Format(CultureInfo.InvariantCulture, "Error_{0:yyyyMMdd_HHmmss}_{1}.log", DateTimeOffset.Now.UtcDateTime, Guid.NewGuid()));
                    bool isDumpSaved = false;
                    try
                    {
                        Directory.CreateDirectory(dumpFileDir);
                        // REVIEW: Should specify the encoding.
                        using (var sw = new StreamWriter(new FileStream(dumpFilePath, FileMode.Create, FileAccess.Write)))
                        {
                            sw.WriteLine("--- Command line args ---");
                            sw.WriteLine(args);
                            sw.WriteLine("--- Exception message ---");
                            PrintFullExceptionDetails(sw, ex);
                        }

                        isDumpSaved = true;
                    }
                    catch (Exception)
                    {
                        // Don't throw an exception if we failed to write to the dump file.
                    }

                    // Process exceptions that we understand.
                    int count = 0;
                    for (var e = ex; e != null; e = e.InnerException)
                    {
                        // Telemetry: Log the exception
                        telemetryPipe.Send(TelemetryMessage.CreateException(e));
                        if (e.IsMarked())
                        {
                            ch.Error(e.Sensitivity(), e.Message);
                            PrintExceptionData(ch, e, false);
                            count++;
                        }
                    }

                    if (count == 0)
                    {
                        // Didn't recognize any of the exceptions.
                        ch.Error(MessageSensitivity.None, "***** Unexpected failure. Please contact 'tlcsupp' with details *****");
                        if (isDumpSaved)
                        {
                            ch.Error(MessageSensitivity.None, "***** Error log has been saved to '{0}', please send this file to 'tlcsupp' *****",
                                dumpFilePath);
                        }
                    }
                    else if (isDumpSaved)
                    {
                        ch.Error(MessageSensitivity.None, "Error log has been saved to '{0}'. Please send this file to 'tlcsupp' if you need assistance.",
                            dumpFilePath);
                    }

                    if (count == 0 || alwaysPrintStacktrace)
                    {
                        ch.Error(MessageSensitivity.None, "===== Begin detailed dump =====");
                        PrintFullExceptionDetails(ch, ex);
                        ch.Error(MessageSensitivity.None, "====== End detailed dump =====");
                    }

                    // Return a negative result code so AEther recognizes this as a failure.
                    result = count > 0 ? -1 : -2;
                }
                finally
                {
                }
                telemetryPipe.Done();
                return result;
            }
        }

        private static void TrackProgress(TlcEnvironment env, CancellationToken ct)
        {
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // Print a dot every 0.6s, which will make 50 dots take 30 seconds.
                    // REVIEW: maybe an adaptive interval that would expand if nothing happens is a better idea.
                    TimeSpan interval = TimeSpan.FromSeconds(0.6);
                    if (ct.WaitHandle.WaitOne(interval))
                    {
                        // Cancellation was requested.
                        return;
                    }
                    env.PrintProgress();
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("Progress tracking terminated with an exception");
                PrintExceptionData(Console.Error, ex, false);
                Console.Error.WriteLine("Progress tracking is terminated.");
            }
        }

        /// <summary>
        /// Prints exception type, message, stack trace and data for every exception in the
        /// <see cref="Exception.InnerException"/> chain.
        /// </summary>
        private static void PrintFullExceptionDetails(TextWriter writer, Exception ex)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValue(ex);

            int index = 0;
            // REVIEW: The innermost exception is almost always the most relevant.
            // That it is printed last is a bit askew.
            for (var e = ex; e != null; e = e.InnerException)
            {
                index++;
                writer.WriteLine("({0}) Unexpected exception: {1}, '{2}'", index, e.Message, e.GetType());
                PrintExceptionData(writer, e, true);
                writer.WriteLine(e.StackTrace);
            }
        }

        /// <summary>
        /// Prints exception type, message, stack trace and data for every exception in the
        /// <see cref="Exception.InnerException"/> chain.
        /// </summary>
        private static void PrintFullExceptionDetails(IChannel ch, Exception ex)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(ex);
            int index = 0;
            for (var e = ex; e != null; e = e.InnerException)
            {
                index++;
                ch.Error(e.Sensitivity(), "({0}) Unexpected exception: {1}, '{2}'", index, e.Message, e.GetType());
                PrintExceptionData(ch, e, true);
                // While the message can be sensitive, we suppose the stack trace itself is not.
                ch.Error(MessageSensitivity.None, e.StackTrace);
            }
        }

        private static void PrintExceptionData(TextWriter writer, Exception ex, bool includeComponents)
        {
            bool anyDataPrinted = false;
            foreach (DictionaryEntry kvp in ex.Data)
            {
                if (Contracts.IsMarkedKey.Equals(kvp.Key))
                    continue;
                if (Contracts.SensitivityKey.Equals(kvp.Key))
                    continue;
                if (!anyDataPrinted)
                {
                    writer.WriteLine();
                    writer.WriteLine("Exception context:");
                }

                if (TlcEnvironment.ComponentHistoryKey.Equals(kvp.Key))
                {
                    if (kvp.Value is string[] createdComponents)
                    {
                        if (!includeComponents)
                            continue;

                        writer.WriteLine("    Created components:");
                        foreach (var name in createdComponents)
                            writer.WriteLine("        {0}", name);

                        anyDataPrinted = true;
                        continue;
                    }
                }

                writer.WriteLine("    {0}: {1}", kvp.Key, kvp.Value);
                anyDataPrinted = true;
            }

            if (anyDataPrinted)
                writer.WriteLine();
        }

        private static void PrintExceptionData(IChannel ch, Exception ex, bool includeComponents)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(ex);

            var sb = new StringBuilder();
            using (var sw = new StringWriter(sb, CultureInfo.InvariantCulture))
                PrintExceptionData(sw, ex, includeComponents);

            if (sb.Length > 0)
                ch.Error(ex.Sensitivity(), sb.ToString());
        }

        private static void Usage()
        {
            Console.WriteLine("Usage: maml <cmd> <args>");
            Console.WriteLine("       To get a list of commands: maml ?");
        }
    }
}
