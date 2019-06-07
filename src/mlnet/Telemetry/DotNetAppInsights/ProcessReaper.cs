using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.DotNet.PlatformAbstractions;
using Microsoft.Win32.SafeHandles;

using RuntimeEnvironment = Microsoft.DotNet.PlatformAbstractions.RuntimeEnvironment;

namespace Microsoft.DotNet.Cli.Utils
{
    /// <summary>
    /// Responsible for reaping a target process if the current process terminates.
    /// </summary>
    /// <remarks>
    /// On Windows, a job object will be used to ensure the termination of the target
    /// process (and its tree) even if the current process is rudely terminated.
    ///
    /// On POSIX systems, the reaper will handle SIGTERM and attempt to forward the
    /// signal to the target process only.
    ///
    /// The reaper also suppresses SIGINT in the current process to allow the target
    /// process to handle the signal.
    /// </remarks>
    internal class ProcessReaper : IDisposable
    {
        /// <summary>
        /// Creates a new process reaper.
        /// </summary>
        /// <param name="process">The target process to reap if the current process terminates. The process should not yet be started.</param>
        public ProcessReaper(Process process)
        {
            _process = process;

            // The tests need the event handlers registered prior to spawning the child to prevent a race
            // where the child writes output the test expects before the intermediate dotnet process
            // has registered the event handlers to handle the signals the tests will generate.
            Console.CancelKeyPress += HandleCancelKeyPress;
            if (RuntimeEnvironment.OperatingSystemPlatform != Platform.Windows)
            {
                _shutdownMutex = new Mutex();
                AppDomain.CurrentDomain.ProcessExit += HandleProcessExit;
            }
        }

        /// <summary>
        /// Call to notify the reaper that the process has started.
        /// </summary>
        public void NotifyProcessStarted()
        {
            if (RuntimeEnvironment.OperatingSystemPlatform == Platform.Windows)
            {
                // Limit the use of job objects to versions of Windows that support nested jobs (i.e. Windows 8/2012 or later).
                // Ideally, we would check for some new API export or OS feature instead of the OS version,
                // but nested jobs are transparently implemented with respect to the Job Objects API.
                // Note: Windows 8.1 and later may report as Windows 8 (see https://docs.microsoft.com/en-us/windows/desktop/sysinfo/operating-system-version).
                //       However, for the purpose of this check that is still sufficient.
                if (Environment.OSVersion.Version.Major > 6 ||
                   (Environment.OSVersion.Version.Major == 6 && Environment.OSVersion.Version.Minor >= 2))
                {
                    _job = AssignProcessToJobObject(_process.Handle);
                }
            }
        }

        public void Dispose()
        {
            if (RuntimeEnvironment.OperatingSystemPlatform == Platform.Windows)
            {
                if (_job != null)
                {
                    // Clear the kill on close flag because the child process terminated successfully
                    // If this fails, then we have no choice but to terminate any remaining processes in the job
                    SetKillOnJobClose(_job.DangerousGetHandle(), false);

                    _job.Dispose();
                    _job = null;
                }
            }
            else
            {
                AppDomain.CurrentDomain.ProcessExit -= HandleProcessExit;

                // If there's been a shutdown via the process exit handler,
                // this will block the current thread so we don't race with the CLR shutdown
                // from the signal handler.
                if (_shutdownMutex != null)
                {
                    _shutdownMutex.WaitOne();
                    _shutdownMutex.ReleaseMutex();
                    _shutdownMutex.Dispose();
                    _shutdownMutex = null;
                }
            }

            Console.CancelKeyPress -= HandleCancelKeyPress;
        }

        private static void HandleCancelKeyPress(object sender, ConsoleCancelEventArgs e)
        {
            // Ignore SIGINT/SIGQUIT so that the process can handle the signal
            e.Cancel = true;
        }

        private static SafeWaitHandle AssignProcessToJobObject(IntPtr process)
        {
            var job = NativeMethods.Windows.CreateJobObjectW(IntPtr.Zero, null);
            if (job == null || job.IsInvalid)
            {
                return null;
            }

            if (!SetKillOnJobClose(job.DangerousGetHandle(), true))
            {
                job.Dispose();
                return null;
            }

            if (!NativeMethods.Windows.AssignProcessToJobObject(job.DangerousGetHandle(), process))
            {
                job.Dispose();
                return null;
            }

            return job;
        }

        private void HandleProcessExit(object sender, EventArgs args)
        {
            int processId;
            try
            {
                processId = _process.Id;
            }
            catch (InvalidOperationException)
            {
                // The process hasn't started yet; nothing to signal
                return;
            }

            // Take ownership of the shutdown mutex; this will ensure that the other
            // thread also waiting on the process to exit won't complete CLR shutdown before
            // this one does.
            _shutdownMutex.WaitOne();

            if (!_process.WaitForExit(0) && NativeMethods.Posix.kill(processId, NativeMethods.Posix.SigTerm) != 0)
            {
                // Couldn't send the signal, don't wait
                return;
            }

            // If SIGTERM was ignored by the target, then we'll still wait
            _process.WaitForExit();

            Environment.ExitCode = _process.ExitCode;
        }

        private static bool SetKillOnJobClose(IntPtr job, bool value)
        {
            var information = new NativeMethods.Windows.JobObjectExtendedLimitInformation
            {
                BasicLimitInformation = new NativeMethods.Windows.JobObjectBasicLimitInformation
                {
                    LimitFlags = (value ? NativeMethods.Windows.JobObjectLimitFlags.JobObjectLimitKillOnJobClose : 0)
                }
            };

            var length = Marshal.SizeOf(typeof(NativeMethods.Windows.JobObjectExtendedLimitInformation));
            var informationPtr = Marshal.AllocHGlobal(length);

            try
            {
                Marshal.StructureToPtr(information, informationPtr, false);

                if (!NativeMethods.Windows.SetInformationJobObject(
                    job,
                    NativeMethods.Windows.JobObjectInfoClass.JobObjectExtendedLimitInformation,
                    informationPtr,
                    (uint)length))
                {
                    return false;
                }

                return true;
            }
            finally
            {
                Marshal.FreeHGlobal(informationPtr);
            }
        }

        private Process _process;
        private SafeWaitHandle _job;
        private Mutex _shutdownMutex;
    }
}
