// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace Microsoft.DotNet.Cli.Utils
{
    internal static class NativeMethods
    {
        internal static class Windows
        {
            internal enum JobObjectInfoClass : uint
            {
                JobObjectExtendedLimitInformation = 9,
            }

            [Flags]
            internal enum JobObjectLimitFlags : uint
            {
                JobObjectLimitKillOnJobClose = 0x2000,
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct JobObjectBasicLimitInformation
            {
                public Int64 PerProcessUserTimeLimit;
                public Int64 PerJobUserTimeLimit;
                public JobObjectLimitFlags LimitFlags;
                public UIntPtr MinimumWorkingSetSize;
                public UIntPtr MaximumWorkingSetSize;
                public UInt32 ActiveProcessLimit;
                public UIntPtr Affinity;
                public UInt32 PriorityClass;
                public UInt32 SchedulingClass;
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct IoCounters
            {
                public UInt64 ReadOperationCount;
                public UInt64 WriteOperationCount;
                public UInt64 OtherOperationCount;
                public UInt64 ReadTransferCount;
                public UInt64 WriteTransferCount;
                public UInt64 OtherTransferCount;
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct JobObjectExtendedLimitInformation
            {
                public JobObjectBasicLimitInformation BasicLimitInformation;
                public IoCounters IoInfo;
                public UIntPtr ProcessMemoryLimit;
                public UIntPtr JobMemoryLimit;
                public UIntPtr PeakProcessMemoryUsed;
                public UIntPtr PeakJobMemoryUsed;
            }

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            internal static extern SafeWaitHandle CreateJobObjectW(IntPtr lpJobAttributes, string lpName);

            [DllImport("kernel32.dll", SetLastError = true)]
            internal static extern bool SetInformationJobObject(IntPtr hJob, JobObjectInfoClass jobObjectInformationClass, IntPtr lpJobObjectInformation, UInt32 cbJobObjectInformationLength);

            [DllImport("kernel32.dll", SetLastError = true)]
            internal static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
        }

        internal static class Posix
        {
            [DllImport("libc", SetLastError = true)]
            internal static extern int kill(int pid, int sig);

            internal const int SigInt = 2;
            internal const int SigTerm = 15;
        }
    }
}
