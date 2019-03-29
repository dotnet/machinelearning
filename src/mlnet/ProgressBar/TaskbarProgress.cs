// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public static class TaskbarProgress
    {
        public enum TaskbarStates
        {
            NoProgress = 0,
            Indeterminate = 0x1,
            Normal = 0x2,
            Error = 0x4,
            Paused = 0x8
        }

        [ComImport()]
        [Guid("ea1afb91-9e28-4b86-90e9-9e9f8a5eefaf")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface ITaskbarList3
        {
            // ITaskbarList
            [PreserveSig]
            void HrInit();

            [PreserveSig]
            void AddTab(IntPtr hwnd);

            [PreserveSig]
            void DeleteTab(IntPtr hwnd);

            [PreserveSig]
            void ActivateTab(IntPtr hwnd);

            [PreserveSig]
            void SetActiveAlt(IntPtr hwnd);

            // ITaskbarList2
            [PreserveSig]
            void MarkFullscreenWindow(IntPtr hwnd, [MarshalAs(UnmanagedType.Bool)] bool fFullscreen);

            // ITaskbarList3
            [PreserveSig]
            void SetProgressValue(IntPtr hwnd, UInt64 ullCompleted, UInt64 ullTotal);

            [PreserveSig]
            void SetProgressState(IntPtr hwnd, TaskbarStates state);
        }

        [ComImport]
        [Guid("56fdf344-fd6d-11d0-958a-006097c9a090")]
        [ClassInterface(ClassInterfaceType.None)]
        private class TaskbarInstance
        { }

        [DllImport("kernel32.dll")]
        static extern IntPtr GetConsoleWindow();

        private static readonly ITaskbarList3 _taskbarInstance = (ITaskbarList3)new TaskbarInstance();
        private static readonly bool _taskbarSupported = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        public static void SetState(TaskbarStates taskbarState)
        {
            if (_taskbarSupported)
                _taskbarInstance.SetProgressState(GetConsoleWindow(), taskbarState);
        }

        public static void SetValue(double progressValue, double progressMax)
        {
            if (_taskbarSupported)
                _taskbarInstance.SetProgressValue(GetConsoleWindow(), (ulong)progressValue, (ulong)progressMax);
        }
    }
}
