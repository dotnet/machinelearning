// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Every unit test assembly should call GlobalBase.AssemblyInit() before running any tests.
// Test assembly should have following test also. 
//    
//    public void AssertHandlerTest()
//    { 
//        GlobalBase.AssertHandlerTest();
//    }

using System;
using System.Management;
using System.Runtime.InteropServices;
using Xunit;

namespace Microsoft.ML.Runtime.Internal.Internallearn.Test
{
    internal static class GlobalBase
    {
        public static void AssemblyInit()
        {
            System.Diagnostics.Debug.WriteLine("*** Setting test assertion handler");
            var prev = Contracts.SetAssertHandler(AssertHandler);
            Contracts.Check(prev == null, "Expected to replace null assertion handler!");

            // HACK: ensure MklImports is loaded very early in the tests so it doesn't deadlock while loading it later.
            // See https://github.com/dotnet/machinelearning/issues/1073
            Mkl.PptrfInternal(Mkl.Layout.RowMajor, Mkl.UpLo.Up, 0, Array.Empty<double>());

            PrintCpuInfo();
        }

        private static void PrintCpuInfo()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                ManagementClass clsMgtClass = new ManagementClass("Win32_Processor");
                ManagementObjectCollection colMgtObjCol = clsMgtClass.GetInstances();

                foreach (ManagementObject objMgtObj in colMgtObjCol)
                {
                    foreach (var property in objMgtObj.Properties)
                    {
                        Console.WriteLine("CPU Property " + property.Name + " = " + property.Value);

                    }
                }

                double totalCapacity = 0;
                ObjectQuery objectQuery = new ObjectQuery("select * from Win32_PhysicalMemory");
                ManagementObjectSearcher searcher = new ManagementObjectSearcher(objectQuery);
                ManagementObjectCollection vals = searcher.Get();

                foreach (ManagementObject val in vals)
                {
                    totalCapacity += System.Convert.ToDouble(val.GetPropertyValue("Capacity"));
                }

                Console.WriteLine("Total Machine Memory = " + totalCapacity.ToString() + " Bytes");
                Console.WriteLine("Total Machine Memory = " + (totalCapacity / 1024) + " KiloBytes");
                Console.WriteLine("Total Machine Memory = " + (totalCapacity / 1048576) + "    MegaBytes");
                Console.WriteLine("Total Machine Memory = " + (totalCapacity / 1073741824) + " GigaBytes");
            }
        }

        private static class Mkl
        {
            public enum Layout
            {
                RowMajor = 101,
                ColMajor = 102
            }

            public enum UpLo : byte
            {
                Up = (byte)'U',
                Lo = (byte)'L'
            }

            [DllImport("MklImports", EntryPoint = "LAPACKE_dpptrf")]
            public static extern int PptrfInternal(Layout layout, UpLo uplo, int n, double[] ap);
        }

        public static void AssemblyCleanup()
        {
            System.Diagnostics.Debug.WriteLine("*** Restoring test assertion handler");
            var prev = Contracts.SetAssertHandler(null);
            Contracts.Check(prev == AssertHandler, "Expected to replace Global.AssertHandler!");
        }

#if DEBUG
        private static bool _ignoreOne;
#endif
        private static void AssertHandler(string msg, IExceptionContext ectx)
        {
#if DEBUG
            if (_ignoreOne)
            {
                _ignoreOne = false;
                Console.WriteLine("Assert handler invoked but ignored: {0}", msg);
            }
            else
#endif
                Assert.True(false, $"Assert failed: {msg}");
        }

        public static void AssertHandlerTest()
        {
#if DEBUG
            Assert.False(_ignoreOne);
            _ignoreOne = true;
            try
            {
                Contracts.Assert(false, "This should invoke the handler");
                Assert.False(_ignoreOne);
            }
            finally
            {
                _ignoreOne = false;
            }
#endif
        }
    }
}
