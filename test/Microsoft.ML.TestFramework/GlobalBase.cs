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
