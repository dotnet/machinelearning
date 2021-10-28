// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RemoteExecutorConsoleApp
{
    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 4)
            {
                Console.Error.WriteLine("Usage: {0} assemblyName typeName methodName exceptionFile [additionalArgs]", typeof(Program).GetTypeInfo().Assembly.GetName().Name);
                Environment.Exit(-1);
                return -1;
            }

            string assemblyName = args[0];
            string typeName = args[1];
            string methodName = args[2];
            string exceptionFile = args[3];
            string[] additionalArgs = args.Length > 4 ?
                Subarray(args, 4, args.Length - 4) :
                Array.Empty<string>();

            // Load the specified assembly, type, and method, then invoke the method.
            // The program's exit code is the return value of the invoked method.
            Assembly a = null;
            Type t = null;
            MethodInfo mi = null;
            object instance = null;
            int exitCode = 0;

            try
            {
                // Create the test class if necessary
                try
                {
                    a = Assembly.Load(assemblyName);
                }
                catch (FileNotFoundException)
                {
                    a = Assembly.LoadFrom(assemblyName.Split(',')[0] + ".dll");
                }

                t = a.GetType(typeName);
                mi = t.GetTypeInfo().GetDeclaredMethod(methodName);
                if (!mi.IsStatic)
                {
                    instance = Activator.CreateInstance(t);
                }

                // Invoke the test
                object result = mi.Invoke(instance, additionalArgs);

                if (result is Task<int> task)
                {
                    exitCode = task.GetAwaiter().GetResult();
                }
                else if (result is int exit)
                {
                    exitCode = exit;
                }
            }
            catch (Exception exc)
            {
                if (exc is TargetInvocationException && exc.InnerException != null)
                    exc = exc.InnerException;

                var output = new StringBuilder();
                output.AppendLine();
                output.AppendLine("Child exception:");
                output.AppendLine("  " + exc);
                output.AppendLine();
                output.AppendLine("Child process:");
                output.AppendLine(string.Format("  {0} {1} {2}", a, t, mi));
                output.AppendLine();

                if (additionalArgs.Length > 0)
                {
                    output.AppendLine("Child arguments:");
                    output.AppendLine("  " + string.Join(", ", additionalArgs));
                }

                File.WriteAllText(exceptionFile, output.ToString());

                ExceptionDispatchInfo.Capture(exc).Throw();
            }
            finally
            {
                (instance as IDisposable)?.Dispose();
            }

            // Use Exit rather than simply returning the exit code so that we forcibly shut down
            // the process even if there are foreground threads created by the operation that would
            // end up keeping the process alive potentially indefinitely.
            try
            {
                Environment.Exit(exitCode);
            }
            catch (PlatformNotSupportedException)
            {
            }

            return exitCode;
        }

        private static T[] Subarray<T>(T[] arr, int offset, int count)
        {
            var newArr = new T[count];
            Array.Copy(arr, offset, newArr, 0, count);
            return newArr;
        }
    }
}
