// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 649 // field is never assigned

using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class CmdLine : BaseTestBaseline
    {
        private const string ResourcePrefix = "Microsoft.ML.Runtime.RunTests.CmdLine.";

        public CmdLine(ITestOutputHelper helper)
            : base(helper)
        {
        }

        /// <summary>
        /// Basic command line parsing test - broad (but not total) coverage
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Cmd Parsing")]
        public void CmdParsingBasic()
        {
            var defaults = new ArgsBasic();
            Run("CmdLine", "BasicParsing", defaults.CallInit, defaults.CallProcess);
        }

        /// <summary>
        /// Command line parsing of all numeric types, char, and string.
        /// Note that bool and enums are not covered (yet).
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Cmd Parsing")]
        public void CmdParsingSingle()
        {
            ArgsBase[] defaults = new ArgsBase[]
                {
                    new ArgsSingle<sbyte>(), new ArgsSingle<sbyte>() { value = 3 },
                    new ArgsSingle<short>(), new ArgsSingle<short>() { value = 3 },
                    new ArgsSingle<int>(), new ArgsSingle<int>() { value = 3 },
                    new ArgsSingle<long>(), new ArgsSingle<long>() { value = 3 },

                    new ArgsSingle<byte>(), new ArgsSingle<byte>() { value = 3 },
                    new ArgsSingle<ushort>(), new ArgsSingle<ushort>() { value = 3 },
                    new ArgsSingle<uint>(), new ArgsSingle<uint>() { value = 3 },
                    new ArgsSingle<ulong>(), new ArgsSingle<ulong>() { value = 3 },

                    new ArgsSingle<float>(), new ArgsSingle<float>() { value = 3 },
                    new ArgsSingle<double>(), new ArgsSingle<double>() { value = 3 },
                    new ArgsSingle<decimal>(), new ArgsSingle<decimal>() { value = 3 },
                    new ArgsSingle<char>(), new ArgsSingle<char>() { value = '3' },

                    new ArgsSingle<sbyte?>(), new ArgsSingle<sbyte?>() { value = 3 },
                    new ArgsSingle<short?>(), new ArgsSingle<short?>() { value = 3 },
                    new ArgsSingle<int?>(), new ArgsSingle<int?>() { value = 3 },
                    new ArgsSingle<long?>(), new ArgsSingle<long?>() { value = 3 },

                    new ArgsSingle<byte?>(), new ArgsSingle<byte?>() { value = 3 },
                    new ArgsSingle<ushort?>(), new ArgsSingle<ushort?>() { value = 3 },
                    new ArgsSingle<uint?>(), new ArgsSingle<uint?>() { value = 3 },
                    new ArgsSingle<ulong?>(), new ArgsSingle<ulong?>() { value = 3 },

                    new ArgsSingle<float?>(), new ArgsSingle<float?>() { value = 3 },
                    new ArgsSingle<double?>(), new ArgsSingle<double?>() { value = 3 },
                    new ArgsSingle<decimal?>(), new ArgsSingle<decimal?>() { value = 3 },
                    new ArgsSingle<char?>(), new ArgsSingle<char?>() { value = '3' },

                    new ArgsSingle<string>(), new ArgsSingle<string>() { value = "3" },
                };

            Action<IndentedTextWriter> init = null;
            Action<IndentedTextWriter, string> action = null;
            foreach (var def in defaults)
            {
                init += def.CallInit;
                action += def.CallProcess;
            }

            Run("CmdLine", "SingleParsing", init, action);
        }

        /// <summary>
        /// Called at the beginning of a test - it dumps the usage of the Arguments class(es).
        /// </summary>
        private static void Init(IndentedTextWriter wrt, object defaults)
        {
            var env = new ConsoleEnvironment(seed: 42);
            wrt.WriteLine("Usage:");
            wrt.WriteLine(CmdParser.ArgumentsUsage(env, defaults.GetType(), defaults, false, 200));
        }

        /// <summary>
        /// Process a script to be parsed (from the input resource).
        /// </summary>
        private static void Process(IndentedTextWriter wrt, string text, ArgsBase defaults)
        {
            var env = new ConsoleEnvironment(seed: 42);
            using (wrt.Nest())
            {
                var args1 = defaults.Clone();
                using (wrt.Nest())
                {
                    if (!CmdParser.ParseArguments(env, text, args1, s => wrt.WriteLine("*** {0}", s)))
                        wrt.WriteLine("*** Failed!");
                }
                string str1 = args1.ToString();
                wrt.WriteLine("ToString: {0}", str1);
                string settings1 = CmdParser.GetSettings(env, args1, defaults, SettingsFlags.None);
                wrt.WriteLine("Settings: {0}", settings1);

                var args2 = defaults.Clone();
                using (wrt.Nest())
                {
                    if (!CmdParser.ParseArguments(env, settings1, args2, s => wrt.WriteLine("*** BUG: {0}", s)))
                        wrt.WriteLine("*** BUG: parsing result of GetSettings failed!");
                }
                string str2 = args2.ToString();
                if (str1 != str2)
                    wrt.WriteLine("*** BUG: ToString Mismatch: {0}", str2);
                string settings2 = CmdParser.GetSettings(env, args2, defaults, SettingsFlags.None);
                if (settings1 != settings2)
                    wrt.WriteLine("*** BUG: Settings Mismatch: {0}", settings2);
            }
        }

        private abstract class ArgsBase
        {
            public void CallInit(IndentedTextWriter wrt)
            {
                Init(wrt, this);
            }

            /// <summary>
            /// Call the Process method passing "this" as the defaults.
            /// </summary>
            public void CallProcess(IndentedTextWriter wrt, string text)
            {
                Process(wrt, text, this);
            }

            public ArgsBase Clone()
            {
                return (ArgsBase)MemberwiseClone();
            }
        }

        private class ArgsBasic : ArgsBase
        {
            public class Nested
            {
                [Argument(ArgumentType.AtMostOnce)]
                public int a = 1;

                [Argument(ArgumentType.AtMostOnce)]
                public int b;

                public override string ToString()
                {
                    return string.Format("{{a={0} b={1}}}", a, b);
                }
            }

            [Argument(ArgumentType.Required, ShortName = "r,r2")]
            public int required = -1;

            [Argument(ArgumentType.AtMostOnce)]
            public int once = 1;

            [Argument(ArgumentType.LastOccurenceWins)]
            public int last = 2;

            [Argument(ArgumentType.Multiple)]
            public int[] multi;

            [Argument(ArgumentType.Multiple)]
            public KeyValuePair<string, int>[] tagMulti;

            [Argument(ArgumentType.MultipleUnique)]
            public int[] unique;

            [Argument(ArgumentType.AtMostOnce)]
            public string text = "";

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> sub = (IComponentFactory<IDataSaver>)CmdParser.CreateComponentFactory(
                typeof(IComponentFactory<IDataSaver>),
                typeof(SignatureDataSaver), 
                "Text");

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver>[] subArray = null;

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureDataSaver))]
            public KeyValuePair<string, IComponentFactory<IDataSaver>>[] subTaggedArray = null;

            [Argument(ArgumentType.Multiple)]
            public Nested[] nest;

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("required=").Append(required);
                sb.Append(" once=").Append(once);
                sb.Append(" last=").Append(last);
                sb.Append(" multi=");
                AppendArray(sb, multi);
                sb.Append(" tagMulti=");
                AppendTaggedArray(sb, tagMulti);
                sb.Append(" unique=");
                AppendArray(sb, unique);
                sb.AppendFormat(" text='{0}'", text);
                sb.Append(" sub=").Append(sub);
                AppendSubArray(sb, subArray);
                AppendSubTaggedArray(sb, subTaggedArray);
                if (Utils.Size(nest) > 0)
                {
                    sb.Append(" nest=");
                    AppendArray(sb, nest);
                }
                return sb.ToString();
            }

            private static void AppendSubArray(StringBuilder sb, IComponentFactory[] subComponents)
            {
                if (subComponents == null)
                    return;
                foreach (var sc in subComponents.Cast<ICommandLineComponentFactory>())
                    sb.AppendFormat(" subArray={0}{1}", sc.Name, sc.GetSettingsString());
            }

            private static void AppendSubTaggedArray<T>(StringBuilder sb, KeyValuePair<string, IComponentFactory<T>>[] pairs)
            {
                if (pairs == null)
                    return;
                foreach (var pair in pairs)
                {
                    var value = (ICommandLineComponentFactory)pair.Value;
                    sb.AppendFormat(" subTaggedArray{1}={0}{2}", value.Name, pair.Key == "" ? "" : "[" + pair.Key + "]", value.GetSettingsString());
                }
            }

            private static void AppendArray<T>(StringBuilder sb, T[] arr)
            {
                sb.Append("{");
                if (arr != null && arr.Length > 0)
                {
                    string sep = "";
                    foreach (T x in arr)
                    {
                        sb.Append(sep);
                        sep = ", ";
                        sb.Append(x);
                    }
                }
                sb.Append("}");
            }

            private static void AppendTaggedArray<T>(StringBuilder sb, KeyValuePair<string, T>[] arr)
            {
                sb.Append("{");
                if (arr != null && arr.Length > 0)
                {
                    string sep = "";
                    foreach (KeyValuePair<string, T> pair in arr)
                    {
                        sb.Append(sep);
                        sep = ", ";
                        if (pair.Key != "")
                            sb.AppendFormat("[{0}]", pair.Key);
                        sb.Append(pair.Value);
                    }
                }
                sb.Append("}");
            }
        }

        private class ArgsSingle<T> : ArgsBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "val")]
            public T value;

            public override string ToString()
            {
                return string.Format("{0}: value={1}", TypeName, value);
            }

            public string TypeName
            {
                get
                {
                    var type = typeof(T);
                    if (type.IsConstructedGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                        return type.GetGenericArguments()[0].Name + "?";
                    return type.Name;
                }
            }
        }

        // Get the name of the input resource.
        private string InResName(string name)
        {
            return ResourcePrefix + name + "Input.txt";
        }

        // Get the name of the output file.
        private string OutFileName(string name)
        {
            return name + "Output.txt";
        }

        // Get the resource text from the resource name.
        private string GetResText(string resName)
        {
            var stream = typeof(CmdLine).Assembly.GetManifestResourceStream(resName);
            if (stream == null)
                return string.Format("<couldn't read {0}>", resName);

            using (var reader = new StreamReader(stream))
            {
                return reader.ReadToEnd();
            }
        }

        // Run the test. The init delegate is called once at the beginning of the test.
        // The action delegate is called on script in the input resource.
        private void Run(string dir, string name, Action<IndentedTextWriter> init,
            Action<IndentedTextWriter, string> action)
        {
            string text = GetResText(InResName(name));

            string outName = OutFileName(name);
            string outPath = DeleteOutputPath(dir, outName);

            using (var writer = File.CreateText(outPath))
            {
                var wrt = new IndentedTextWriter(writer);

                init(wrt);

                // Individual scripts are separated by $
                int count = 0;
                int ichLim = 0;
                int lineLim = 1;
                while (ichLim < text.Length)
                {
                    int ichMin = ichLim;
                    int lineMin = lineLim;

                    while (ichLim < text.Length && text[ichLim] != '$')
                    {
                        if (text[ichLim] == '\n')
                            lineLim++;
                        ichLim++;
                    }

                    while (ichMin < ichLim && char.IsWhiteSpace(text[ichMin]))
                    {
                        if (text[ichMin] == '\n')
                            lineMin++;
                        ichMin++;
                    }

                    if (ichMin < ichLim)
                    {
                        // Process the script.
                        count++;
                        string scriptName = string.Format("Script {0}, lines {1} to {2}", count, lineMin, lineLim);
                        wrt.WriteLine("===== Start {0} =====", scriptName);
                        string input = text.Substring(ichMin, ichLim - ichMin);
                        try
                        {
                            action(wrt, input);
                        }
                        catch (Exception ex)
                        {
                            if (!ex.IsMarked())
                                wrt.WriteLine("Unknown Exception!");
                            wrt.WriteLine("Exception: {0}", ex.Message);
                        }

                        wrt.WriteLine("===== End {0} =====", scriptName);
                    }

                    // Skip the $
                    ichLim++;
                }
            }

            CheckEquality(dir, outName);

            Done();
        }
    }
}
