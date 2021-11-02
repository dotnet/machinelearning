// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Tests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

[assembly: LoadableClass(typeof(TestFuncs1), null, typeof(SignatureFunctionProvider), "Test Functions 1", "__test1")]

[assembly: LoadableClass(typeof(TestFuncs2), null, typeof(SignatureFunctionProvider), "Test Functions 2", "__test2")]

namespace Microsoft.ML.Tests
{
    using BL = System.Boolean;
    using I4 = System.Int32;
    using I8 = System.Int64;
    using R4 = Single;
    using R8 = Double;
    using TX = ReadOnlyMemory<char>;

    public sealed partial class ExprLanguageTests : BaseTestBaseline
    {
        private const string ResourcePrefix = "Microsoft.ML.Tests.ExpressionLanguageTests.TestData.";
        private object _sync = new object();

        public ExprLanguageTests(ITestOutputHelper output)
            : base(output)
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(TestFuncs1).Assembly);
        }

        [Fact, TestCategory("Expr Language")]
        public void ExprParse()
        {
            // Code coverage test for the parser.
            Run("ExprParse");
        }

#if !NETFRAMEWORK
        // Bug in sin(x) in .Net Framework: sin(1e+30) returns 1e+30.
        [X64Fact("sin(1e+30) gives different value on x86."), TestCategory("Expr Language")]
        public void ExprBind()
        {
            // Code coverage test for the binder.
            Run("ExprBind");
        }
#endif

        [Fact, TestCategory("Expr Language")]
        public void ExprBindEx()
        {
            // Addition code coverage test for the binder.
            Run("ExprBindEx");
        }

        [Fact, TestCategory("Expr Language")]
        public void ExprCodeGen()
        {
            // Code coverage test for code gen.
            Run("ExprCodeGen");
        }

        [Fact, TestCategory("Expr Language")]
        public void ExprEval()
        {
            // Code coverage test evaluation. Note that VS can't help us measure this one :-(.
            Run("ExprEval");
        }

        private string InResName(string name)
        {
            return ResourcePrefix + name + "Input.txt";
        }

        private string GetResText(string resName)
        {
            var stream = typeof(ExprLanguageTests).Assembly.GetManifestResourceStream(resName);
            if (stream == null)
                return string.Format("<couldn't read {0}>", resName);

            using (var reader = new StreamReader(stream))
            {
                return reader.ReadToEnd();
            }
        }

        private void Run(string name)
        {
            string outDir = "ExprParser";

            string text = GetResText(InResName(name));
            string inName = name + "Input.txt";
            string outName = name + "Output.txt";
            string outNameAssem = name + "Output.Assem.txt";
            string outPath = DeleteOutputPath(outDir, outName);
            string outPathAssem = DeleteOutputPath(outDir, outNameAssem);

            using (var wr = OpenWriter(outPath))
            {
                var wrt = new IndentedTextWriter(wr, "  ");

                // Individual scripts are separated by $.
                // Inputs start after #.
                int count = 0;
                int ichLim = 0;
                int lineLim = 1;
                for (; ichLim < text.Length; ichLim++)
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

                    if (ichMin >= ichLim)
                        continue;

                    // Process the script.
                    count++;
                    string scriptName = string.Format("Script {0}, lines {1} to {2}", count, lineMin, lineLim);
                    wrt.WriteLine("===== Start {0} =====", scriptName);
                    var types = ParseTypes(text, ref ichMin, ichLim);
                    int ichLimChars = text.IndexOf('#', ichMin);
                    if (ichLimChars < 0 || ichLimChars >= ichLim)
                        ichLimChars = ichLim;
                    else
                        Contracts.Assert(ichMin < ichLimChars && ichLimChars < ichLim);
                    CharCursor chars = new CharCursor(text, ichMin, ichLimChars);
                    Delegate del = null;
                    lock (_sync)
                    {
                        try
                        {
                            LambdaNode node;
                            List<Error> errors;
                            List<int> lineMap;
                            var perm = Utils.GetIdentityPermutation(types.Length);
                            using (wrt.Nest())
                            {
                                node = LambdaParser.Parse(out errors, out lineMap, chars, perm, types);
                            }
                            Check(node != null, "Null LambdaNode");
                            if (Utils.Size(errors) > 0)
                            {
                                DumpErrors(wrt, lineMap, lineMin, inName, "Parsing", errors);
                                goto LDone;
                            }

                            LambdaBinder.Run(Env, ref errors, node, msg => wr.WriteLine(msg));
                            if (Utils.Size(errors) > 0)
                            {
                                DumpErrors(wrt, lineMap, lineMin, inName, "Binding", errors);
                                goto LDone;
                            }
                            wrt.WriteLine("Binding succeeded. Output type: {0}", node.ResultType);

                            del = LambdaCompiler.Compile(out errors, node);
                            Contracts.Assert(TestFuncs1.Writer == null);
                            TestFuncs1.Writer = wr;
                            if (Utils.Size(errors) > 0)
                            {
                                DumpErrors(wrt, lineMap, lineMin, inName, "Compiling", errors);
                                goto LDone;
                            }

                            wrt.WriteLine("Compiling succeeded.");
                            if (ichLimChars < ichLim)
                                Evaluate(wrt, del, node.ResultType, types, text, ichLimChars + 1, ichLim);
                        }
                        catch (Exception ex)
                        {
                            if (!ex.IsMarked())
                                wrt.WriteLine("Unknown Exception: {0}!", del != null ? del.GetMethodInfo().DeclaringType : (object)"<null>");
                            wrt.WriteLine("Exception: {0}", ex.Message);
                        }
                        finally
                        {
                            TestFuncs1.Writer = null;
                        }

LDone:
                        wrt.WriteLine("===== End {0} =====", scriptName);
                    }
                }
            }

            CheckEquality(outDir, outName, digitsOfPrecision: 6);

            Done();
        }

        private DataViewType[] ParseTypes(string text, ref int ichMin, int ichLim)
        {
            int ichCol = text.IndexOf(':', ichMin);
            Contracts.Assert(ichMin < ichCol && ichCol < ichLim);
            string[] toks = text.Substring(ichMin, ichCol - ichMin).Split(',');
            var res = new DataViewType[toks.Length];
            for (int i = 0; i < toks.Length; i++)
            {
                InternalDataKind kind;
                bool tmp = Enum.TryParse(toks[i], out kind);
                Contracts.Assert(tmp);
                res[i] = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
            }
            ichMin = ichCol + 1;
            return res;
        }

        private void Evaluate(IndentedTextWriter wrt, Delegate del, DataViewType typeRes, DataViewType[] types,
            string text, int ichMin, int ichLim)
        {
            Contracts.AssertValue(del);
            Contracts.AssertNonEmpty(types);
            var args = new object[types.Length];
            var getters = new Func<ReadOnlyMemory<char>, bool>[types.Length];
            for (int i = 0; i < getters.Length; i++)
                getters[i] = GetGetter(i, types[i], args);

            StringBuilder sb = new StringBuilder();
            Action<object> printer = GetPrinter(typeRes, sb);

            ReadOnlyMemory<char> chars = text.AsMemory().Slice(ichMin, ichLim - ichMin);
            for (bool more = true; more;)
            {
                ReadOnlyMemory<char> line;
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    more = ReadOnlyMemoryUtils.SplitOne(chars, '\x0D', out line, out chars);
                else
                    more = ReadOnlyMemoryUtils.SplitOne(chars, '\x0A', out line, out chars);
                line = ReadOnlyMemoryUtils.TrimWhiteSpace(line);
                if (line.IsEmpty)
                    continue;

                // Note this "hack" to map _ to empty. It's easier than fully handling quoting and is sufficient
                // for these tests.
                var vals = ReadOnlyMemoryUtils.Split(line, new char[] { ',' })
                        .Select(x => ReadOnlyMemoryUtils.TrimWhiteSpace(x))
                        .Select(x => ReadOnlyMemoryUtils.EqualsStr("_", x) ? ReadOnlyMemory<char>.Empty : x)
                        .ToArray();

                Contracts.Assert(vals.Length == getters.Length);
                for (int i = 0; i < getters.Length; i++)
                {
                    if (!getters[i](vals[i]))
                        wrt.Write("*** Parsing {0} Failed *** ", vals[i]);
                }
                var res = del.DynamicInvoke(args);
                printer(res);
                wrt.WriteLine(sb);
            }
        }

        private Func<ReadOnlyMemory<char>, bool> GetGetter(int i, DataViewType dst, object[] args)
        {
            switch (dst.GetRawKind())
            {
                case InternalDataKind.BL:
                    return
                        src =>
                        {
                            bool v;
                            bool tmp = Conversions.DefaultInstance.TryParse(in src, out v);
                            args[i] = v;
                            return tmp;
                        };
                case InternalDataKind.I4:
                    return
                        src =>
                        {
                            int v;
                            bool tmp = Conversions.DefaultInstance.TryParse(in src, out v);
                            args[i] = v;
                            return tmp;
                        };
                case InternalDataKind.I8:
                    return
                        src =>
                        {
                            long v;
                            bool tmp = Conversions.DefaultInstance.TryParse(in src, out v);
                            args[i] = v;
                            return tmp;
                        };
                case InternalDataKind.R4:
                    return
                        src =>
                        {
                            float v;
                            bool tmp = Conversions.DefaultInstance.TryParse(in src, out v);
                            args[i] = v;
                            return tmp;
                        };
                case InternalDataKind.R8:
                    return
                        src =>
                        {
                            double v;
                            bool tmp = Conversions.DefaultInstance.TryParse(in src, out v);
                            args[i] = v;
                            return tmp;
                        };
                case InternalDataKind.TX:
                    return
                        src =>
                        {
                            args[i] = src;
                            return true;
                        };
            }

            Contracts.Assert(false);
            return null;
        }

        private Action<object> GetPrinter(DataViewType dst, StringBuilder sb)
        {
            switch (dst.GetRawKind())
            {
                case InternalDataKind.BL:
                    return
                        src =>
                        {
                            var v = (bool)src;
                            Conversions.DefaultInstance.Convert(in v, ref sb);
                        };
                case InternalDataKind.I4:
                    return
                        src =>
                        {
                            var v = (int)src;
                            Conversions.DefaultInstance.Convert(in v, ref sb);
                        };
                case InternalDataKind.I8:
                    return
                        src =>
                        {
                            var v = (long)src;
                            Conversions.DefaultInstance.Convert(in v, ref sb);
                        };
                case InternalDataKind.R4:
                    return
                        src =>
                        {
                            var v = (Single)src;
                            Conversions.DefaultInstance.Convert(in v, ref sb);
                        };
                case InternalDataKind.R8:
                    return
                        src =>
                        {
                            var v = (Double)src;
                            Conversions.DefaultInstance.Convert(in v, ref sb);
                        };
                case InternalDataKind.TX:
                    return
                        src =>
                        {
                            var v = (ReadOnlyMemory<char>)src;
                            TextSaverUtils.MapText(v.Span, ref sb, '\t');
                        };
            }

            Contracts.Assert(false);
            return null;
        }

        private void DumpErrors(IndentedTextWriter wrt, List<int> lineMap, int lineMin,
            string fileName, string phase, List<Error> errors)
        {
            Contracts.AssertValue(wrt);
            Contracts.AssertValue(lineMap);
            Contracts.AssertNonEmpty(phase);
            Contracts.AssertNonEmpty(errors);

            using (wrt.Nest())
            {
                foreach (var err in errors)
                {
                    var tok = err.Token;
                    Contracts.AssertValue(tok);
                    var pos = new LambdaParser.SourcePos(lineMap, tok.Span, lineMin);
                    wrt.Write("{0}({1},{2})-({3},{4}): ",
                        fileName, pos.LineMin, pos.ColumnMin, pos.LineLim, pos.ColumnLim);
                    wrt.Write("error: ");
                    wrt.WriteLine(err.GetMessage());
                }
            }
        }
    }

    public sealed class TestFuncs1 : IFunctionProvider
    {
        // REVIEW: This is a temporary hack to baseline test the _dump functions. 
        // Should probably figure out a proper way to do this.
        internal static TextWriter Writer;

        private static volatile TestFuncs1 _instance;

        public static TestFuncs1 Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new TestFuncs1(), null);
                return _instance;
            }
        }

        private static TextWriter OutWriter { get { return Writer ?? Console.Out; } }

        public string NameSpace { get { return "__test1"; } }

        public MethodInfo[] Lookup(string name)
        {
            switch (name)
            {
                // This one should be ambigous when invoked on an I4.
                case "_aa":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I8>(A),
                        FunctionProviderUtils.Fn<R4, R4>(A));
                case "_ab":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I8>(A));
                case "_ac":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I8>(A));
                case "_ad":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I8>(A));

                case "_var":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, BL, R4[], R4>(Var));

                case "_ba":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4>(B),
                        FunctionProviderUtils.Fn<I4, I4>(B),
                        FunctionProviderUtils.Fn<I4, I4, I4>(B));

                case "_bad":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<object>(X),
                        FunctionProviderUtils.Fn<string, I4>(X),
                        FunctionProviderUtils.Fn<I4, I4>(X),
                        ((Func<I8, I8>)(X)).GetMethodInfo(),
                        FunctionProviderUtils.Fn<R4, R4>(X));

                case "_fa":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<BL, BL>(F));

                case "_dump":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4>(Dump),
                        FunctionProviderUtils.Fn<I8, I8>(Dump),
                        FunctionProviderUtils.Fn<R4, R4>(Dump),
                        FunctionProviderUtils.Fn<R8, R8>(Dump),
                        FunctionProviderUtils.Fn<BL, BL>(Dump),
                        FunctionProviderUtils.Fn<TX, TX>(Dump),
                        FunctionProviderUtils.Fn<TX, I4, I4>(Dump),
                        FunctionProviderUtils.Fn<TX, I8, I8>(Dump),
                        FunctionProviderUtils.Fn<TX, R4, R4>(Dump),
                        FunctionProviderUtils.Fn<TX, R8, R8>(Dump),
                        FunctionProviderUtils.Fn<TX, BL, BL>(Dump),
                        FunctionProviderUtils.Fn<TX, TX, TX>(Dump));

                case "_chars":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, TX>(DumpChars));
            }

            return null;
        }

        public object ResolveToConstant(string name, MethodInfo meth, object[] values)
        {
            switch (name)
            {
                case "_bad":
                    // Note this is intentionally wrong (should return an I4, not int), to test
                    // handling of buggy implementations of IExprFunctions.
                    return 3;
            }

            return null;
        }

        public static I8 A(I8 a)
        {
            return a * 2;
        }

        public static R4 A(R4 a)
        {
            return -a;
        }

        /// <summary>
        /// For testing variable-arg functions. This selects the element in c indicated by a.
        /// If b is true, it negates the result.
        /// </summary>
        public static R4 Var(I4 a, BL b, R4[] c)
        {
            if (a < 0 || a >= c.Length)
                return R4.NaN;
            R4 res = c[a];
            if (b)
                res = -res;
            return res;
        }

        public static I4 B()
        {
            return 1;
        }

        public static I4 B(I4 a)
        {
            return 2;
        }

        public static I4 B(I4 a, I4 b)
        {
            return 3;
        }

        public static object X()
        {
            return null;
        }

        public static I4 X(string a)
        {
            return 41;
        }

        public static I4 X(I4 a)
        {
            return a;
        }

        public I8 X(I8 a)
        {
            return a;
        }

        internal static R4 X(R4 a)
        {
            return a;
        }

        public static BL F(BL a)
        {
            return a;
        }

        public static T Dump<T>(T a)
        {
            OutWriter.WriteLine("ExprDump: {0}", a);
            return a;
        }

        public static T Dump<T>(TX fmt, T a)
        {
            OutWriter.WriteLine(fmt.ToString(), a);
            return a;
        }

        public static TX DumpChars(TX a)
        {
            var sb = new StringBuilder();
            for (int ich = 0; ich < a.Length; ich++)
                sb.AppendFormat("{0:X4} ", (short)a.Span[ich]);
            OutWriter.WriteLine("ExprDumpChars: {0}", sb);
            return a;
        }
    }

    public sealed class TestFuncs2 : IFunctionProvider
    {
        private static volatile TestFuncs2 _instance;
        public static TestFuncs2 Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new TestFuncs2(), null);
                return _instance;
            }
        }

        public string NameSpace { get { return "__test2"; } }

        private TestFuncs2()
        {
        }

        private MethodInfo[] R(params Delegate[] funcs)
        {
            Contracts.AssertValue(funcs);
            var meths = new MethodInfo[funcs.Length];
            for (int i = 0; i < funcs.Length; i++)
            {
                Contracts.Assert(funcs[i] != null);
                Contracts.Assert(funcs[i].Target == null);
                Contracts.Assert(funcs[i].GetMethodInfo() != null);
                meths[i] = funcs[i].GetMethodInfo();
            }
            return meths;
        }

        public MethodInfo[] Lookup(string name)
        {
            switch (name)
            {
                case "_ab":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(A));
                case "_ac":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I8>(A));
                case "_ad":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4>(A));
            }

            return null;
        }

        public object ResolveToConstant(string name, MethodInfo meth, object[] values)
        {
            return null;
        }

        public static I4 A(I4 a)
        {
            return a * 3 * 10;
        }

        public static I8 A(I8 a)
        {
            return a * 2 * 10;
        }

        public static R4 A(R4 a)
        {
            return -a * 10;
        }
    }
}
