using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using System;
namespace Bubba
{
    public class Foo
    {
        public Foo(int yuck) { if (false) Contracts.ExceptParam(nameof(yuck)); }
        public static void Bar(float tom, Arguments args)
        {
            string str = "hello";
            Contracts.CheckValue(str, nameof(str));
            Contracts.CheckUserArg(0 <= A.B.Foo && A.B.Foo < 10, nameof(A.B.Foo), "Should be in range[0, 10)");
            Contracts.CheckUserArg(A.B.Bar.Length == 2, nameof(A.B.Bar), "Length must be exactly 2");
            Contracts.CheckUserArg(A.B.Bar.Length == 2, nameof(A), "Length must be exactly 2");
            if (false) throw Contracts.ExceptParam(nameof(A.B.Bar), $"Length should have been 2 but was { A.B.Bar.Length}");
            Func<int, bool> isFive = val => val == 5;
            Contracts.CheckParam(!isFive(4),
                nameof(isFive));
            Contracts.CheckValue(typeof(X.Y.Z), nameof(X.Y));
            if (false) throw Contracts.ExceptParam(nameof(tom));
            Contracts.CheckValue(str, "noMatch");
            Contracts.CheckUserArg(str.Length == 2, nameof(args.chumble), "Whoa!");
            Contracts.CheckUserArg(str.Length == 2, nameof(args.spuzz), "Git along, little dogies, git along...");
        }
    }
    public static class A { public static class B { public const int Foo = 5; public const string Bar = "Yo"; } }
    public static class X { public static class Y { public static class Z { } } }
    public sealed class Arguments
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Yakka foob mog.", ShortName = "chum")]
        public int chumble;
        [Argument(ArgumentType.AtMostOnce, HelpText = "Grug pubbawup zink wattoom gazork.", ShortName = "spu,sp")]
        public int spuzz;
    }
}