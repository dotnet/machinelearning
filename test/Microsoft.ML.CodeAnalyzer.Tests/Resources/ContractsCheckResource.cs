// Note that this is *not* an actual source file, it is instead an embedded
// resource for the ContractsCheckTest. It contains both the utilizing test
// code, as well as code for "Contracts" derived from and intended to resemble
// the corresponding code in ML.NET.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Model;

namespace TestNamespace
{
    public sealed class TypeName
    {
        public TypeName(IHostEnvironment env, float p, int foo)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckParam(0 <= p && p <= 1, nameof(p), "Should be in range [0,1]");
            env.CheckParam(0 <= p && p <= 1, "p"); // Should fail.
            env.CheckParam(0 <= p && p <= 1, nameof(p) + nameof(p)); // Should fail.
            env.CheckValue(paramName: nameof(p), val: "p"); // Should succeed despite confusing order.
            env.CheckValue(paramName: "p", val: nameof(p)); // Should fail despite confusing order.
            env.CheckValue("p", nameof(p));
            env.CheckUserArg(foo > 5, "foo", "Nice");
            env.CheckUserArg(foo > 5, nameof(foo), "Nice");
            env.Except(); // Not throwing or doing anything with the exception, so should fail.
            Contracts.ExceptParam(nameof(env), "What a silly env"); // Should also fail.
            if (false)
                throw env.Except(); // Should not fail.
            if (false)
                throw env.ExceptParam(nameof(env), "What a silly env"); // Should not fail.
            if (false)
                throw env.ExceptParam("env", "What a silly env"); // Should fail due to name error.
            var e = env.Except();
            env.Check(true, $"Hello {foo} is cool");
            env.Check(true, "Hello it is cool");
            string coolMessage = "Hello it is cool";
            env.Check(true, coolMessage);
            env.Check(true, string.Format("Hello {0} is cool", foo));
            env.Check(true, Messages.CoolMessage);
            env.CheckDecode(true, "Not suspicious, no ModelLoadContext");
            Contracts.Check(true, "Fine: " + nameof(env));
            Contracts.Check(true, "Less fine: " + env.GetType().Name);
            Contracts.CheckUserArg(0 <= p && p <= 1,
                "p", "On a new line");
        }

        private void Loader(ModelLoadContext ctx)
        {
            Contracts.CheckDecode(true, "This message is suspicious");
        }

        private Exception CreateException() => Contracts.Except(); // This should be fine, since it's a return value not a standalone.
    }

    public static class Messages
    {
        public const string CoolMessage = "This is super cool";
    }
}
