// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public class CmdLineReverseTests
    {
        /// <summary>
        /// This tests CmdParser.GetSettings
        /// </summary>
        [Fact(Skip = "Sub Components Not Getting Parsed Correctly")]
        [TestCategory("Cmd Parsing")]
        public void ArgumentParseTest()
        {
            var env = new TlcEnvironment(seed: 42);
            var innerArg1 = new SimpleArg()
            {
                required = -2,
                text1 = "}",
                text2 = "{",
                text3 = "  ",
                text4 = "\n",
            };

            var innerArg2 = new SimpleArg()
            {
                required = -2,
                text1 = "{!@# $%^}&*{()}",
                text2 = "=",
                text3 = "\t",
                text4 = @"\\",
            };

            var innerArg3 = new SimpleArg()
            {
                required = -2,
                text1 = "\r\n",
                text2 = "\"",
                text3 = "\" \" ",
                text4 = "{/a=2 /b=3 /c=4}",
                sub1 = new SubComponent("S1", innerArg1.ToString(env)),
                sub2 = new SubComponent("S2", innerArg2.ToString(env)),
            };

            var outerArg1 = new SimpleArg()
            {
                required = -2,
                once = 2,
                text2 = "Testing",
                text3 = "a=7",
                sub1 = new SubComponent("S1", innerArg1.ToString(env)),
                sub2 = new SubComponent("S2", innerArg2.ToString(env)),
                sub3 = new SubComponent("S3", innerArg3.ToString(env)),
            };

            var testArg = new SimpleArg();
            CmdParser.ParseArguments(env, outerArg1.ToString(env), testArg);
            Assert.Equal(innerArg1, testArg);

            CmdParser.ParseArguments(env, outerArg1.sub1.SubComponentSettings, testArg = new SimpleArg());
            Assert.Equal(innerArg1, testArg);

            CmdParser.ParseArguments(env, outerArg1.sub1.Settings[0], testArg = new SimpleArg());
            Assert.Equal(innerArg1, testArg);

            CmdParser.ParseArguments(env, CmdParser.CombineSettings(outerArg1.sub1.Settings), testArg = new SimpleArg());
            Assert.Equal(innerArg1, testArg);

            CmdParser.ParseArguments(env, outerArg1.sub2.SubComponentSettings, testArg = new SimpleArg());
            Assert.Equal(innerArg2, testArg);

            CmdParser.ParseArguments(env, outerArg1.sub3.SubComponentSettings, testArg = new SimpleArg());
            Assert.Equal(innerArg3, testArg);
        }

        private class SimpleArg
        {
            [Argument(ArgumentType.Required, ShortName = "r")]
            public int required = -1;

            [Argument(ArgumentType.AtMostOnce)]
            public int once = 1;

            [Argument(ArgumentType.LastOccurenceWins)]
            public string text1 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text2 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text3 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text4 = "";

            [Argument(ArgumentType.Multiple)]
            public SubComponent sub1 = new SubComponent("sub1", "settings1");

            [Argument(ArgumentType.Multiple)]
            public SubComponent sub2 = new SubComponent("sub2", "settings2");

            [Argument(ArgumentType.Multiple)]
            public SubComponent sub3 = new SubComponent("sub3", "settings3");

            // REVIEW: include Subcomponent array for testing once it is supported
            //[Argument(ArgumentType.Multiple)]
            //public SubComponent[] sub4 = new SubComponent[] { new SubComponent("sub4", "settings4"), new SubComponent("sub5", "settings5") };

            /// <summary>
            /// ToString is overrided by CmdParser.GetSettings which is of primary for this test
            /// </summary>
            /// <returns></returns>
            public string ToString(IExceptionContext ectx)
            {
                return CmdParser.GetSettings(ectx, this, new SimpleArg(), SettingsFlags.None);
            }

            public override bool Equals(object obj)
            {
                var arg = (SimpleArg)obj;
                if (arg.required != this.required)
                    return false;
                if (arg.once != this.once)
                    return false;
                if (arg.text1 != this.text1)
                    return false;
                if (arg.text2 != this.text2)
                    return false;
                if (arg.text3 != this.text3)
                    return false;
                if (arg.text4 != this.text4)
                    return false;
                if (arg.sub1.Kind != this.sub1.Kind)
                    return false;
                if (arg.sub1.SubComponentSettings != this.sub1.SubComponentSettings)
                    return false;
                if (arg.sub2.Kind != this.sub2.Kind)
                    return false;
                if (arg.sub2.SubComponentSettings != this.sub2.SubComponentSettings)
                    return false;
                if (arg.sub3.Kind != this.sub3.Kind)
                    return false;
                if (arg.sub3.SubComponentSettings != this.sub3.SubComponentSettings)
                    return false;

                return true;
            }

            public override int GetHashCode()
            {
                return base.GetHashCode();
            }
        }
    }
}
