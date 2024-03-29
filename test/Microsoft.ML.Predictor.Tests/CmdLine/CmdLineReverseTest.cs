﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class CmdLineReverseTests : BaseTestClass
    {
        /// <summary>
        /// This tests CmdParser.GetSettings
        /// </summary>
        [Fact]
        [TestCategory("Cmd Parsing")]
        public void ArgumentParseTest()
        {
            var env = new MLContext(seed: 42);
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
                sub1 = CreateComponentFactory("S1", innerArg1.ToString(env)),
                sub2 = CreateComponentFactory("S2", innerArg2.ToString(env)),
            };

            var outerArg1 = new SimpleArg()
            {
                required = -2,
                once = 2,
                text2 = "Testing",
                text3 = "a=7",
                sub1 = CreateComponentFactory("S1", innerArg1.ToString(env)),
                sub2 = CreateComponentFactory("S2", innerArg2.ToString(env)),
                sub3 = CreateComponentFactory("S3", innerArg3.ToString(env)),
            };

            var testArg = new SimpleArg();
            CmdParser.ParseArguments(env, outerArg1.ToString(env), testArg);
            Assert.Equal(outerArg1, testArg);

            CmdParser.ParseArguments(env, ((ICommandLineComponentFactory)outerArg1.sub1).GetSettingsString(), testArg = new SimpleArg());
            Assert.Equal(innerArg1, testArg);

            CmdParser.ParseArguments(env, ((ICommandLineComponentFactory)outerArg1.sub2).GetSettingsString(), testArg = new SimpleArg());
            Assert.Equal(innerArg2, testArg);

            CmdParser.ParseArguments(env, ((ICommandLineComponentFactory)outerArg1.sub3).GetSettingsString(), testArg = new SimpleArg());
            Assert.Equal(innerArg3, testArg);
        }

        [Fact]
        [TestCategory("Cmd Parsing")]
        public void NewTest()
        {
            var ml = new MLContext(1);
            ml.AddStandardComponents();
            var findLoadableClassesMethodInfo = new FuncInstanceMethodInfo1<ComponentCatalog, ComponentCatalog.LoadableClassInfo[]>(ml.ComponentCatalog.FindLoadableClasses<int>);
            var classes = Utils.MarshalInvoke(findLoadableClassesMethodInfo, ml.ComponentCatalog, typeof(SignatureCalibrator));
            foreach (var cls in classes)
            {
                var factory = CmdParser.CreateComponentFactory(typeof(IComponentFactory<ICalibratorTrainer>), typeof(SignatureCalibrator), cls.LoadNames[0]);
                var calibrator = ((IComponentFactory<ICalibratorTrainer>)factory).CreateComponent(ml);
            }
            var components = ml.ComponentCatalog.GetAllComponents(typeof(ICalibratorTrainerFactory));
            foreach (var component in components)
            {
                var factory = CmdParser.CreateComponentFactory(typeof(IComponentFactory<ICalibratorTrainer>), typeof(SignatureCalibrator), component.Aliases[0]);
                var calibrator = ((IComponentFactory<ICalibratorTrainer>)factory).CreateComponent(ml);
            }
        }

        private delegate void SignatureSimpleComponent();

        private class SimpleArg
        {
            [Argument(ArgumentType.Required, ShortName = "r")]
            public int required = -1;

            [Argument(ArgumentType.AtMostOnce)]
            public int once = 1;

            [Argument(ArgumentType.LastOccurrenceWins)]
            public string text1 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text2 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text3 = "";

            [Argument(ArgumentType.AtMostOnce)]
            public string text4 = "";

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureSimpleComponent))]
            public IComponentFactory<SimpleArg> sub1 = CreateComponentFactory("sub1", "settings1");

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureSimpleComponent))]
            public IComponentFactory<SimpleArg> sub2 = CreateComponentFactory("sub2", "settings2");

            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureSimpleComponent))]
            public IComponentFactory<SimpleArg> sub3 = CreateComponentFactory("sub3", "settings3");

            // REVIEW: include Subcomponent array for testing once it is supported
            //[Argument(ArgumentType.Multiple)]
            //public SubComponent[] sub4 = new SubComponent[] { new SubComponent("sub4", "settings4"), new SubComponent("sub5", "settings5") };

            /// <summary>
            /// ToString is overridden by CmdParser.GetSettings which is of primary for this test
            /// </summary>
            /// <returns></returns>
            public string ToString(IHostEnvironment env)
            {
                return CmdParser.GetSettings(env, this, new SimpleArg(), SettingsFlags.None);
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
                if (!ComponentFactoryEquals(arg.sub1, this.sub1))
                    return false;
                if (!ComponentFactoryEquals(arg.sub2, this.sub2))
                    return false;
                if (!ComponentFactoryEquals(arg.sub3, this.sub3))
                    return false;

                return true;
            }

            public override int GetHashCode()
            {
                return base.GetHashCode();
            }
        }

        private static bool ComponentFactoryEquals(IComponentFactory left, IComponentFactory right)
        {
            if (!(left is ICommandLineComponentFactory commandLineLeft &&
                right is ICommandLineComponentFactory commandLineRight))
            {
                // can't compare non-command-line component factories
                return false;
            }

            return commandLineLeft.Name == commandLineRight.Name &&
                commandLineLeft.SignatureType == commandLineRight.SignatureType &&
                commandLineLeft.GetSettingsString() == commandLineRight.GetSettingsString();
        }

        private static readonly MethodInfo _createComponentFactoryMethod = typeof(CmdParser)
            .GetNestedType("ComponentFactoryFactory", BindingFlags.NonPublic)
            .GetMethod("CreateComponentFactory");

        public CmdLineReverseTests(ITestOutputHelper output) : base(output)
        {
        }

        private static IComponentFactory<SimpleArg> CreateComponentFactory(string name, string settings)
        {
            return (IComponentFactory<SimpleArg>)_createComponentFactoryMethod.Invoke(null,
                new object[]
                {
                    typeof(IComponentFactory<SimpleArg>),
                    typeof(SignatureSimpleComponent),
                    name,
                    new string[] { settings }
                });
        }
    }
}
