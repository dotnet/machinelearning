// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class GroupUngroupTests : TestDataPipeBase
    {
        /// <summary>
        /// Input data of <see cref="GroupTransform"/>.
        /// </summary>
        private class GroupExample
        {
            public int Age;
            public string UserName;
            public string Gender;
        }

        /// <summary>
        /// Input data of <see cref="UngroupTransform"/>.
        /// </summary>
        private class UngroupExample
        {
            public int Age;
            public string[] UserName; // Names grouped by Age
            public string[] Gender;   // Genders grouped by Age
        }

        public GroupUngroupTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void GroupTest()
        {
            var data = new List<GroupExample> {
                new GroupExample { Age=18, UserName="Amy", Gender="Girl"},
                new GroupExample { Age=18, UserName="Willy", Gender="Boy"},
                new GroupExample { Age=20, UserName="Dori", Gender="Fish" },
                new GroupExample { Age=20, UserName="Ariel", Gender="Mermaid" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var groupTransform = new GroupTransform(Env, dataView, "Age", "UserName", "Gender");
            var grouped = ML.Data.CreateEnumerable<UngroupExample>(groupTransform, false).ToList();

            // Expected content of grouped should contains two rows.
            // Age, UserName, Gender
            // 18,  {"Amy", "Willy"}, {"Girl", "Boy"} 
            // 20,  {"Dori", "Ariel"}, {"Fish", "Mermaid"} 
            // Note that "Age, UserName, Gender" is not a row; it just shows column names per row below it.
            Assert.Equal(2, grouped.Count);

            // grouped[0] is the first output row --- 18,  {"Amy", "Willy"}, {"Girl", "Boy"} 
            Assert.Equal(18, grouped[0].Age);
            Assert.Equal(2, grouped[0].UserName.Length);
            Assert.Equal("Amy", grouped[0].UserName[0]);
            Assert.Equal("Willy", grouped[0].UserName[1]);
            Assert.Equal(2, grouped[0].Gender.Length);
            Assert.Equal("Girl", grouped[0].Gender[0]);
            Assert.Equal("Boy", grouped[0].Gender[1]);

            // grouped[1] is the second output row --- 20,  {"Dori", "Ariel"}, {"Fish", "Mermaid"} 
            Assert.Equal(20, grouped[1].Age);
            Assert.Equal(2, grouped[1].Gender.Length);
            Assert.Equal("Dori", grouped[1].UserName[0]);
            Assert.Equal("Ariel", grouped[1].UserName[1]);
            Assert.Equal(2, grouped[1].Gender.Length);
            Assert.Equal("Fish", grouped[1].Gender[0]);
            Assert.Equal("Mermaid", grouped[1].Gender[1]);
        }

        [Fact]
        public void UgroupTest()
        {
            var data = new List<UngroupExample> {
                new UngroupExample { Age=18, UserName=new[]{"Amy", "Willy"}, Gender=new[]{"Girl", "Boy"} },
                new UngroupExample { Age=20, UserName=new[]{"Dori", "Ariel"}, Gender=new[]{"Fish", "Mermaid"} } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var ungroupTransform = new UngroupTransform(Env, dataView, UngroupTransform.UngroupMode.Inner, "UserName", "Gender");
            var ungrouped = ML.Data.CreateEnumerable<GroupExample>(ungroupTransform, false).ToList();

            Assert.Equal(4, ungrouped.Count);

            Assert.Equal(18, ungrouped[0].Age);
            Assert.Equal("Amy", ungrouped[0].UserName);
            Assert.Equal("Girl", ungrouped[0].Gender);

            Assert.Equal(18, ungrouped[1].Age);
            Assert.Equal("Willy", ungrouped[1].UserName);
            Assert.Equal("Boy", ungrouped[1].Gender);

            Assert.Equal(20, ungrouped[2].Age);
            Assert.Equal("Dori", ungrouped[2].UserName);
            Assert.Equal("Fish", ungrouped[2].Gender);

            Assert.Equal(20, ungrouped[3].Age);
            Assert.Equal("Ariel", ungrouped[3].UserName);
            Assert.Equal("Mermaid", ungrouped[3].Gender);
        }
    }
}

