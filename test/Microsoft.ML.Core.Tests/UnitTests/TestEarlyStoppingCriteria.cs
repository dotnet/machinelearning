﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public sealed class TestEarlyStoppingCriteria : BaseTestClass
    {
        public TestEarlyStoppingCriteria(ITestOutputHelper output) : base(output)
        {
        }

        private EarlyStoppingRuleBase CreateEarlyStoppingCriterion(string name, string args, bool lowerIsBetter)
        {
            var env = new MLContext(1)
                .AddStandardComponents();
            var sub = new SubComponent<EarlyStoppingRuleBase, SignatureEarlyStoppingCriterion>(name, args);
            return sub.CreateInstance(env, lowerIsBetter);
        }

        [Fact]
        public void TolerantEarlyStoppingCriterionTest()
        {
            EarlyStoppingRuleBase cr = CreateEarlyStoppingCriterion("tr", "th=0.01", false);

            bool isBestCandidate;
            bool shouldStop;

            for (int i = 0; i < 100; i++)
            {
                float score = 0.001f * i;
                shouldStop = cr.CheckScore(score, 0, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }

            shouldStop = cr.CheckScore(0.09f, 0, out isBestCandidate);
            Assert.False(shouldStop);
            Assert.False(isBestCandidate);

            shouldStop = cr.CheckScore(0.07f, 0, out isBestCandidate);
            Assert.True(shouldStop);
            Assert.False(isBestCandidate);
        }

        [Fact]
        public void GLEarlyStoppingCriterionTest()
        {
            EarlyStoppingRuleBase cr = CreateEarlyStoppingCriterion("gl", "th=0.01", false);

            bool isBestCandidate;
            bool shouldStop;

            for (int i = 0; i < 100; i++)
            {
                float score = 0.001f * i;
                shouldStop = cr.CheckScore(score, 0, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }

            shouldStop = cr.CheckScore(1.0f, 0, out isBestCandidate);
            Assert.True(isBestCandidate);
            Assert.False(shouldStop);

            shouldStop = cr.CheckScore(0.98f, 0, out isBestCandidate);
            Assert.False(isBestCandidate);
            Assert.True(shouldStop);
        }

        [Fact]
        public void LPEarlyStoppingCriterionTest()
        {
            EarlyStoppingRuleBase cr = CreateEarlyStoppingCriterion("lp", "th=0.01 w=5", false);

            bool isBestCandidate;
            bool shouldStop;

            for (int i = 0; i < 100; i++)
            {
                float score = 0.001f * i;
                shouldStop = cr.CheckScore(score, score, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }

            for (int i = 1; i <= 10; i++)
            {
                shouldStop = cr.CheckScore(i, i, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }
            // At this point, average of score should be 8 and the best score should be 10.

            for (int i = 0; i < 3; i++)
            {
                shouldStop = cr.CheckScore(0, 10f, out isBestCandidate);
                Assert.False(isBestCandidate);
                Assert.False(shouldStop);
            }

            shouldStop = cr.CheckScore(0, 10f, out isBestCandidate);
            Assert.False(isBestCandidate);
            Assert.True(shouldStop);
        }

        [Fact]
        public void PQEarlyStoppingCriterionTest()
        {
            EarlyStoppingRuleBase cr = CreateEarlyStoppingCriterion("pq", "th=0.01 w=5", false);

            bool isBestCandidate;
            bool shouldStop;

            for (int i = 0; i < 100; i++)
            {
                float score = 0.001f * i;
                shouldStop = cr.CheckScore(score, score, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }

            for (int i = 1; i <= 10; i++)
            {
                shouldStop = cr.CheckScore(i, i, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }
            // At this point, average of score should be 8 and the best score should be 10.

            for (int i = 0; i < 3; i++)
            {
                shouldStop = cr.CheckScore(10f, 10f, out isBestCandidate);
                Assert.False(isBestCandidate);
                Assert.False(shouldStop);
            }

            shouldStop = cr.CheckScore(0, 10f, out isBestCandidate);
            Assert.False(isBestCandidate);
            Assert.True(shouldStop);
        }

        [Fact]
        public void UPEarlyStoppingCriterionTest()
        {
            const int windowSize = 8;
            EarlyStoppingRuleBase cr = CreateEarlyStoppingCriterion("up", "w=8", false);

            bool isBestCandidate;
            bool shouldStop;

            for (int i = 0; i < 100; i++)
            {
                float score = 0.001f * i;
                shouldStop = cr.CheckScore(score, 0, out isBestCandidate);
                Assert.True(isBestCandidate);
                Assert.False(shouldStop);
            }

            for (int i = 0; i < windowSize - 1; i++)
            {
                float score = 0.09f - 0.001f * i;
                shouldStop = cr.CheckScore(score, 0, out isBestCandidate);
                Assert.False(isBestCandidate);
                Assert.False(shouldStop);
            }

            shouldStop = cr.CheckScore(0.0f, 0, out isBestCandidate);
            Assert.True(shouldStop);
            Assert.False(isBestCandidate);
        }
    }
}
