// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Xunit;
namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class TestContracts
    {
        private void Helper(IExceptionContext ectx, MessageSensitivity expected)
        {
            Contracts.AssertValueOrNull(ectx);
            bool caught = false;

            try
            {
                ectx.CheckParam(false, nameof(ectx), "Whoops");
            }
            catch (Exception e)
            {
                Assert.Equal(expected, e.Sensitivity());
                caught = true;
            }
            Assert.True(caught, "Exception was not caught at all");

            caught = false;
            try
            {
                throw ectx.ExceptNotSupp("Oops!");
            }
            catch (Exception e)
            {
                Assert.Equal(expected, e.Sensitivity());
                caught = true;
            }
            Assert.True(caught, "Exception was not caught at all");
        }

        [Fact]
        public void ExceptionSensitivity()
        {
            var env = new ConsoleEnvironment();
            // Default sensitivity should be unknown, that is, all bits set.
            Helper(null, MessageSensitivity.Unknown);
            // If we set it to be not sensitive, then the messages should be marked insensitive,
            // and so forth.
            Helper(Contracts.NotSensitive(), MessageSensitivity.None);
            Helper(Contracts.UserSensitive(), MessageSensitivity.UserData);
            Helper(Contracts.SchemaSensitive(), MessageSensitivity.Schema);
            Helper(Contracts.UserSensitive().SchemaSensitive(), MessageSensitivity.UserData | MessageSensitivity.Schema);

            // Run these same tests with the environment.
            Helper(env, MessageSensitivity.Unknown);
            Helper(env.NotSensitive(), MessageSensitivity.None);
            Helper(env.UserSensitive(), MessageSensitivity.UserData);
            Helper(env.SchemaSensitive(), MessageSensitivity.Schema);
            Helper(env.UserSensitive().SchemaSensitive(), MessageSensitivity.UserData | MessageSensitivity.Schema);
        }
    }
}
