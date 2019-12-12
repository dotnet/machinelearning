﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using DiffPlex;
using DiffPlex.DiffBuilder;
using DiffPlex.DiffBuilder.Model;
using Xunit;

namespace Microsoft.ML.TestFramework
{
    public static class AssertEx
    {
        /// <summary>
        /// Asserts that two strings are equal, and prints a diff between the two if they are not.
        /// </summary>
        /// <param name="expected">The expected string. This is presented as the "baseline/before" side in the diff.</param>
        /// <param name="actual">The actual string. This is presented as the changed or "after" side in the diff.</param>
        /// <param name="message">The message to precede the diff, if the values are not equal.</param>
        public static void EqualOrDiff(string expected, string actual, string message = null)
        {
            if (expected == actual)
            {
                return;
            }

            var diffBuilder = new InlineDiffBuilder(new Differ());
            var diff = diffBuilder.BuildDiffModel(expected, actual, ignoreWhitespace: false);
            var messageBuilder = new StringBuilder();
            messageBuilder.AppendLine(
                string.IsNullOrEmpty(message)
                    ? "Actual and expected values differ. Expected shown in baseline of diff:"
                    : message);

            foreach (var line in diff.Lines)
            {
                switch (line.Type)
                {
                    case ChangeType.Inserted:
                        messageBuilder.Append("+");
                        break;
                    case ChangeType.Deleted:
                        messageBuilder.Append("-");
                        break;
                    default:
                        messageBuilder.Append(" ");
                        break;
                }

                messageBuilder.AppendLine(line.Text);
            }

            Assert.True(false, messageBuilder.ToString());
        }
    }
}
