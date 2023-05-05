// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A base class for environment-specific fact attributes.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = true)]
    public abstract class EnvironmentSpecificTheoryAttribute : TheoryAttribute
    {
        private readonly string _skipMessage;

        /// <summary>
        /// Creates a new instance of the <see cref="EnvironmentSpecificTheoryAttribute" /> class.
        /// </summary>
        /// <param name="skipMessage">The message to be used when skipping the test marked with this attribute.</param>
        protected EnvironmentSpecificTheoryAttribute(string skipMessage)
        {
            _skipMessage = skipMessage ?? throw new ArgumentNullException(nameof(skipMessage));
        }

        public sealed override string Skip => IsEnvironmentSupported() ? null : _skipMessage;

        /// <summary>
        /// A method used to evaluate whether to skip a test marked with this attribute. Skips iff this method evaluates to false.
        /// </summary>
        protected abstract bool IsEnvironmentSupported();
    }
}
