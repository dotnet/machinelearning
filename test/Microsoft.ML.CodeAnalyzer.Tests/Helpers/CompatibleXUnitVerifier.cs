// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    /// <summary>
    /// A drop-in replacement for xUnit.net's default verifier that composes nicely with xUnit.net 2.8+.
    /// </summary>
    internal class CompatibleXUnitVerifier : IVerifier
    {
        public CompatibleXUnitVerifier()
            : this(ImmutableStack<string>.Empty)
        {
        }

        private CompatibleXUnitVerifier(ImmutableStack<string> context)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
        }

        private ImmutableStack<string> Context { get; }

        public void Empty<T>(string collectionName, IEnumerable<T> collection)
        {
            using var enumerator = collection.GetEnumerator();
            if (enumerator.MoveNext())
            {
                throw new XunitException(ComposeMessage($"'{collectionName}' is not empty"));
            }
        }

        public void Equal<T>(T expected, T actual, string? message = null)
        {
            if (message is null && Context.IsEmpty)
            {
                Assert.Equal(expected, actual);
                return;
            }

            if (EqualityComparer<T>.Default.Equals(expected, actual))
            {
                return;
            }

            throw new XunitException(ComposeMessageWithDetails(message, BuildEqualityMessage(expected, actual)));
        }

#if NETCOREAPP
        public void True([DoesNotReturnIf(false)] bool assert, string? message = null)
#else
        public void True(bool assert, string? message = null)
#endif
        {
            if (message is null && Context.IsEmpty)
            {
                Assert.True(assert);
            }
            else
            {
                Assert.True(assert, ComposeMessage(message));
            }
        }

#if NETCOREAPP
        public void False([DoesNotReturnIf(true)] bool assert, string? message = null)
#else
        public void False(bool assert, string? message = null)
#endif
        {
            if (message is null && Context.IsEmpty)
            {
                Assert.False(assert);
            }
            else
            {
                Assert.False(assert, ComposeMessage(message));
            }
        }

#if NETCOREAPP
        [DoesNotReturn]
#endif
#if !NETCOREAPP
#pragma warning disable CS8770 // Attribute unavailable on this target.
#endif
        public void Fail(string? message = null)
            => throw new XunitException(ComposeMessage(message));
#if !NETCOREAPP
#pragma warning restore CS8770
#endif

        public void LanguageIsSupported(string language)
        {
            if (language != LanguageNames.CSharp && language != LanguageNames.VisualBasic)
            {
                throw new XunitException(ComposeMessage($"Unsupported Language: '{language}'"));
            }
        }

        public void NotEmpty<T>(string collectionName, IEnumerable<T> collection)
        {
            using var enumerator = collection.GetEnumerator();
            if (!enumerator.MoveNext())
            {
                throw new XunitException(ComposeMessage($"'{collectionName}' is empty"));
            }
        }

        public void SequenceEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual, IEqualityComparer<T>? equalityComparer = null, string? message = null)
        {
            var comparer = new SequenceEqualEnumerableEqualityComparer<T>(equalityComparer);
            if (comparer.Equals(expected, actual))
            {
                return;
            }

            throw new XunitException(ComposeMessageWithDetails(message, BuildEqualityMessage(expected?.ToArray(), actual?.ToArray())));
        }

        public IVerifier PushContext(string context)
            => new CompatibleXUnitVerifier(Context.Push(context));

        private string ComposeMessage(string? message)
        {
            foreach (var frame in Context)
            {
                message = "Context: " + frame + Environment.NewLine + message;
            }

            return message ?? string.Empty;
        }

        private string ComposeMessageWithDetails(string? message, string details)
        {
            var baseMessage = ComposeMessage(message);
            if (string.IsNullOrEmpty(baseMessage))
            {
                return details;
            }

            return baseMessage + Environment.NewLine + details;
        }

        private static string BuildEqualityMessage<TExpected, TActual>(TExpected expected, TActual actual)
        {
            try
            {
                Assert.Equal((object?)expected, (object?)actual);
            }
            catch (XunitException ex)
            {
                return ex.Message;
            }

            return "Values are not equal.";
        }

        private sealed class SequenceEqualEnumerableEqualityComparer<T> : IEqualityComparer<IEnumerable<T>?>
        {
            private readonly IEqualityComparer<T> _itemComparer;

            public SequenceEqualEnumerableEqualityComparer(IEqualityComparer<T>? itemComparer)
            {
                _itemComparer = itemComparer ?? EqualityComparer<T>.Default;
            }

            public bool Equals(IEnumerable<T>? x, IEnumerable<T>? y)
            {
                if (ReferenceEquals(x, y))
                {
                    return true;
                }

                if (x is null || y is null)
                {
                    return false;
                }

                using var enumeratorX = x.GetEnumerator();
                using var enumeratorY = y.GetEnumerator();

                while (true)
                {
                    var hasX = enumeratorX.MoveNext();
                    var hasY = enumeratorY.MoveNext();

                    if (!hasX || !hasY)
                    {
                        return hasX == hasY;
                    }

                    if (!_itemComparer.Equals(enumeratorX.Current, enumeratorY.Current))
                    {
                        return false;
                    }
                }
            }

            public int GetHashCode(IEnumerable<T>? obj)
            {
                if (obj is null)
                {
                    return 0;
                }

                return obj.Select(item => _itemComparer.GetHashCode(item!))
                    .Aggregate(0, (agg, next) => ((agg << 5) + agg) ^ next);
            }
        }
    }
}
