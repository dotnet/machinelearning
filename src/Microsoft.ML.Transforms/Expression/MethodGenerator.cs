// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// MethodGenerator is used to build a method by specifying the IL.
    /// </summary>
    internal sealed class MethodGenerator : IDisposable
    {
        private DynamicMethod _method;

        public ILGenerator Il { get; private set; }

        public MethodGenerator(string name, Type thisType, Type returnType, params Type[] parameterTypes)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(thisType, nameof(thisType));
            Contracts.AssertValueOrNull(returnType);
            Contracts.AssertValueOrNull(parameterTypes);

            _method = new DynamicMethod(name, returnType, parameterTypes, thisType);
            Il = _method.GetILGenerator();
        }

        /// <summary>
        /// Create and return a delegate of the given "delegateType" and currying the
        /// given "thisObj". Note that "thisObj" may be null and "delegateType" should
        /// match the types indicated when the MethodGenerator was created.
        /// </summary>
        public Delegate CreateDelegate(Type delegateType)
        {
            Contracts.CheckValue(delegateType, nameof(delegateType));

            var del = _method.CreateDelegate(delegateType, null);
            Dispose();
            return del;
        }

        /// <summary>
        /// This is idempotent (calling multiple times has the same affect as calling once).
        /// </summary>
        public void Dispose()
        {
            _method = null;
            Il = null;
            _locals = null;
        }

        /// <summary>
        /// Represents a temporary local. A MethodGenerator maintains a cache of temporary locals.
        /// Clients can 'check-out' temporaries from the cache, use them, then release (Dispose) when
        /// they are no longer needed so other clients can reuse the temporary to cut down on the number
        /// of locals created. Note that Temporary is a struct, but should NOT be copied around. It is
        /// critical that a stray copy of a Temporary is not disposed!
        ///
        /// We have two 'types' of temporaries: 'Regular' and 'Ref'. 'Ref' temporaries are special in that
        /// they should be used when you need to use ldloca. Grouping this way should help the JIT optimize
        /// local usage.
        /// </summary>
        public struct Temporary : IDisposable
        {
            private Action<LocalBuilder, bool> _dispose;
            private readonly bool _isRef;

            // Should only be created by MethodGenerator. Too bad C# can't enforce this without
            // reversing the class nesting.
            internal Temporary(Action<LocalBuilder, bool> dispose, LocalBuilder localBuilder, bool isRef)
            {
                Contracts.AssertValue(dispose);
                Contracts.AssertValue(localBuilder);

                _dispose = dispose;
                Local = localBuilder;
                _isRef = isRef;
            }

            public LocalBuilder Local { get; private set; }

            public void Dispose()
            {
                if (Local == null)
                    return;

                Contracts.AssertValue(_dispose);
                _dispose(Local, _isRef);
                Local = null;
                _dispose = null;
            }
        }

        private struct LocalKey : IEquatable<LocalKey>
        {
            public readonly Type Type;
            public readonly bool IsRef;

            public LocalKey(Type type, bool isRef)
            {
                Contracts.AssertValue(type);
                Type = type;
                IsRef = isRef;
            }

            public static bool operator ==(LocalKey key0, LocalKey key1)
            {
                return key0.Equals(key1);
            }

            public static bool operator !=(LocalKey key0, LocalKey key1)
            {
                return !key0.Equals(key1);
            }

            public override int GetHashCode()
            {
                return Hashing.CombineHash(Type.GetHashCode(), Hashing.HashInt(IsRef.GetHashCode()));
            }

            public override bool Equals(object obj)
            {
                if (obj == null || !(obj is LocalKey))
                    return false;
                return Equals((LocalKey)obj);
            }

            public bool Equals(LocalKey other)
            {
                return IsRef == other.IsRef && Type == other.Type;
            }
        }

        private Dictionary<LocalKey, List<LocalBuilder>> _locals;
        private Action<LocalBuilder, bool> _tempDisposer;

        public Temporary AcquireTemporary(Type type, bool isRef = false)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.Check(Il != null, "Cannot access IL for a method that has already been created");

            if (_tempDisposer == null)
                _tempDisposer = ReleaseLocal;

            LocalKey key = new LocalKey(type, isRef);
            List<LocalBuilder> locals;
            if (_locals != null && _locals.TryGetValue(key, out locals) && locals.Count > 0)
            {
                var temp = locals[locals.Count - 1];
                locals.RemoveAt(locals.Count - 1);
                return new Temporary(_tempDisposer, temp, key.IsRef);
            }
            return new Temporary(_tempDisposer, Il.DeclareLocal(key.Type), key.IsRef);
        }

        /// <summary>
        /// Called by the Temporary struct's Dispose method, through the _tempDisposer delegate.
        /// </summary>
        private void ReleaseLocal(LocalBuilder localBuilder, bool isRef)
        {
            Contracts.AssertValue(localBuilder);

            LocalKey key = new LocalKey(localBuilder.LocalType, isRef);

            List<LocalBuilder> locals;
            if (_locals == null)
                _locals = new Dictionary<LocalKey, List<LocalBuilder>>();
            else if (_locals.TryGetValue(key, out locals))
            {
                Contracts.Assert(!locals.Contains(localBuilder));
                locals.Add(localBuilder);
                return;
            }

            locals = new List<LocalBuilder>(4);
            _locals.Add(key, locals);
            locals.Add(localBuilder);
        }
    }
}
