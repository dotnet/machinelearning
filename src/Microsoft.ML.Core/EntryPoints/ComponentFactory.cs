// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This is a token interface that all component factories must implement.
    /// </summary>
    public interface IComponentFactory
    {
    }

    /// <summary>
    /// This is a token interface for component classes that don't directly create
    /// a runtime component, but only need to represent a polymorphic set of arguments.
    /// </summary>
    public interface IArgsComponent : IComponentFactory
    {
    }

    /// <summary>
    /// An interface for creating a component with no extra parameters (other than an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env);
    }

    /// <summary>
    /// An interface for creating a component when we take one extra parameter (and an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<in TArg1, out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env, TArg1 argument1);
    }

    /// <summary>
    /// An interface for creating a component when we take two extra parameters (and an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<in TArg1, in TArg2, out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env, TArg1 argument1, TArg2 argument2);
    }

    /// <summary>
    /// An interface for creating a component when we take three extra parameters (and an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<in TArg1, in TArg2, in TArg3, out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env, TArg1 argument1, TArg2 argument2, TArg3 argument3);
    }

    /// <summary>
    /// A utility class for creating <see cref="IComponentFactory"/> instances.
    /// </summary>
    public static class ComponentFactoryUtils
    {
        /// <summary>
        /// Creates a component factory with no extra parameters (other than an <see cref="IHostEnvironment"/>)
        /// that simply wraps a delegate which creates the component.
        /// </summary>
        public static IComponentFactory<TComponent> CreateFromFunction<TComponent>(Func<IHostEnvironment, TComponent> factory)
        {
            return new SimpleComponentFactory<TComponent>(factory);
        }

        /// <summary>
        /// Creates a component factory when we take one extra parameter (and an
        /// <see cref="IHostEnvironment"/>) that simply wraps a delegate which creates the component.
        /// </summary>
        public static IComponentFactory<TArg1, TComponent> CreateFromFunction<TArg1, TComponent>(Func<IHostEnvironment, TArg1, TComponent> factory)
        {
            return new SimpleComponentFactory<TArg1, TComponent>(factory);
        }

        /// <summary>
        /// Creates a component factory when we take three extra parameters (and an
        /// <see cref="IHostEnvironment"/>) that simply wraps a delegate which creates the component.
        /// </summary>
        public static IComponentFactory<TArg1, TArg2, TArg3, TComponent> CreateFromFunction<TArg1, TArg2, TArg3, TComponent>(Func<IHostEnvironment, TArg1, TArg2, TArg3, TComponent> factory)
        {
            return new SimpleComponentFactory<TArg1, TArg2, TArg3, TComponent>(factory);
        }

        /// <summary>
        /// A class for creating a component with no extra parameters (other than an <see cref="IHostEnvironment"/>)
        /// that simply wraps a delegate which creates the component.
        /// </summary>
        private sealed class SimpleComponentFactory<TComponent> : IComponentFactory<TComponent>
        {
            private readonly Func<IHostEnvironment, TComponent> _factory;

            public SimpleComponentFactory(Func<IHostEnvironment, TComponent> factory)
            {
                _factory = factory;
            }

            public TComponent CreateComponent(IHostEnvironment env)
            {
                return _factory(env);
            }
        }

        /// <summary>
        /// A class for creating a component when we take one extra parameter
        /// (and an <see cref="IHostEnvironment"/>) that simply wraps a delegate which
        /// creates the component.
        /// </summary>
        private sealed class SimpleComponentFactory<TArg1, TComponent> : IComponentFactory<TArg1, TComponent>
        {
            private readonly Func<IHostEnvironment, TArg1, TComponent> _factory;

            public SimpleComponentFactory(Func<IHostEnvironment, TArg1, TComponent> factory)
            {
                _factory = factory;
            }

            public TComponent CreateComponent(IHostEnvironment env, TArg1 argument1)
            {
                return _factory(env, argument1);
            }
        }

        /// <summary>
        /// A class for creating a component when we take three extra parameters
        /// (and an <see cref="IHostEnvironment"/>) that simply wraps a delegate which
        /// creates the component.
        /// </summary>
        private sealed class SimpleComponentFactory<TArg1, TArg2, TArg3, TComponent> : IComponentFactory<TArg1, TArg2, TArg3, TComponent>
        {
            private readonly Func<IHostEnvironment, TArg1, TArg2, TArg3, TComponent> _factory;

            public SimpleComponentFactory(Func<IHostEnvironment, TArg1, TArg2, TArg3, TComponent> factory)
            {
                _factory = factory;
            }

            public TComponent CreateComponent(IHostEnvironment env, TArg1 argument1, TArg2 argument2, TArg3 argument3)
            {
                return _factory(env, argument1, argument2, argument3);
            }
        }
    }
}
