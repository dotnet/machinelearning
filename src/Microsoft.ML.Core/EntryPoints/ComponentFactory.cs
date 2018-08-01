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
    public interface IComponentFactory<out TComponent>: IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env);
    }

    public class SimpleComponentFactory<TComponent> : IComponentFactory<TComponent>
    {
        private Func<IHostEnvironment, TComponent> _factory;

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
    /// An interface for creating a component when we take one extra parameter (and an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<in TArg1, out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env, TArg1 argument1);
    }

    public class SimpleComponentFactory<TArg1, TComponent> : IComponentFactory<TArg1, TComponent>
    {
        private Func<IHostEnvironment, TArg1, TComponent> _factory;

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
    /// An interface for creating a component when we take two extra parameters (and an <see cref="IHostEnvironment"/>).
    /// </summary>
    public interface IComponentFactory<in TArg1, in TArg2, out TComponent> : IComponentFactory
    {
        TComponent CreateComponent(IHostEnvironment env, TArg1 argument1, TArg2 argument2);
    }
}
