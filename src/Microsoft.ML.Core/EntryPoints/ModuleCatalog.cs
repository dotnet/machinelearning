// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This is a signature for classes that are 'holders' of entry points and components.
    /// </summary>
    public delegate void SignatureEntryPointModule();

    /// <summary>
    /// A simplified assembly attribute for marking EntryPoint modules.
    /// </summary>
    [AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
    public sealed class EntryPointModuleAttribute : LoadableClassAttributeBase
    {
        public EntryPointModuleAttribute(Type loaderType)
            : base(null, typeof(void), loaderType, null, new[] { typeof(SignatureEntryPointModule) }, loaderType.FullName)
        { }
    }

    /// <summary>
    /// This is a class that contains all entry point and component declarations.
    /// REVIEW: it looks like a good idea to actually make <see cref="ModuleCatalog"/> a part of <see cref="IHostEnvironment"/>. Currently, this is not the case,
    /// but this is the plan.
    /// </summary>
    public sealed class ModuleCatalog
    {
        /// <summary>
        /// A description of a single entry point.
        /// </summary>
        public sealed class EntryPointInfo
        {
            public readonly string Name;
            public readonly string Description;
            public readonly string ShortName;
            public readonly string FriendlyName;
            public readonly MethodInfo Method;
            public readonly Type InputType;
            public readonly Type OutputType;
            public readonly Type[] InputKinds;
            public readonly Type[] OutputKinds;

            internal EntryPointInfo(IExceptionContext ectx, MethodInfo method, TlcModule.EntryPointAttribute attribute)
            {
                Contracts.AssertValueOrNull(ectx);
                ectx.AssertValue(method);
                ectx.AssertValue(attribute);

                Name = attribute.Name ?? string.Join(".", method.DeclaringType.Name, method.Name);
                Description = attribute.Desc;
                Method = method;
                ShortName = attribute.ShortName;
                FriendlyName = attribute.UserName;

                // There are supposed to be 2 parameters, env and input for non-macro nodes.
                // Macro nodes have a 3rd parameter, the entry point node.
                var parameters = method.GetParameters();
                if (parameters.Length != 2 && parameters.Length != 3)
                    throw ectx.Except("Method '{0}' has {1} parameters, but must have 2 or 3", method.Name, parameters.Length);
                if (parameters[0].ParameterType != typeof(IHostEnvironment))
                    throw ectx.Except("Method '{0}', 1st parameter is {1}, but must be IHostEnvironment", method.Name, parameters[0].ParameterType);
                InputType = parameters[1].ParameterType;
                var outputType = method.ReturnType;
                if (!outputType.IsClass)
                    throw ectx.Except("Method '{0}' returns {1}, but must return a class", method.Name, outputType);
                OutputType = outputType;

                InputKinds = FindEntryPointKinds(InputType);
                OutputKinds = FindEntryPointKinds(OutputType);
            }

            private Type[] FindEntryPointKinds(Type type)
            {
                var kindAttr = type.GetTypeInfo().GetCustomAttributes(typeof(TlcModule.EntryPointKindAttribute), false).FirstOrDefault()
                    as TlcModule.EntryPointKindAttribute;
                var baseType = type.BaseType;

                if (baseType == null)
                    return kindAttr?.Kinds;
                var baseKinds = FindEntryPointKinds(baseType);
                if (kindAttr == null)
                    return baseKinds;
                if (baseKinds == null)
                    return kindAttr.Kinds;
                return kindAttr.Kinds.Concat(baseKinds).ToArray();
            }

            public override string ToString() => $"{Name}: {Description}";
        }

        /// <summary>
        /// A description of a single component.
        /// The 'component' is a non-standalone building block that is used to parametrize entry points or other TLC components.
        /// For example, 'Loss function', or 'similarity calculator' could be components.
        /// </summary>
        public sealed class ComponentInfo
        {
            public readonly string Name;
            public readonly string Description;
            public readonly string FriendlyName;
            public readonly string Kind;
            public readonly Type ArgumentType;
            public readonly Type InterfaceType;
            public readonly string[] Aliases;

            internal ComponentInfo(IExceptionContext ectx, Type interfaceType, string kind, Type argumentType, TlcModule.ComponentAttribute attribute)
            {
                Contracts.AssertValueOrNull(ectx);
                ectx.AssertValue(interfaceType);
                ectx.AssertNonEmpty(kind);
                ectx.AssertValue(argumentType);
                ectx.AssertValue(attribute);

                Name = attribute.Name;
                Description = attribute.Desc;
                if (string.IsNullOrWhiteSpace(attribute.FriendlyName))
                    FriendlyName = Name;
                else
                    FriendlyName = attribute.FriendlyName;

                Kind = kind;
                if (!IsValidName(Kind))
                    throw ectx.Except("Invalid component kind: '{0}'", Kind);

                Aliases = attribute.Aliases;
                if (!IsValidName(Name))
                    throw ectx.Except("Component name '{0}' is not valid.", Name);

                if (Aliases != null && Aliases.Any(x => !IsValidName(x)))
                    throw ectx.Except("Component '{0}' has an invalid alias '{1}'", Name, Aliases.First(x => !IsValidName(x)));

                if (!typeof(IComponentFactory).IsAssignableFrom(argumentType))
                    throw ectx.Except("Component '{0}' must inherit from IComponentFactory", argumentType);

                ArgumentType = argumentType;
                InterfaceType = interfaceType;
            }
        }

        private static volatile ModuleCatalog _instance;
        private readonly EntryPointInfo[] _entryPoints;
        private readonly Dictionary<string, EntryPointInfo> _entryPointMap;

        private readonly List<ComponentInfo> _components;
        private readonly Dictionary<string, ComponentInfo> _componentMap;

        /// <summary>
        /// Get all registered entry points.
        /// </summary>
        public IEnumerable<EntryPointInfo> AllEntryPoints()
        {
            return _entryPoints.AsEnumerable();
        }

        private ModuleCatalog(IExceptionContext ectx)
        {
            Contracts.AssertValue(ectx);

            _entryPointMap = new Dictionary<string, EntryPointInfo>();
            _componentMap = new Dictionary<string, ComponentInfo>();
            _components = new List<ComponentInfo>();

            var moduleClasses = ComponentCatalog.FindLoadableClasses<SignatureEntryPointModule>();
            var entryPoints = new List<EntryPointInfo>();

            foreach (var lc in moduleClasses)
            {
                var type = lc.LoaderType;

                // Scan for entry points.
                foreach (var methodInfo in type.GetMethods(BindingFlags.Static | BindingFlags.Public))
                {
                    var attr = methodInfo.GetCustomAttributes(typeof(TlcModule.EntryPointAttribute), false).FirstOrDefault() as TlcModule.EntryPointAttribute;
                    if (attr == null)
                        continue;
                    var info = new EntryPointInfo(ectx, methodInfo, attr);
                    entryPoints.Add(info);
                    if (_entryPointMap.ContainsKey(info.Name))
                    {
                        // Duplicate entry point name. We need to show a warning here.
                        // REVIEW: we will be able to do this once catalog becomes a part of env.
                        continue;
                    }

                    _entryPointMap[info.Name] = info;
                }

                // Scan for components.
                // First scan ourself, and then all nested types, for component info.
                ScanForComponents(ectx, type);
                foreach (var nestedType in type.GetTypeInfo().GetNestedTypes())
                    ScanForComponents(ectx, nestedType);
            }
            _entryPoints = entryPoints.ToArray();
        }

        private bool ScanForComponents(IExceptionContext ectx, Type nestedType)
        {
            var attr = nestedType.GetTypeInfo().GetCustomAttributes(typeof(TlcModule.ComponentAttribute), true).FirstOrDefault()
                as TlcModule.ComponentAttribute;
            if (attr == null)
                return false;

            bool found = false;
            foreach (var faceType in nestedType.GetInterfaces())
            {
                var faceAttr = faceType.GetTypeInfo().GetCustomAttributes(typeof(TlcModule.ComponentKindAttribute), false).FirstOrDefault()
                        as TlcModule.ComponentKindAttribute;
                if (faceAttr == null)
                    continue;

                if (!typeof(IComponentFactory).IsAssignableFrom(faceType))
                    throw ectx.Except("Component signature '{0}' doesn't inherit from '{1}'", faceType, typeof(IComponentFactory));

                try
                {
                    // In order to populate from JSON, we need to invoke the parameterless ctor. Testing that this is possible.
                    Activator.CreateInstance(nestedType);
                }
                catch (MissingMemberException ex)
                {
                    throw ectx.Except(ex, "Component type '{0}' doesn't have a default constructor", faceType);
                }

                var info = new ComponentInfo(ectx, faceType, faceAttr.Kind, nestedType, attr);
                var names = (info.Aliases ?? new string[0]).Concat(new[] { info.Name }).Distinct();
                _components.Add(info);

                foreach (var alias in names)
                {
                    var tag = $"{info.Kind}:{alias}";
                    if (_componentMap.ContainsKey(tag))
                    {
                        // Duplicate component name. We need to show a warning here.
                        // REVIEW: we will be able to do this once catalog becomes a part of env.
                        continue;
                    }
                    _componentMap[tag] = info;
                }
            }

            return found;
        }

        /// <summary>
        /// The valid names for the components and entry points must consist of letters, digits, underscores and dots, 
        /// and begin with a letter or digit.
        /// </summary>
        private static readonly Regex _nameRegex = new Regex(@"^\w[_\.\w]*$", RegexOptions.Compiled);
        private static bool IsValidName(string name)
        {
            Contracts.AssertValueOrNull(name);
            if (string.IsNullOrWhiteSpace(name))
                return false;
            return _nameRegex.IsMatch(name);
        }

        /// <summary>
        /// Create a module catalog (or reuse the one created before).
        /// </summary>
        /// <param name="ectx">The exception context to use to report errors while scanning the assemblies.</param>
        public static ModuleCatalog CreateInstance(IExceptionContext ectx)
        {
            Contracts.CheckValueOrNull(ectx);
#pragma warning disable 420 // volatile with Interlocked.CompareExchange.
            if (_instance == null)
                Interlocked.CompareExchange(ref _instance, new ModuleCatalog(ectx), null);
#pragma warning restore 420
            return _instance;
        }

        public bool TryFindEntryPoint(string name, out EntryPointInfo entryPoint)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            return _entryPointMap.TryGetValue(name, out entryPoint);
        }

        public bool TryFindComponent(string kind, string alias, out ComponentInfo component)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckNonEmpty(alias, nameof(alias));

            // Note that, if kind or alias contain the colon character, the kind:name 'tag' will contain more than one colon.
            // Since colon may not appear in any valid name, the dictionary lookup is guaranteed to fail.
            return _componentMap.TryGetValue($"{kind}:{alias}", out component);
        }

        public bool TryFindComponent(Type argumentType, out ComponentInfo component)
        {
            Contracts.CheckValue(argumentType, nameof(argumentType));

            component = _components.FirstOrDefault(x => x.ArgumentType == argumentType);
            return component != null;
        }

        public bool TryFindComponent(Type interfaceType, Type argumentType, out ComponentInfo component)
        {
            Contracts.CheckValue(interfaceType, nameof(interfaceType));
            Contracts.CheckParam(interfaceType.IsInterface, nameof(interfaceType), "Must be interface");
            Contracts.CheckValue(argumentType, nameof(argumentType));

            component = _components.FirstOrDefault(x => x.InterfaceType == interfaceType &&  x.ArgumentType == argumentType);
            return component != null;
        }

        public bool TryFindComponent(Type interfaceType, string alias, out ComponentInfo component)
        {
            Contracts.CheckValue(interfaceType, nameof(interfaceType));
            Contracts.CheckParam(interfaceType.IsInterface, nameof(interfaceType), "Must be interface");
            Contracts.CheckNonEmpty(alias, nameof(alias));
            component = _components.FirstOrDefault(x => x.InterfaceType == interfaceType && (x.Name == alias || (x.Aliases != null && x.Aliases.Contains(alias))));
            return component != null;
        }

        /// <summary>
        /// Akin to <see cref="TryFindComponent(Type, string, out ComponentInfo)"/>, except if the regular (case sensitive) comparison fails, it will
        /// attempt to back off to a case-insensitive comparison.
        /// </summary>
        public bool TryFindComponentCaseInsensitive(Type interfaceType, string alias, out ComponentInfo component)
        {
            Contracts.CheckValue(interfaceType, nameof(interfaceType));
            Contracts.CheckParam(interfaceType.IsInterface, nameof(interfaceType), "Must be interface");
            Contracts.CheckNonEmpty(alias, nameof(alias));
            if (TryFindComponent(interfaceType, alias, out component))
                return true;
            alias = alias.ToLowerInvariant();
            component = _components.FirstOrDefault(x => x.InterfaceType == interfaceType && (x.Name.ToLowerInvariant() == alias || AnyMatch(alias, x.Aliases)));
            return component != null;
        }

        private static bool AnyMatch(string name, string[] aliases)
        {
            if (aliases == null)
                return false;
            return aliases.Any(a => string.Equals(name, a, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Returns all valid component kinds.
        /// </summary>
        public IEnumerable<string> GetAllComponentKinds()
        {
            return _components.Select(x => x.Kind).Distinct().OrderBy(x => x);
        }

        /// <summary>
        /// Returns all components of the specified kind.
        /// </summary>
        public IEnumerable<ComponentInfo> GetAllComponents(string kind)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(IsValidName(kind), nameof(kind), "Invalid component kind");
            return _components.Where(x => x.Kind == kind).OrderBy(x => x.Name);
        }

        /// <summary>
        /// Returns all components that implement the specified interface.
        /// </summary>
        public IEnumerable<ComponentInfo> GetAllComponents(Type interfaceType)
        {
            Contracts.CheckValue(interfaceType, nameof(interfaceType));
            return _components.Where(x => x.InterfaceType == interfaceType).OrderBy(x => x.Name);
        }

        public bool TryGetComponentKind(Type signatureType, out string kind)
        {
            Contracts.CheckValue(signatureType, nameof(signatureType));
            // REVIEW: replace with a dictionary lookup.

            var faceAttr = signatureType.GetTypeInfo().GetCustomAttributes(typeof(TlcModule.ComponentKindAttribute), false).FirstOrDefault()
                    as TlcModule.ComponentKindAttribute;
            kind = faceAttr == null ? null : faceAttr.Kind;
            return faceAttr != null;
        }

        public bool TryGetComponentShortName(Type type, out string name)
        {
            ComponentInfo component;
            if (!TryFindComponent(type, out component))
            {
                name = null;
                return false;
            }

            name = component.Aliases != null && component.Aliases.Length > 0 ? component.Aliases[0] : component.Name;
            return true;
        }
    }
}
