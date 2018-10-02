// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;

// REVIEW: Determine ideal namespace.
namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// This catalogs instantiatable components (aka, loadable classes). Components are registered via
    /// a descendant of <see cref="LoadableClassAttributeBase"/>, identifying the names and signature types under which the component
    /// type should be registered. Signatures are delegate types that return void and specify that parameter
    /// types for component instantiation. Each component may also specify an "arguments object" that should
    /// be provided at instantiation time.
    /// </summary>
    public sealed class ComponentCatalog
    {
        internal ComponentCatalog()
        {
            _lock = new object();
            _cachedAssemblies = new HashSet<string>();
            _classesByKey = new Dictionary<LoadableClassInfo.Key, LoadableClassInfo>();
            _classes = new List<LoadableClassInfo>();
            _signatures = new Dictionary<Type, bool>();

            _entryPoints = new List<EntryPointInfo>();
            _entryPointMap = new Dictionary<string, EntryPointInfo>();
            _componentMap = new Dictionary<string, ComponentInfo>();
            _components = new List<ComponentInfo>();
        }

        /// <summary>
        /// Provides information on an instantiatable component, aka, loadable class.
        /// </summary>
        public sealed class LoadableClassInfo
        {
            /// <summary>
            /// Used for dictionary lookup based on signature and name.
            /// </summary>
            internal struct Key : IEquatable<Key>
            {
                public readonly string Name;
                public readonly Type Signature;

                public Key(string name, Type sig)
                {
                    Name = name;
                    Signature = sig;
                }

                public override int GetHashCode()
                {
                    return Hashing.CombinedHash(Name.GetHashCode(), Signature.GetHashCode());
                }

                public override bool Equals(object obj)
                {
                    return obj is Key && Equals((Key)obj);
                }

                public bool Equals(Key other)
                {
                    return other.Name == Name && other.Signature == Signature;
                }
            }

            /// <summary>
            /// Count of component construction arguments, NOT including the arguments object (if there is one).
            /// This matches the number of arguments for the signature type delegate(s).
            /// </summary>
            internal int ExtraArgCount => ArgType == null ? CtorTypes.Length : CtorTypes.Length - 1;

            public Type Type { get; }

            /// <summary>
            /// The type that contains the construction method, whether static Instance property,
            /// static Create method, or constructor.
            /// </summary>
            public Type LoaderType { get; }

            public IReadOnlyList<Type> SignatureTypes { get; }

            /// <summary>
            /// Summary of the component.
            /// </summary>
            public string Summary { get; }

            /// <summary>
            /// UserName may be null or empty, indicating that it should be hidden in UI.
            /// </summary>
            public string UserName { get; }

            /// <summary>
            /// Whether this is a "hidden" component, that generally shouldn't be displayed
            /// to users.
            /// </summary>
            public bool IsHidden => string.IsNullOrWhiteSpace(UserName);

            /// <summary>
            /// All load names. The first is the default.
            /// </summary>
            public IReadOnlyList<string> LoadNames { get; }

            /// <summary>
            /// The static property that returns an instance of this loadable class.
            /// This creation method does not support an arguments class.
            /// Only one of Ctor, Create and InstanceGetter can be non-null.
            /// </summary>
            public MethodInfo InstanceGetter { get; }

            /// <summary>
            /// The constructor to create an instance of this loadable class.
            /// This creation method supports an arguments class.
            /// Only one of Ctor, Create and InstanceGetter can be non-null.
            /// </summary>
            public ConstructorInfo Constructor { get; }

            /// <summary>
            /// The static method that creates an instance of this loadable class.
            /// This creation method supports an arguments class.
            /// Only one of Ctor, Create and InstanceGetter can be non-null.
            /// </summary>
            public MethodInfo CreateMethod { get; }

            public bool RequireEnvironment { get; }

            /// <summary>
            /// A name of an embedded resource containing documentation for this
            /// loadable class. This is non-null only in the event that we have
            /// verified the assembly of <see cref="LoaderType"/> actually contains
            /// this resource.
            /// </summary>
            public string DocName { get; }

            /// <summary>
            /// The type that contains the arguments to the component.
            /// </summary>
            public Type ArgType { get; }

            private Type[] CtorTypes { get; }

            internal LoadableClassInfo(LoadableClassAttributeBase attr, MethodInfo getter, ConstructorInfo ctor, MethodInfo create, bool requireEnvironment)
            {
                Contracts.AssertValue(attr);
                Contracts.AssertValue(attr.InstanceType);
                Contracts.AssertValue(attr.LoaderType);
                Contracts.AssertValueOrNull(attr.Summary);
                Contracts.AssertValueOrNull(attr.DocName);
                Contracts.AssertValueOrNull(attr.UserName);
                Contracts.AssertNonEmpty(attr.LoadNames);
                Contracts.Assert(getter == null || Utils.Size(attr.CtorTypes) == 0);

                Type = attr.InstanceType;
                LoaderType = attr.LoaderType;
                Summary = attr.Summary;
                UserName = attr.UserName;
                LoadNames = attr.LoadNames.AsReadOnly();

                if (getter != null)
                    InstanceGetter = getter;
                else if (ctor != null)
                    Constructor = ctor;
                else if (create != null)
                    CreateMethod = create;
                ArgType = attr.ArgType;
                SignatureTypes = attr.SigTypes.AsReadOnly();
                CtorTypes = attr.CtorTypes ?? Type.EmptyTypes;
                RequireEnvironment = requireEnvironment;

                if (!string.IsNullOrWhiteSpace(attr.DocName))
                    DocName = attr.DocName;

                Contracts.Assert(ArgType == null || CtorTypes.Length > 0 && CtorTypes[0] == ArgType);
            }

            internal object CreateInstanceCore(object[] ctorArgs)
            {
                Contracts.Assert(Utils.Size(ctorArgs) == CtorTypes.Length + ((RequireEnvironment) ? 1 : 0));

                if (InstanceGetter != null)
                {
                    Contracts.Assert(Utils.Size(ctorArgs) == 0);
                    return InstanceGetter.Invoke(null, null);
                }
                if (Constructor != null)
                    return Constructor.Invoke(ctorArgs);
                if (CreateMethod != null)
                    return CreateMethod.Invoke(null, ctorArgs);
                throw Contracts.Except("Can't instantiate class '{0}'", Type.Name);
            }

            /// <summary>
            /// Create an instance, given the arguments object and arguments to the signature delegate.
            /// The args should be non-null iff ArgType is non-null. The length of the extra array should
            /// match the number of paramters for the signature delgate. When that number is zero, extra
            /// may be null.
            /// </summary>
            public object CreateInstance(IHostEnvironment env, object args, object[] extra)
            {
                Contracts.CheckValue(env, nameof(env));
                env.Check((ArgType != null) == (args != null));
                env.Check(Utils.Size(extra) == ExtraArgCount);

                List<object> prefix = new List<object>();
                if (RequireEnvironment)
                    prefix.Add(env);
                if (ArgType != null)
                    prefix.Add(args);
                var values = Utils.Concat(prefix.ToArray(), extra);
                return CreateInstanceCore(values);
            }

            /// <summary>
            /// Create an instance, given the arguments object and arguments to the signature delegate.
            /// The args should be non-null iff ArgType is non-null. The length of the extra array should
            /// match the number of paramters for the signature delgate. When that number is zero, extra
            /// may be null.
            /// </summary>
            public TRes CreateInstance<TRes>(IHostEnvironment env, object args, object[] extra)
            {
                if (!typeof(TRes).IsAssignableFrom(Type))
                    throw Contracts.Except("Loadable class '{0}' does not derive from '{1}'", LoadNames[0], typeof(TRes).FullName);
                return (TRes)CreateInstance(env, args, extra);
            }

            /// <summary>
            /// Create an instance with default arguments.
            /// </summary>
            public TRes CreateInstance<TRes>(IHostEnvironment env)
            {
                if (!typeof(TRes).IsAssignableFrom(Type))
                    throw Contracts.Except("Loadable class '{0}' does not derive from '{1}'", LoadNames[0], typeof(TRes).FullName);
                return (TRes)CreateInstance(env, CreateArguments(), null);
            }

            /// <summary>
            /// If <see cref="ArgType"/> is not null, returns a new default instance of <see cref="ArgType"/>.
            /// Otherwise, returns null.
            /// </summary>
            public object CreateArguments()
            {
                if (ArgType == null)
                    return null;

                var ctor = ArgType.GetConstructor(Type.EmptyTypes);
                if (ctor == null)
                {
                    throw Contracts.Except("Loadable class '{0}' has ArgType '{1}', which has no suitable constructor",
                        UserName, ArgType);
                }

                return ctor.Invoke(null);
            }
        }

        /// <summary>
        /// A description of a single entry point.
        /// </summary>
        public sealed class EntryPointInfo
        {
            public readonly string Name;
            public readonly string Description;
            public readonly string ShortName;
            public readonly string FriendlyName;
            public readonly string[] XmlInclude;
            public readonly MethodInfo Method;
            public readonly Type InputType;
            public readonly Type OutputType;
            public readonly Type[] InputKinds;
            public readonly Type[] OutputKinds;
            public readonly ObsoleteAttribute ObsoleteAttribute;

            internal EntryPointInfo(MethodInfo method,
                TlcModule.EntryPointAttribute attribute, ObsoleteAttribute obsoleteAttribute)
            {
                Contracts.AssertValue(method);
                Contracts.AssertValue(attribute);

                Name = attribute.Name ?? string.Join(".", method.DeclaringType.Name, method.Name);
                Description = attribute.Desc;
                Method = method;
                ShortName = attribute.ShortName;
                FriendlyName = attribute.UserName;
                XmlInclude = attribute.XmlInclude;
                ObsoleteAttribute = obsoleteAttribute;

                // There are supposed to be 2 parameters, env and input for non-macro nodes.
                // Macro nodes have a 3rd parameter, the entry point node.
                var parameters = method.GetParameters();
                if (parameters.Length != 2 && parameters.Length != 3)
                    throw Contracts.Except("Method '{0}' has {1} parameters, but must have 2 or 3", method.Name, parameters.Length);
                if (parameters[0].ParameterType != typeof(IHostEnvironment))
                    throw Contracts.Except("Method '{0}', 1st parameter is {1}, but must be IHostEnvironment", method.Name, parameters[0].ParameterType);
                InputType = parameters[1].ParameterType;
                var outputType = method.ReturnType;
                if (!outputType.IsClass)
                    throw Contracts.Except("Method '{0}' returns {1}, but must return a class", method.Name, outputType);
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
        /// The 'component' is a non-standalone building block that is used to parametrize entry points or other ML.NET components.
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

            internal ComponentInfo(Type interfaceType, string kind, Type argumentType, TlcModule.ComponentAttribute attribute)
            {
                Contracts.AssertValue(interfaceType);
                Contracts.AssertNonEmpty(kind);
                Contracts.AssertValue(argumentType);
                Contracts.AssertValue(attribute);

                Name = attribute.Name;
                Description = attribute.Desc;
                if (string.IsNullOrWhiteSpace(attribute.FriendlyName))
                    FriendlyName = Name;
                else
                    FriendlyName = attribute.FriendlyName;

                Kind = kind;
                if (!IsValidName(Kind))
                    throw Contracts.Except("Invalid component kind: '{0}'", Kind);

                Aliases = attribute.Aliases;
                if (!IsValidName(Name))
                    throw Contracts.Except("Component name '{0}' is not valid.", Name);

                if (Aliases != null && Aliases.Any(x => !IsValidName(x)))
                    throw Contracts.Except("Component '{0}' has an invalid alias '{1}'", Name, Aliases.First(x => !IsValidName(x)));

                if (!typeof(IComponentFactory).IsAssignableFrom(argumentType))
                    throw Contracts.Except("Component '{0}' must inherit from IComponentFactory", argumentType);

                ArgumentType = argumentType;
                InterfaceType = interfaceType;
            }
        }

        // This lock protects adding to the below collections.
        private readonly object _lock;
        private readonly HashSet<string> _cachedAssemblies;

        // Map from key/name to loadable class. Note that the same ClassInfo may appear
        // multiple times. For the set of unique infos, use _classes.
        private readonly Dictionary<LoadableClassInfo.Key, LoadableClassInfo> _classesByKey;

        // The unique ClassInfos and Signatures.
        private readonly List<LoadableClassInfo> _classes;
        private readonly Dictionary<Type, bool> _signatures;

        private readonly List<EntryPointInfo> _entryPoints;
        private readonly Dictionary<string, EntryPointInfo> _entryPointMap;

        private readonly List<ComponentInfo> _components;
        private readonly Dictionary<string, ComponentInfo> _componentMap;

        private static bool TryGetIniters(Type instType, Type loaderType, Type[] parmTypes,
            out MethodInfo getter, out ConstructorInfo ctor, out MethodInfo create, out bool requireEnvironment)
        {
            getter = null;
            ctor = null;
            create = null;
            requireEnvironment = false;
            var parmTypesWithEnv = Utils.Concat(new Type[1] { typeof(IHostEnvironment) }, parmTypes);
            if (Utils.Size(parmTypes) == 0 && (getter = FindInstanceGetter(instType, loaderType)) != null)
                return true;
            if (instType.IsAssignableFrom(loaderType) && (ctor = loaderType.GetConstructor(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, parmTypes ?? Type.EmptyTypes, null)) != null)
                return true;
            if (instType.IsAssignableFrom(loaderType) && (ctor = loaderType.GetConstructor(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, parmTypesWithEnv ?? Type.EmptyTypes, null)) != null)
            {
                requireEnvironment = true;
                return true;
            }
            if ((create = FindCreateMethod(instType, loaderType, parmTypes ?? Type.EmptyTypes)) != null)
                return true;
            if ((create = FindCreateMethod(instType, loaderType, parmTypesWithEnv ?? Type.EmptyTypes)) != null)
            {
                requireEnvironment = true;
                return true;
            }

            return false;
        }

        private void AddClass(LoadableClassInfo info, string[] loadNames, bool throwOnError)
        {
            _classes.Add(info);
            bool isEntryPoint = false;
            foreach (var sigType in info.SignatureTypes)
            {
                _signatures[sigType] = true;

                foreach (var name in loadNames)
                {
                    string nameCi = name.ToLowerInvariant();

                    var key = new LoadableClassInfo.Key(nameCi, sigType);
                    if (_classesByKey.TryGetValue(key, out var infoCur))
                    {
                        if (throwOnError)
                        {
                            throw Contracts.Except($"ComponentCatalog cannot map name '{name}' and SignatureType '{sigType}' to {info.Type.Name}, already mapped to {infoCur.Type.Name}.");
                        }
                    }
                    else
                    {
                        _classesByKey.Add(key, info);
                    }
                }

                if (sigType == typeof(SignatureEntryPointModule))
                {
                    isEntryPoint = true;
                }
            }

            if (isEntryPoint)
            {
                ScanForEntryPoints(info);
            }
        }

        private void ScanForEntryPoints(LoadableClassInfo info)
        {
            var type = info.LoaderType;

            // Scan for entry points.
            foreach (var methodInfo in type.GetMethods(BindingFlags.Static | BindingFlags.Public))
            {
                var attr = methodInfo.GetCustomAttributes(typeof(TlcModule.EntryPointAttribute), false).FirstOrDefault() as TlcModule.EntryPointAttribute;
                if (attr == null)
                    continue;

                var entryPointInfo = new EntryPointInfo(methodInfo, attr,
                    methodInfo.GetCustomAttributes(typeof(ObsoleteAttribute), false).FirstOrDefault() as ObsoleteAttribute);

                _entryPoints.Add(entryPointInfo);
                if (_entryPointMap.ContainsKey(entryPointInfo.Name))
                {
                    // Duplicate entry point name. We need to show a warning here.
                    // REVIEW: we will be able to do this once catalog becomes a part of env.
                    continue;
                }

                _entryPointMap[entryPointInfo.Name] = entryPointInfo;
            }

            // Scan for components.
            // First scan ourself, and then all nested types, for component info.
            ScanForComponents(type);
            foreach (var nestedType in type.GetTypeInfo().GetNestedTypes())
                ScanForComponents(nestedType);
        }

        private bool ScanForComponents(Type nestedType)
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
                    throw Contracts.Except("Component signature '{0}' doesn't inherit from '{1}'", faceType, typeof(IComponentFactory));

                try
                {
                    // In order to populate from JSON, we need to invoke the parameterless ctor. Testing that this is possible.
                    Activator.CreateInstance(nestedType);
                }
                catch (MissingMemberException ex)
                {
                    throw Contracts.Except(ex, "Component type '{0}' doesn't have a default constructor", faceType);
                }

                var info = new ComponentInfo(faceType, faceAttr.Kind, nestedType, attr);
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

        private static MethodInfo FindInstanceGetter(Type instType, Type loaderType)
        {
            // Look for a public static property named Instance of the correct type.
            var prop = loaderType.GetProperty("Instance", instType);
            if (prop == null)
                return null;
            if (prop.DeclaringType != loaderType)
                return null;
            var meth = prop.GetGetMethod(false);
            if (meth == null)
                return null;
            if (meth.ReturnType != instType)
                return null;
            if (!meth.IsPublic || !meth.IsStatic)
                return null;
            return meth;
        }

        private static MethodInfo FindCreateMethod(Type instType, Type loaderType, Type[] parmTypes)
        {
            var meth = loaderType.GetMethod("Create", BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy, null, parmTypes ?? Type.EmptyTypes, null);
            if (meth == null)
                return null;
            if (meth.DeclaringType != loaderType)
                return null;
            if (meth.ReturnType != instType)
                return null;
            if (!meth.IsStatic)
                return null;
            return meth;
        }

        /// <summary>
        /// Registers all the components in the specified assembly by looking for loadable classes
        /// and adding them to the catalog.
        /// </summary>
        /// <param name="assembly">
        /// The assembly to register.
        /// </param>
        /// <param name="throwOnError">
        /// true to throw an exception if there are errors with registering the components;
        /// false to skip any errors.
        /// </param>
        public void RegisterAssembly(Assembly assembly, bool throwOnError = true)
        {
            lock (_lock)
            {
                if (_cachedAssemblies.Add(assembly.FullName))
                {
                    foreach (LoadableClassAttributeBase attr in assembly.GetCustomAttributes(typeof(LoadableClassAttributeBase)))
                    {
                        MethodInfo getter = null;
                        ConstructorInfo ctor = null;
                        MethodInfo create = null;
                        bool requireEnvironment = false;
                        if (attr.InstanceType != typeof(void) && !TryGetIniters(attr.InstanceType, attr.LoaderType, attr.CtorTypes, out getter, out ctor, out create, out requireEnvironment))
                        {
                            if (throwOnError)
                            {
                                throw Contracts.Except(
                                    $"Can't instantiate loadable class '{attr.InstanceType.Name}' with name '{attr.LoadNames[0]}'");
                            }
                            Contracts.Assert(getter == null && ctor == null && create == null);
                        }
                        var info = new LoadableClassInfo(attr, getter, ctor, create, requireEnvironment);

                        AddClass(info, attr.LoadNames, throwOnError);
                    }
                }
            }
        }

        /// <summary>
        /// Return an array containing information for all instantiatable components.
        /// If provided, the given set of assemblies is loaded first.
        /// </summary>
        public LoadableClassInfo[] GetAllClasses()
        {
            return _classes.ToArray();
        }

        /// <summary>
        /// Return an array containing information for instantiatable components with the given
        /// signature and base type. If provided, the given set of assemblies is loaded first.
        /// </summary>
        public LoadableClassInfo[] GetAllDerivedClasses(Type typeBase, Type typeSig)
        {
            Contracts.CheckValue(typeBase, nameof(typeBase));
            Contracts.CheckValueOrNull(typeSig);

            // Apply the default.
            if (typeSig == null)
                typeSig = typeof(SignatureDefault);

            return _classes
                .Where(info => info.SignatureTypes.Contains(typeSig) && typeBase.IsAssignableFrom(info.Type))
                .ToArray();
        }

        /// <summary>
        /// Return an array containing all the known signature types. If provided, the given set of assemblies
        /// is loaded first.
        /// </summary>
        public Type[] GetAllSignatureTypes()
        {
            return _signatures.Select(kvp => kvp.Key).ToArray();
        }

        /// <summary>
        /// Returns a string name for a given signature type.
        /// </summary>
        public static string SignatureToString(Type sig)
        {
            Contracts.CheckValue(sig, nameof(sig));
            Contracts.CheckParam(sig.BaseType == typeof(MulticastDelegate), nameof(sig), "Must be a delegate type");
            string kind = sig.Name;
            if (kind.Length > "Signature".Length && kind.StartsWith("Signature"))
                kind = kind.Substring("Signature".Length);
            return kind;
        }

        private LoadableClassInfo FindClassCore(LoadableClassInfo.Key key)
        {
            LoadableClassInfo info;
            if (_classesByKey.TryGetValue(key, out info))
                return info;

            return null;
        }

        public LoadableClassInfo[] FindLoadableClasses(string name)
        {
            name = name.ToLowerInvariant().Trim();

            var res = _classes
                .Where(ci => ci.LoadNames.Select(n => n.ToLowerInvariant().Trim()).Contains(name))
                .ToArray();
            return res;
        }

        public LoadableClassInfo[] FindLoadableClasses<TSig>()
        {
            return _classes
                .Where(ci => ci.SignatureTypes.Contains(typeof(TSig)))
                .ToArray();
        }

        public LoadableClassInfo[] FindLoadableClasses<TArgs, TSig>()
        {
            // REVIEW: this and above methods perform a linear search over all the loadable classes.
            // On 6/15/2015, TLC release build contained 431 of them, so adding extra lookups looks unnecessary at this time.
            return _classes
                .Where(ci => ci.ArgType == typeof(TArgs) && ci.SignatureTypes.Contains(typeof(TSig)))
                .ToArray();
        }

        public LoadableClassInfo GetLoadableClassInfo<TSig>(string loadName)
        {
            return GetLoadableClassInfo(loadName, typeof(TSig));
        }

        public LoadableClassInfo GetLoadableClassInfo(string loadName, Type signatureType)
        {
            Contracts.CheckParam(signatureType.BaseType == typeof(MulticastDelegate), nameof(signatureType), "signatureType must be a delegate type");
            Contracts.CheckValueOrNull(loadName);
            loadName = (loadName ?? "").ToLowerInvariant().Trim();
            return FindClassCore(new LoadableClassInfo.Key(loadName, signatureType));
        }

        /// <summary>
        /// Get all registered entry points.
        /// </summary>
        public IEnumerable<EntryPointInfo> AllEntryPoints()
        {
            return _entryPoints.AsEnumerable();
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

            component = _components.FirstOrDefault(x => x.InterfaceType == interfaceType && x.ArgumentType == argumentType);
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
        /// Create an instance of the indicated component with the given extra parameters.
        /// </summary>
        public static TRes CreateInstance<TRes>(IHostEnvironment env, Type signatureType, string name, string options, params object[] extra)
            where TRes : class
        {
            TRes result;
            if (TryCreateInstance(env, signatureType, out result, name, options, extra))
                return result;
            throw Contracts.Except("Unknown loadable class: {0}", name).MarkSensitive(MessageSensitivity.None);
        }

        /// <summary>
        /// Try to create an instance of the indicated component and settings with the given extra parameters.
        /// If there is no such component in the catalog, returns false. Any other error results in an exception.
        /// </summary>
        public static bool TryCreateInstance<TRes, TSig>(IHostEnvironment env, out TRes result, string name, string options, params object[] extra)
            where TRes : class
        {
            return TryCreateInstance<TRes>(env, typeof(TSig), out result, name, options, extra);
        }

        private static bool TryCreateInstance<TRes>(IHostEnvironment env, Type signatureType, out TRes result, string name, string options, params object[] extra)
            where TRes : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.Check(signatureType.BaseType == typeof(MulticastDelegate));
            env.CheckValueOrNull(name);

            string nameLower = (name ?? "").ToLowerInvariant().Trim();
            LoadableClassInfo info = env.ComponentCatalog.FindClassCore(new LoadableClassInfo.Key(nameLower, signatureType));
            if (info == null)
            {
                result = null;
                return false;
            }

            if (!typeof(TRes).IsAssignableFrom(info.Type))
                throw env.Except("Loadable class '{0}' does not derive from '{1}'", name, typeof(TRes).FullName);

            int carg = Utils.Size(extra);

            if (info.ExtraArgCount != carg)
            {
                throw env.Except(
                    "Wrong number of extra parameters for loadable class '{0}', need '{1}', given '{2}'",
                    name, info.ExtraArgCount, carg);
            }

            if (info.ArgType == null)
            {
                if (!string.IsNullOrEmpty(options))
                    throw env.Except("Loadable class '{0}' doesn't support settings", name);
                result = (TRes)info.CreateInstance(env, null, extra);
                return true;
            }

            object args = info.CreateArguments();
            if (args == null)
                throw Contracts.Except("Can't instantiate arguments object '{0}' for '{1}'", info.ArgType.Name, name);

            ParseArguments(env, args, options, name);
            result = (TRes)info.CreateInstance(env, args, extra);
            return true;
        }

        /// <summary>
        /// Parses arguments using CmdParser. If there's a problem, it throws an InvalidOperationException,
        /// with a message giving usage.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="args">The argument object</param>
        /// <param name="settings">The settings string</param>
        /// <param name="name">The name is used for error reporting only</param>
        private static void ParseArguments(IHostEnvironment env, object args, string settings, string name)
        {
            Contracts.AssertValue(args);
            Contracts.AssertNonEmpty(name);

            if (string.IsNullOrWhiteSpace(settings))
                return;

            string errorMsg = null;
            try
            {
                string err = null;
                string helpText;
                if (!CmdParser.ParseArguments(env, settings, args, e => { err = err ?? e; }, out helpText))
                    errorMsg = err + Environment.NewLine + "Usage For '" + name + "':" + Environment.NewLine + helpText;
            }
            catch (Exception e)
            {
                Contracts.Assert(false);
                throw Contracts.Except(e, "Unexpected exception thrown while parsing: " + e.Message);
            }

            if (errorMsg != null)
                throw Contracts.Except(errorMsg);
        }
    }
}
