// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Common signature type with no extra parameters.
    /// </summary>
    public delegate void SignatureDefault();

    [AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
    public sealed class LoadableClassAttribute : LoadableClassAttributeBase
    {
        /// <summary>
        /// Assembly attribute used to specify that a class is loadable by a machine learning
        /// host enviroment, such as TLC
        /// </summary>
        /// <param name="instType">The class type that is loadable</param>
        /// <param name="argType">The argument type that the constructor takes (may be null)</param>
        /// <param name="sigType">The signature of the constructor of this class (in addition to the arguments parameter)</param>
        /// <param name="userName">The name to use when presenting a list to users</param>
        /// <param name="loadNames">The names that can be used to load the class, for example, from a command line</param>
        public LoadableClassAttribute(Type instType, Type argType, Type sigType, string userName, params string[] loadNames)
            : base(null, instType, instType, argType, new[] { sigType }, userName, loadNames)
        {
        }

        /// <summary>
        /// Assembly attribute used to specify that a class is loadable by a machine learning
        /// host enviroment, such as TLC
        /// </summary>
        /// <param name="instType">The class type that is loadable</param>
        /// <param name="loaderType">The class type that contains the construction method</param>
        /// <param name="argType">The argument type that the constructor takes (may be null)</param>
        /// <param name="sigType">The signature of the constructor of this class (in addition to the arguments parameter)</param>
        /// <param name="userName">The name to use when presenting a list to users</param>
        /// <param name="loadNames">The names that can be used to load the class, for example, from a command line</param>
        public LoadableClassAttribute(Type instType, Type loaderType, Type argType, Type sigType, string userName, params string[] loadNames)
            : base(null, instType, loaderType, argType, new[] { sigType }, userName, loadNames)
        {
        }

        public LoadableClassAttribute(Type instType, Type argType, Type[] sigTypes, string userName, params string[] loadNames)
            : base(null, instType, instType, argType, sigTypes, userName, loadNames)
        {
        }

        public LoadableClassAttribute(Type instType, Type loaderType, Type argType, Type[] sigTypes, string userName, params string[] loadNames)
            : base(null, instType, loaderType, argType, sigTypes, userName, loadNames)
        {
        }

        /// <summary>
        /// Assembly attribute used to specify that a class is loadable by a machine learning
        /// host enviroment, such as TLC
        /// </summary>
        /// <param name="summary">The description summary of the class type</param>
        /// <param name="instType">The class type that is loadable</param>
        /// <param name="argType">The argument type that the constructor takes (may be null)</param>
        /// <param name="sigType">The signature of the constructor of this class (in addition to the arguments parameter)</param>
        /// <param name="userName">The name to use when presenting a list to users</param>
        /// <param name="loadNames">The names that can be used to load the class, for example, from a command line</param>
        public LoadableClassAttribute(string summary, Type instType, Type argType, Type sigType, string userName, params string[] loadNames)
            : base(summary, instType, instType, argType, new[] { sigType }, userName, loadNames)
        {
        }

        /// <summary>
        /// Assembly attribute used to specify that a class is loadable by a machine learning
        /// host enviroment, such as TLC
        /// </summary>
        /// <param name="summary">The description summary of the class type</param>
        /// <param name="instType">The class type that is loadable</param>
        /// <param name="loaderType">The class type that contains the construction method</param>
        /// <param name="argType">The argument type that the constructor takes (may be null)</param>
        /// <param name="sigType">The signature of the constructor of this class (in addition to the arguments parameter)</param>
        /// <param name="userName">The name to use when presenting a list to users</param>
        /// <param name="loadNames">The names that can be used to load the class, for example, from a command line</param>
        public LoadableClassAttribute(string summary, Type instType, Type loaderType, Type argType, Type sigType, string userName, params string[] loadNames)
            : base(summary, instType, loaderType, argType, new[] { sigType }, userName, loadNames)
        {
        }

        public LoadableClassAttribute(string summary, Type instType, Type argType, Type[] sigTypes, string userName, params string[] loadNames)
            : base(summary, instType, instType, argType, sigTypes, userName, loadNames)
        {
        }

        public LoadableClassAttribute(string summary, Type instType, Type loaderType, Type argType, Type[] sigTypes, string userName, params string[] loadNames)
            : base(summary, instType, loaderType, argType, sigTypes, userName, loadNames)
        {
        }
    }

    public abstract class LoadableClassAttributeBase : Attribute
    {
        // Note: these properties have private setters to make attribute parsing easier - the values
        // are all guaranteed to be in the ConstructorArguments of the CustomAttributeData
        // (no named arguments).

        /// <summary>
        /// The type that is created/loaded.
        /// </summary>
        public Type InstanceType { get; private set; }

        /// <summary>
        /// The type that contains the construction method, whether static Instance property,
        /// static Create method, or constructor. Of course, a constructor is only permissible if
        /// this type derives from InstanceType. This defaults to the same as InstanceType.
        /// </summary>
        public Type LoaderType { get; private set; }

        /// <summary>
        /// The command line arguments object type. This should be null if there isn't one.
        /// </summary>
        public Type ArgType { get; private set; }

        /// <summary>
        /// This indicates the extra parameter types. It must be a delegate type. The return type should be void.
        /// The parameter types of the SigType delegate should NOT include the ArgType.
        /// </summary>
        public Type[] SigTypes { get; private set; }

        /// <summary>
        /// Note that CtorTypes includes the ArgType (if there is one), and the parameter types of the SigType.
        /// </summary>
        public Type[] CtorTypes { get; private set; }

        /// <summary>
        /// The description summary of the class type.
        /// </summary>
        public string Summary { get; private set; }

        /// <summary>
        /// UserName may be null or empty indicating that it should be hidden in UI.
        /// </summary>
        public string UserName { get; private set; }
        public string[] LoadNames { get; private set; }

        // REVIEW: This is out of step with the remainder of the class. However, my opinion is that the
        // LoadableClassAttribute class's design is worth reconsideration: having so many Type and string arguments
        // be defined *without names* in a constructor has led to enormous confusion.

        // REVIEW: Presumably it would be beneficial to have multiple documents.

        /// <summary>
        /// This should indicate a path within the <code>doc/public</code> directory next to the TLC
        /// solution, where the documentation lies. This value will be used as part of a URL, so,
        /// the path separator should be phrased as '/' forward slashes rather than backslashes.</summary>
        public string DocName { get; set; }

        protected LoadableClassAttributeBase(string summary, Type instType, Type loaderType, Type argType, Type[] sigTypes, string userName, params string[] loadNames)
        {
            Contracts.CheckValueOrNull(summary);
            Contracts.CheckValue(instType, nameof(instType));
            Contracts.CheckValue(loaderType, nameof(loaderType));
            Contracts.CheckNonEmpty(sigTypes, nameof(sigTypes));

            if (Utils.Size(loadNames) == 0)
                loadNames = new string[] { userName };

            if (loadNames.Any(s => string.IsNullOrWhiteSpace(s)))
                throw Contracts.ExceptEmpty(nameof(loadNames), "LoadableClass loadName parameter can't be empty");

            var sigType = sigTypes[0];
            Contracts.CheckValue(sigType, nameof(sigTypes));
            Type[] types;
            Contracts.CheckParam(sigType.BaseType == typeof(System.MulticastDelegate), nameof(sigTypes), "LoadableClass signature type must be a delegate type");

            var meth = sigType.GetMethod("Invoke");
            Contracts.CheckParam(meth != null, nameof(sigTypes), "LoadableClass signature type must be a delegate type");
            Contracts.CheckParam(meth.ReturnType == typeof(void), nameof(sigTypes), "LoadableClass signature type must be a delegate type with void return");

            var parms = meth.GetParameters();
            int itypeBase = 0;

            if (argType != null)
            {
                types = new Type[1 + parms.Length];
                types[itypeBase++] = argType;
            }
            else if (parms.Length > 0)
                types = new Type[parms.Length];
            else
                types = Type.EmptyTypes;

            for (int itype = 0; itype < parms.Length; itype++)
            {
                var parm = parms[itype];
                if ((parm.Attributes & (ParameterAttributes.Out | ParameterAttributes.Retval)) != 0)
                    throw Contracts.Except("Invalid signature parameter attributes");
                types[itypeBase + itype] = parm.ParameterType;
            }

            for (int i = 1; i < sigTypes.Length; i++)
            {
                sigType = sigTypes[i];
                Contracts.CheckValue(sigType, nameof(sigTypes));

                Contracts.Check(sigType.BaseType == typeof(System.MulticastDelegate), "LoadableClass signature type must be a delegate type");

                meth = sigType.GetMethod("Invoke");
                Contracts.CheckParam(meth != null, nameof(sigTypes), "LoadableClass signature type must be a delegate type");
                Contracts.CheckParam(meth.ReturnType == typeof(void), nameof(sigTypes), "LoadableClass signature type must be a delegate type with void return");
                parms = meth.GetParameters();
                Contracts.CheckParam(parms.Length + itypeBase == types.Length, nameof(sigTypes), "LoadableClass signatures must have the same number of parameters");
                for (int itype = 0; itype < parms.Length; itype++)
                {
                    var parm = parms[itype];
                    if ((parm.Attributes & (ParameterAttributes.Out | ParameterAttributes.Retval)) != 0)
                        throw Contracts.ExceptParam(nameof(sigTypes), "Invalid signature parameter attributes");
                    Contracts.CheckParam(types[itypeBase + itype] == parm.ParameterType, nameof(sigTypes),
                        "LoadableClass signatures must have the same set of parameters");
                }
            }

            InstanceType = instType;
            LoaderType = loaderType;
            ArgType = argType;
            SigTypes = sigTypes;
            CtorTypes = types;
            Summary = summary;
            UserName = userName;
            LoadNames = loadNames;
        }
    }
}
