// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System.Runtime.Serialization.Formatters.Binary;

[assembly: LoadableClass(SerializableLambdaTransform.Summary, typeof(ITransformTemplate), typeof(SerializableLambdaTransform), null,
    typeof(SignatureLoadDataTransform), "", SerializableLambdaTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Api
{
    internal static class SerializableLambdaTransform
    {
        // This static class exists so that we can expose the Create loader delegate without having
        // to specify bogus type arguments on the generic class.

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "USERMAPX",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public const string LoaderSignature = "UserLambdaMapTransform";
        public const string Summary = "Allows the definition of convenient user defined transforms";

        /// <summary>
        /// Creates an instance of the transform from a context.
        /// </summary>
        public static ITransformTemplate Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);
            host.CheckValue(ctx, nameof(ctx));

            host.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: Number of bytes the load method was serialized to
            // byte[n]: The serialized load method info
            // <arbitrary>: Arbitrary bytes saved by the save action

            var loadMethodBytes = ctx.Reader.ReadByteArray();
            host.CheckDecode(Utils.Size(loadMethodBytes) > 0);
            // Attempt to reconstruct the method.
            Exception error;
            var loadFunc = DeserializeStaticDelegateOrNull(host, loadMethodBytes, out error);
            if (loadFunc == null)
            {
                host.AssertValue(error);
                throw error;
            }

            var bytes = ctx.Reader.ReadByteArray() ?? new byte[0];

            using (var ms = new MemoryStream(bytes))
            using (var reader = new BinaryReader(ms))
            {
                var result = loadFunc(reader, env, input);
                env.Check(result != null, "Load method returned null");
                return result;
            }
        }

        /// <summary>
        /// Given a single item function that should be a static method, this builds a serialized version of
        /// that method that should be enough to "recover" it, assuming it is a "recoverable" method (recoverable
        /// here is a loose definition, meaning that <see cref="DeserializeStaticDelegateOrNull"/> is capable
        /// of creating it, which includes among other things that it's static, non-lambda, accessible to
        /// this assembly, etc.). 
        /// </summary>
        /// <param name="func">The method that should be "recoverable"</param>
        /// <returns>A string array describing the input method</returns>
        public static byte[] GetSerializedStaticDelegate(LambdaTransform.LoadDelegate func)
        {
            Contracts.CheckValue(func, nameof(func));
            Contracts.CheckParam(func.Target == null, nameof(func), "The load delegate must be static");
            Contracts.CheckParam(Utils.Size(func.GetInvocationList()) <= 1, nameof(func),
                "The load delegate must not be a multicast delegate");

            var meth = func.GetMethodInfo();
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
#if CORECLR
                var m = new CoreHackMethodInfo();
                m.AssemblyName = meth.Module.Assembly.FullName;
                m.MethodName = meth.Name;
                m.ClassName = meth.DeclaringType.ToString();
                formatter.Serialize(ms, m);
#else
                formatter.Serialize(ms, meth);
#endif
                var result = ms.ToArray();
                // I assume it must be impossible to serialize in 0 bytes.
                Contracts.Assert(Utils.Size(result) > 0);
                return result;
            }
        }

        /// <summary>
        /// This is essentially the inverse function to <see cref="GetSerializedStaticDelegate"/>. If the function
        /// is not recoverable for any reason, this will return <c>null</c>, and the error parameter will be set.
        /// </summary>
        /// <param name="ectx">Exception context.</param>
        /// <param name="serialized">The serialized bytes, as returned by <see cref="GetSerializedStaticDelegate"/></param>
        /// <param name="inner">An exception the caller may raise as an inner exception if the return value is
        /// <c>null</c>, else, this itself will be <c>null</c></param>
        /// <returns>The recovered function wrapping the recovered method, or <c>null</c> if it could not
        /// be created, for some reason</returns>
        public static LambdaTransform.LoadDelegate DeserializeStaticDelegateOrNull(IExceptionContext ectx, byte[] serialized, out Exception inner)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertNonEmpty(serialized);
            MethodInfo info = null;
            try
            {
                using (var ms = new MemoryStream(serialized, false))
                {
#if CORECLR
                    var formatter = new BinaryFormatter();
                    object obj = formatter.Deserialize(ms);
                    var hack = obj as CoreHackMethodInfo;
                    var assembly = Assembly.Load(new AssemblyName(hack.AssemblyName));
                    Type t = assembly.GetType(hack.ClassName);
                    info = t.GetTypeInfo().GetDeclaredMethod(hack.MethodName);
#else
                    var formatter = new BinaryFormatter();
                    object obj = formatter.Deserialize(ms);
                    info = obj as MethodInfo;
#endif
                }
            }
            catch (Exception e)
            {
                inner = ectx.ExceptDecode(e, "Failed to deserialize a .NET object");
                return null;
            }
            // Either it's not the right type, or obj itself may be null. Either way we have an error.
            if (info == null)
            {
                inner = ectx.ExceptDecode("Failed to deserialize the method");
                return null;
            }
            if (!info.IsStatic)
            {
                inner = ectx.ExceptDecode("Deserialized method is not static");
                return null;
            }
            try
            {
                var del = info.CreateDelegate(typeof(LambdaTransform.LoadDelegate));
                inner = null;
                return (LambdaTransform.LoadDelegate)del;
            }
            catch (Exception)
            {
                inner = ectx.ExceptDecode("Deserialized method has wrong signature");
                return null;
            }
        }
#if CORECLR
        [Serializable]
        internal sealed class CoreHackMethodInfo
        {
            public string MethodName;
            public string AssemblyName;
            public string ClassName;
        }
#endif
    }

}