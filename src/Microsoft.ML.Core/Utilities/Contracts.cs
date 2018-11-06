// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

// We want the conditional code in this file to always be available to
// client assemblies that might be DEBUG versions. That is, if someone uses
// the release build of this assembly to build a DEBUG version of their code,
// we want Contracts.Assert to be fully functional for that client.
#define DEBUG

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

#if PRIVATE_CONTRACTS
namespace Microsoft.ML.Runtime.Internal
#else
namespace Microsoft.ML.Runtime
#endif
{
    using Conditional = System.Diagnostics.ConditionalAttribute;
    using Debug = System.Diagnostics.Debug;

    /// <summary>
    /// Interface for "processing" exceptions before they are thrown. This can
    /// be used to add context to the exception, wrap the exception in another one,
    /// totally replace the exception, etc. It is not legal to return null from
    /// Process (unless null was passed in, which really shouldn't happen).
    /// </summary>
#if PRIVATE_CONTRACTS
    internal interface IExceptionContext
#else
    public interface IExceptionContext
#endif
    {
        TException Process<TException>(TException ex)
            where TException : Exception;

        /// <summary>
        /// A string describing the context itself.
        /// </summary>
        string ContextDescription { get; }
    }

#if PRIVATE_CONTRACTS
    [Flags]
    internal enum MessageSensitivity
    {
        None = 0,
        Unknown = ~None
    }
#endif

#if PRIVATE_CONTRACTS
    internal static partial class Contracts
#else
    public static partial class Contracts
#endif
    {
        public const string IsMarkedKey = "ML_IsMarked";
        public const string SensitivityKey = "ML_Sensitivity";

        // This is the assert handler. Typically unit tests set a handler that throws
        // a test failure.
        private static volatile Action<string, IExceptionContext> _handler;

        private static string GetMsg(string msg, params object[] args)
        {
            try
            {
                msg = string.Format(CultureInfo.InvariantCulture, msg, args);
            }
            catch (FormatException ex)
            {
                Contracts.Assert(false, "Format string arg mismatch: " + ex.Message);
            }
            return msg;
        }

        private static int Size<T>(ICollection<T> c)
        {
            return c == null ? 0 : c.Count;
        }

        /// <summary>
        /// Does standard processing of an exception (typically called after construction
        /// but before it is thrown).
        /// </summary>
        public static TException Process<TException>(this TException ex, IExceptionContext ectx = null)
            where TException : Exception
        {
            AssertValue(ex);
            AssertValueOrNull(ectx);
            ex = ectx != null ? ectx.Process(ex) : Mark(ex);
            return ex;
        }

        /// <summary>
        /// Mark the exception by setting <see cref="IsMarkedKey"/> in the exception
        /// <see cref="Exception.Data"/> to 1.
        /// </summary>
        public static TException Mark<TException>(TException ex)
            where TException : Exception
        {
            AssertValue(ex);
            ex.Data[IsMarkedKey] = 1;
            return ex;
        }

        /// <summary>
        /// Indicates whether the exception was "marked" the Contracts code.
        /// </summary>
        public static bool IsMarked(this Exception ex)
        {
            AssertValue(ex);
            return ex.Data.Contains(IsMarkedKey);
        }

        /// <summary>
        /// Exceptions whose message communicates potentially sensitive information should be
        /// marked using this method, before they are thrown. Note that if the exception already
        /// had this flag set, the message will be flagged with the bitwise or of the existing
        /// flag, alongside the passed in sensivity.
        /// </summary>
        public static TException MarkSensitive<TException>(this TException ex, MessageSensitivity sensitivity)
            where TException : Exception
        {
            AssertValue(ex);
            MessageSensitivity innerSensitivity;
            if (!ex.Data.Contains(SensitivityKey))
                innerSensitivity = MessageSensitivity.None;
            else
                innerSensitivity = (ex.Data[SensitivityKey] as MessageSensitivity?) ?? MessageSensitivity.None;
            ex.Data[SensitivityKey] = innerSensitivity | sensitivity;
            return ex;
        }

        /// <summary>
        /// This is a convenience method to get the sensitivity of an exception,
        /// as encoded with <see cref="SensitivityKey"/>. If there is no key, then
        /// the message is assumed to be of unknown sensitivity, i.e., it is assumed
        /// that it might contain literally anything.
        /// </summary>
        /// <param name="ex">The exception to query</param>
        /// <returns>The value encoded at the <see cref="SensitivityKey"/>, if it is
        /// a <see cref="MessageSensitivity"/> value. If neither of these conditions
        /// hold then <see cref="MessageSensitivity.Unknown"/> is returned.</returns>
        public static MessageSensitivity Sensitivity(this Exception ex)
        {
            AssertValue(ex);
            if (!ex.Data.Contains(SensitivityKey))
                return MessageSensitivity.Unknown;
            return (ex.Data[SensitivityKey] as MessageSensitivity?) ?? MessageSensitivity.Unknown;
        }

#if !PRIVATE_CONTRACTS
        /// <summary>
        /// This is an internal convenience implementation of an exception context to make marking
        /// exceptions with a specific sensitivity flag a bit less onorous. The alternative to a scheme
        /// like this, where messages are marked through use of <see cref="Process{TException}(TException)"/>,
        /// would be that every check and exception method in this file would need some "peer" where
        /// sensitivity was set. Since there are so many, we have this method instead. I'm not sure if
        /// there will be performance implications. There shouldn't be, since checks rarely happen in
        /// tight loops.
        /// </summary>
        private readonly struct SensitiveExceptionContext : IExceptionContext
        {
            /// <summary>
            /// We will run this instances <see cref="IExceptionContext.Process{TException}(TException)"/> first.
            /// This can be null.
            /// </summary>
            public readonly IExceptionContext Inner;

            /// <summary>
            /// Exceptions will be marked with this. If <see cref="Inner"/> happens to mark it with a sensivity
            /// flag, then the result will not only be this value, but the bitwise or of this with the existing
            /// value.
            /// </summary>
            public readonly MessageSensitivity ToMark;

            public string ContextDescription => Inner?.ContextDescription ?? "";

            public SensitiveExceptionContext(IExceptionContext inner, MessageSensitivity toMark)
            {
                AssertValueOrNull(inner);
                Inner = inner;
                ToMark = toMark;
            }

            public TException Process<TException>(TException ex) where TException : Exception
            {
                CheckValue(ex, nameof(ex));
                ex = Inner?.Process(ex) ?? ex;
                return ex.MarkSensitive(ToMark);
            }
        }

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.None"/>.
        /// </summary>
        public static IExceptionContext NotSensitive() => NotSensitive(null);

        // REVIEW: The above could be a property, but then it would look unlike the
        // extension method, and extension properties are still not a thing as of C# 7.2. :(

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.None"/>.
        /// </summary>
        public static IExceptionContext NotSensitive(this IExceptionContext ctx)
            => new SensitiveExceptionContext(ctx, MessageSensitivity.None);

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.UserData"/>.
        /// </summary>
        public static IExceptionContext UserSensitive() => UserSensitive(null);

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.UserData"/>.
        /// </summary>
        public static IExceptionContext UserSensitive(this IExceptionContext ctx)
            => new SensitiveExceptionContext(ctx, MessageSensitivity.UserData);

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.Schema"/>.
        /// </summary>
        public static IExceptionContext SchemaSensitive() => SchemaSensitive(null);

        /// <summary>
        /// A convenience context for marking exceptions from checks and excepts with <see cref="MessageSensitivity.Schema"/>.
        /// </summary>
        public static IExceptionContext SchemaSensitive(this IExceptionContext ctx)
            => new SensitiveExceptionContext(ctx, MessageSensitivity.Schema);
#endif

        /// <summary>
        /// Sets the assert handler to the given function, returning the previous handler.
        /// </summary>
        public static Action<string, IExceptionContext> SetAssertHandler(Action<string, IExceptionContext> handler)
        {
            return Interlocked.Exchange(ref _handler, handler);
        }

        // Standard Exception generation. Note that these do NOT throw the exception,
        // merely construct (and log) it.
        // NOTE: The ordering of arguments to these is standardized to be (when they exist):
        // * inner exception
        // * parameter value of type T
        // * parameter name
        // * message composition - either a single string or format followed by params array

        // Default exception type (currently InvalidOperationException)

        /// <summary>
        /// Default exception type (currently InvalidOperationException)
        /// </summary>
        public static Exception Except()
            => Process(new InvalidOperationException());
        public static Exception Except(this IExceptionContext ctx)
            => Process(new InvalidOperationException(), ctx);
        public static Exception Except(string msg)
            => Process(new InvalidOperationException(msg));
        public static Exception Except(this IExceptionContext ctx, string msg)
            => Process(new InvalidOperationException(msg), ctx);
        public static Exception Except(string msg, params object[] args)
            => Process(new InvalidOperationException(GetMsg(msg, args)));
        public static Exception Except(this IExceptionContext ctx, string msg, params object[] args)
            => Process(new InvalidOperationException(GetMsg(msg, args)), ctx);
        public static Exception Except(Exception inner, string msg)
            => Process(new InvalidOperationException(msg, inner));
        public static Exception Except(this IExceptionContext ctx, Exception inner, string msg)
            => Process(new InvalidOperationException(msg, inner), ctx);
        public static Exception Except(Exception inner, string msg, params object[] args)
            => Process(new InvalidOperationException(GetMsg(msg, args), inner));
        public static Exception Except(this IExceptionContext ctx, Exception inner, string msg, params object[] args)
            => Process(new InvalidOperationException(GetMsg(msg, args), inner), ctx);

        // REVIEW: Change ExceptUser*** to use a custom exception type.

        /// <summary>
        /// For signalling bad user input.
        /// </summary>
        public static Exception ExceptUserArg(string name)
            => Process(new ArgumentOutOfRangeException(name));
        public static Exception ExceptUserArg(this IExceptionContext ctx, string name)
            => Process(new ArgumentOutOfRangeException(name), ctx);
        public static Exception ExceptUserArg(string name, string msg)
            => Process(new ArgumentOutOfRangeException(name, msg));
        public static Exception ExceptUserArg(this IExceptionContext ctx, string name, string msg)
            => Process(new ArgumentOutOfRangeException(name, msg), ctx);
        public static Exception ExceptUserArg(string name, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(name, GetMsg(msg, args)));
        public static Exception ExceptUserArg(this IExceptionContext ctx, string name, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(name, GetMsg(msg, args)), ctx);

        /// <summary>
        /// For signalling bad function parameters.
        /// </summary>
        public static Exception ExceptParam(string paramName)
            => Process(new ArgumentOutOfRangeException(paramName));
        public static Exception ExceptParam(this IExceptionContext ctx, string paramName)
            => Process(new ArgumentOutOfRangeException(paramName), ctx);
        public static Exception ExceptParam(string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg));
        public static Exception ExceptParam(this IExceptionContext ctx, string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg), ctx);
        public static Exception ExceptParam(string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, GetMsg(msg, args)));
        public static Exception ExceptParam(this IExceptionContext ctx, string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, GetMsg(msg, args)), ctx);
        public static Exception ExceptParamValue<T>(T value, string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(msg, value, paramName));
        public static Exception ExceptParamValue<T>(this IExceptionContext ctx, T value, string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(msg, value, paramName), ctx);
        public static Exception ExceptParamValue<T>(T value, string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, value, GetMsg(msg, args)));
        public static Exception ExceptParamValue<T>(this IExceptionContext ctx, T value, string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, value, GetMsg(msg, args)), ctx);

        /// <summary>
        /// For signalling null function parameters.
        /// </summary>
        public static Exception ExceptValue(string paramName)
            => Process(new ArgumentNullException(paramName));
        public static Exception ExceptValue(this IExceptionContext ctx, string paramName)
            => Process(new ArgumentNullException(paramName), ctx);
        public static Exception ExceptValue(string paramName, string msg)
            => Process(new ArgumentNullException(paramName, msg));
        public static Exception ExceptValue(this IExceptionContext ctx, string paramName, string msg)
            => Process(new ArgumentNullException(paramName, msg), ctx);
        public static Exception ExceptValue(string paramName, string msg, params object[] args)
            => Process(new ArgumentNullException(paramName, GetMsg(msg, args)));
        public static Exception ExceptValue(this IExceptionContext ctx, string paramName, string msg, params object[] args)
            => Process(new ArgumentNullException(paramName, GetMsg(msg, args)), ctx);

        // For signalling null or empty function parameters (strings, arrays, collections, etc).

        /// <summary>
        /// For signalling null or empty function parameters (strings, arrays, collections, etc).
        /// </summary>
        public static Exception ExceptEmpty(string paramName)
            => Process(new ArgumentOutOfRangeException(paramName, string.Format("{0} cannot be null or empty", paramName)));
        public static Exception ExceptEmpty(this IExceptionContext ctx, string paramName)
            => Process(new ArgumentOutOfRangeException(paramName, string.Format("{0} cannot be null or empty", paramName)), ctx);
        public static Exception ExceptEmpty(string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg));
        public static Exception ExceptEmpty(this IExceptionContext ctx, string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg), ctx);
        public static Exception ExceptEmpty(string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, GetMsg(msg, args)));
        public static Exception ExceptEmpty(this IExceptionContext ctx, string paramName, string msg, params object[] args)
            => Process(new ArgumentOutOfRangeException(paramName, GetMsg(msg, args)), ctx);

        /// <summary>
        /// For signalling null, empty or white-space function parameters (strings, arrays, collections, etc).
        /// </summary>
        public static Exception ExceptWhiteSpace(string paramName)
            => Process(new ArgumentOutOfRangeException(paramName, string.Format("{0} cannot be null or white space", paramName)));
        public static Exception ExceptWhiteSpace(this IExceptionContext ctx, string paramName)
            => Process(new ArgumentOutOfRangeException(paramName, string.Format("{0} cannot be null or white space", paramName)), ctx);
        public static Exception ExceptWhiteSpace(string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg));
        public static Exception ExceptWhiteSpace(this IExceptionContext ctx, string paramName, string msg)
            => Process(new ArgumentOutOfRangeException(paramName, msg), ctx);

        /// <summary>
        /// For signalling errors in decoding information, whether while reading from a file,
        /// parsing user input, etc.
        /// </summary>
        /// <returns></returns>
        public static Exception ExceptDecode()
            => Process(new FormatException());
        public static Exception ExceptDecode(this IExceptionContext ctx)
            => Process(new FormatException(), ctx);
        public static Exception ExceptDecode(string msg)
            => Process(new FormatException(msg));
        public static Exception ExceptDecode(this IExceptionContext ctx, string msg)
            => Process(new FormatException(msg), ctx);
        public static Exception ExceptDecode(string msg, params object[] args)
            => Process(new FormatException(GetMsg(msg, args)));
        public static Exception ExceptDecode(this IExceptionContext ctx, string msg, params object[] args)
            => Process(new FormatException(GetMsg(msg, args)), ctx);
        public static Exception ExceptDecode(Exception inner, string msg)
            => Process(new FormatException(msg, inner));
        public static Exception ExceptDecode(this IExceptionContext ctx, Exception inner, string msg)
            => Process(new FormatException(msg, inner), ctx);
        public static Exception ExceptDecode(Exception inner, string msg, params object[] args)
            => Process(new FormatException(GetMsg(msg, args), inner));
        public static Exception ExceptDecode(this IExceptionContext ctx, Exception inner, string msg, params object[] args)
            => Process(new FormatException(GetMsg(msg, args), inner), ctx);

        /// <summary>
        /// For signalling IO failures.
        /// </summary>
        public static Exception ExceptIO()
            => Process(new IOException());
        public static Exception ExceptIO(this IExceptionContext ctx)
            => Process(new IOException(), ctx);
        public static Exception ExceptIO(string msg)
            => Process(new IOException(msg));
        public static Exception ExceptIO(this IExceptionContext ctx, string msg)
            => Process(new IOException(msg), ctx);
        public static Exception ExceptIO(string msg, params object[] args)
            => Process(new IOException(GetMsg(msg, args)));
        public static Exception ExceptIO(this IExceptionContext ctx, string msg, params object[] args)
            => Process(new IOException(GetMsg(msg, args)), ctx);
        public static Exception ExceptIO(Exception inner, string msg)
            => Process(new IOException(msg, inner));
        public static Exception ExceptIO(this IExceptionContext ctx, Exception inner, string msg)
            => Process(new IOException(msg, inner), ctx);
        public static Exception ExceptIO(Exception inner, string msg, params object[] args)
            => Process(new IOException(GetMsg(msg, args), inner));
        public static Exception ExceptIO(this IExceptionContext ctx, Exception inner, string msg, params object[] args)
            => Process(new IOException(GetMsg(msg, args), inner), ctx);

        /// <summary>
        /// For signalling functionality that is not YET implemented.
        /// </summary>
        public static Exception ExceptNotImpl()
            => Process(new NotImplementedException());
        public static Exception ExceptNotImpl(this IExceptionContext ctx)
            => Process(new NotImplementedException(), ctx);
        public static Exception ExceptNotImpl(string msg)
            => Process(new NotImplementedException(msg));
        public static Exception ExceptNotImpl(this IExceptionContext ctx, string msg)
            => Process(new NotImplementedException(msg), ctx);
        public static Exception ExceptNotImpl(string msg, params object[] args)
            => Process(new NotImplementedException(GetMsg(msg, args)));
        public static Exception ExceptNotImpl(this IExceptionContext ctx, string msg, params object[] args)
            => Process(new NotImplementedException(GetMsg(msg, args)), ctx);

        /// <summary>
        /// For signalling functionality that is not implemented by design.
        /// </summary>
        public static Exception ExceptNotSupp()
            => Process(new NotSupportedException());
        public static Exception ExceptNotSupp(this IExceptionContext ctx)
            => Process(new NotSupportedException(), ctx);
        public static Exception ExceptNotSupp(string msg)
            => Process(new NotSupportedException(msg));
        public static Exception ExceptNotSupp(this IExceptionContext ctx, string msg)
            => Process(new NotSupportedException(msg), ctx);
        public static Exception ExceptNotSupp(string msg, params object[] args)
            => Process(new NotSupportedException(GetMsg(msg, args)));
        public static Exception ExceptNotSupp(this IExceptionContext ctx, string msg, params object[] args)
            => Process(new NotSupportedException(GetMsg(msg, args)), ctx);

        /// <summary>
        /// For signalling schema validation issues.
        /// </summary>
        public static Exception ExceptSchemaMismatch(string paramName, string columnRole, string columnName)
            => Process(new ArgumentOutOfRangeException(paramName, MakeSchemaMismatchMsg(columnRole, columnName)));
        public static Exception ExceptSchemaMismatch(this IExceptionContext ctx, string paramName, string columnRole, string columnName)
            => Process(new ArgumentOutOfRangeException(paramName, MakeSchemaMismatchMsg(columnRole, columnName)), ctx);
        public static Exception ExceptSchemaMismatch(string paramName, string columnRole, string columnName, string expectedType, string actualType)
            => Process(new ArgumentOutOfRangeException(paramName, MakeSchemaMismatchMsg(columnRole, columnName, expectedType, actualType)));
        public static Exception ExceptSchemaMismatch(this IExceptionContext ctx, string paramName, string columnRole, string columnName, string expectedType, string actualType)
            => Process(new ArgumentOutOfRangeException(paramName, MakeSchemaMismatchMsg(columnRole, columnName, expectedType, actualType)), ctx);

        private static string MakeSchemaMismatchMsg(string columnRole, string columnName, string expectedType = null, string actualType = null)
        {
            if (actualType == null)
                return $"Could not find {columnRole} column '{columnName}'";
            return $"Schema mismatch for {columnRole} column '{columnName}': expected {expectedType}, got {actualType}";
        }

    // Check - these check a condition and if it fails, throw the corresponding exception.
    // NOTE: The ordering of arguments to these is standardized to be:
    // * boolean condition
    // * parameter name
    // * parameter value
    // * message string
    //
    // Note that these do NOT support a params array of arguments since that would
    // involve memory allocation whenever the condition is checked. When message string
    // args are need, the condition test should be inlined, eg:
    //   if (!condition)
    //       throw Contracts.ExceptXxx(fmt, arg1, arg2);

    public static void Check(bool f)
    {
        if (!f)
            throw Except();
    }
    public static void Check(this IExceptionContext ctx, bool f)
    {
        if (!f)
            throw Except(ctx);
    }
    public static void Check(bool f, string msg)
    {
        if (!f)
            throw Except(msg);
    }
    public static void Check(this IExceptionContext ctx, bool f, string msg)
    {
        if (!f)
            throw Except(ctx, msg);
    }

    /// <summary>
    /// CheckUserArg / ExceptUserArg should be used when the validation of user-provided arguments failed.
    /// Typically, this is shortly after the arguments are parsed using CmdParser.
    /// </summary>
    public static void CheckUserArg(bool f, string name)
    {
        if (!f)
            throw ExceptUserArg(name);
    }
    public static void CheckUserArg(this IExceptionContext ctx, bool f, string name)
    {
        if (!f)
            throw ExceptUserArg(ctx, name);
    }
    public static void CheckUserArg(bool f, string name, string msg)
    {
        if (!f)
            throw ExceptUserArg(name, msg);
    }
    public static void CheckUserArg(this IExceptionContext ctx, bool f, string name, string msg)
    {
        if (!f)
            throw ExceptUserArg(ctx, name, msg);
    }

    public static void CheckParam(bool f, string paramName)
    {
        if (!f)
            throw ExceptParam(paramName);
    }
    public static void CheckParam(this IExceptionContext ctx, bool f, string paramName)
    {
        if (!f)
            throw ExceptParam(ctx, paramName);
    }
    public static void CheckParam(bool f, string paramName, string msg)
    {
        if (!f)
            throw ExceptParam(paramName, msg);
    }
    public static void CheckParam(this IExceptionContext ctx, bool f, string paramName, string msg)
    {
        if (!f)
            throw ExceptParam(ctx, paramName, msg);
    }
    public static void CheckParamValue<T>(bool f, T value, string paramName, string msg)
    {
        if (!f)
            throw ExceptParamValue(value, paramName, msg);
    }
    public static void CheckParamValue<T>(this IExceptionContext ctx, bool f, T value, string paramName, string msg)
    {
        if (!f)
            throw ExceptParamValue(ctx, value, paramName, msg);
    }

    public static T CheckRef<T>(T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(paramName);
        return val;
    }
    public static T CheckRef<T>(this IExceptionContext ctx, T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(ctx, paramName);
        return val;
    }

    public static T CheckRef<T>(this IExceptionContext ctx, T val, string paramName, string msg) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(ctx, paramName, msg);
        return val;
    }

        public static void CheckValue<T>(T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(paramName);
    }
    public static void CheckValue<T>(this IExceptionContext ctx, T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(ctx, paramName);
    }
    public static T CheckValue<T>(T val, string paramName, string msg) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(paramName, msg);
        return val;
    }
    public static T CheckValue<T>(this IExceptionContext ctx, T val, string paramName, string msg) where T : class
    {
        if (object.ReferenceEquals(val, null))
            throw ExceptValue(ctx, paramName, msg);
        return val;
    }

    public static string CheckNonEmpty(string s, string paramName)
    {
        if (string.IsNullOrEmpty(s))
            throw ExceptEmpty(paramName);
        return s;
    }
    public static string CheckNonEmpty(this IExceptionContext ctx, string s, string paramName)
    {
        if (string.IsNullOrEmpty(s))
            throw ExceptEmpty(ctx, paramName);
        return s;
    }

    public static string CheckNonWhiteSpace(string s, string paramName)
    {
        if (string.IsNullOrWhiteSpace(s))
            throw ExceptWhiteSpace(paramName);
        return s;
    }
    public static string CheckNonWhiteSpace(this IExceptionContext ctx, string s, string paramName)
    {
        if (string.IsNullOrWhiteSpace(s))
            throw ExceptWhiteSpace(ctx, paramName);
        return s;
    }

    public static string CheckNonEmpty(string s, string paramName, string msg)
    {
        if (string.IsNullOrEmpty(s))
            throw ExceptEmpty(paramName, msg);
        return s;
    }
    public static string CheckNonEmpty(this IExceptionContext ctx, string s, string paramName, string msg)
    {
        if (string.IsNullOrEmpty(s))
            throw ExceptEmpty(ctx, paramName, msg);
        return s;
    }

    public static string CheckNonWhiteSpace(string s, string paramName, string msg)
    {
        if (string.IsNullOrWhiteSpace(s))
            throw ExceptWhiteSpace(paramName, msg);
        return s;
    }
    public static string CheckNonWhiteSpace(this IExceptionContext ctx, string s, string paramName, string msg)
    {
        if (string.IsNullOrWhiteSpace(s))
            throw ExceptWhiteSpace(ctx, paramName, msg);
        return s;
    }

    public static T[] CheckNonEmpty<T>(T[] args, string paramName)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(paramName);
        return args;
    }
    public static T[] CheckNonEmpty<T>(this IExceptionContext ctx, T[] args, string paramName)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(ctx, paramName);
        return args;
    }
    public static T[] CheckNonEmpty<T>(T[] args, string paramName, string msg)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(paramName, msg);
        return args;
    }
    public static T[] CheckNonEmpty<T>(this IExceptionContext ctx, T[] args, string paramName, string msg)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(ctx, paramName, msg);
        return args;
    }
    public static ICollection<T> CheckNonEmpty<T>(ICollection<T> args, string paramName)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(paramName);
        return args;
    }
    public static ICollection<T> CheckNonEmpty<T>(this IExceptionContext ctx, ICollection<T> args, string paramName)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(ctx, paramName);
        return args;
    }
    public static ICollection<T> CheckNonEmpty<T>(ICollection<T> args, string paramName, string msg)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(paramName, msg);
        return args;
    }
    public static ICollection<T> CheckNonEmpty<T>(this IExceptionContext ctx, ICollection<T> args, string paramName, string msg)
    {
        if (Size(args) == 0)
            throw ExceptEmpty(ctx, paramName, msg);
        return args;
    }

    public static void CheckDecode(bool f)
    {
        if (!f)
            throw ExceptDecode();
    }
    public static void CheckDecode(this IExceptionContext ctx, bool f)
    {
        if (!f)
            throw ExceptDecode(ctx);
    }
    public static void CheckDecode(bool f, string msg)
    {
        if (!f)
            throw ExceptDecode(msg);
    }
    public static void CheckDecode(this IExceptionContext ctx, bool f, string msg)
    {
        if (!f)
            throw ExceptDecode(ctx, msg);
    }

    public static void CheckIO(bool f)
    {
        if (!f)
            throw ExceptIO();
    }
    public static void CheckIO(this IExceptionContext ctx, bool f)
    {
        if (!f)
            throw ExceptIO(ctx);
    }
    public static void CheckIO(bool f, string msg)
    {
        if (!f)
            throw ExceptIO(msg);
    }
    public static void CheckIO(this IExceptionContext ctx, bool f, string msg)
    {
        if (!f)
            throw ExceptIO(ctx, msg);
    }

#if !PRIVATE_CONTRACTS
        /// <summary>
        /// Check state of the host and throw exception if host marked to stop all exection.
        /// </summary>
        public static void CheckAlive(this IHostEnvironment env)
        {
            if (env.IsCancelled)
                throw Process(new OperationCanceledException("Operation was cancelled."), env);
        }
#endif
    /// <summary>
    /// This documents that the parameter can legally be null.
    /// </summary>
    [Conditional("INVARIANT_CHECKS")]
    public static void CheckValueOrNull<T>(T val) where T : class
    {
    }
    [Conditional("INVARIANT_CHECKS")]
    public static void CheckValueOrNull<T>(this IExceptionContext ctx, T val) where T : class
    {
    }

    // Assert

    #region Private assert handling

    private static void DbgFailCore(string msg, IExceptionContext ctx = null)
    {
        var handler = _handler;

        if (handler != null)
            handler(msg, ctx);
        else if (ctx != null)
            Debug.Fail(msg, ctx.ContextDescription);
        else
            Debug.Fail(msg);
    }

    private static void DbgFail(IExceptionContext ctx = null)
    {
        DbgFailCore("Assertion Failed", ctx);
    }
    private static void DbgFail(string msg)
    {
        DbgFailCore(msg);
    }
    private static void DbgFail(IExceptionContext ctx, string msg)
    {
        DbgFailCore(msg, ctx);
    }
    private static void DbgFailValue(IExceptionContext ctx = null)
    {
        DbgFailCore("Non-null assertion failure", ctx);
    }
    private static void DbgFailValue(string paramName)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-null assertion failure: {0}", paramName));
    }
    private static void DbgFailValue(IExceptionContext ctx, string paramName)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-null assertion failure: {0}", paramName), ctx);
    }
    private static void DbgFailValue(string paramName, string msg)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-null assertion failure: {0}: {1}", paramName, msg));
    }
    private static void DbgFailValue(IExceptionContext ctx, string paramName, string msg)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-null assertion failure: {0}: {1}", paramName, msg), ctx);
    }
    private static void DbgFailEmpty(IExceptionContext ctx = null)
    {
        DbgFailCore("Non-empty assertion failure", ctx);
    }
    private static void DbgFailEmpty(string msg)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-empty assertion failure: {0}", msg));
    }
    private static void DbgFailEmpty(IExceptionContext ctx, string msg)
    {
        DbgFailCore(string.Format(CultureInfo.InvariantCulture, "Non-empty assertion failure: {0}", msg), ctx);
    }

    #endregion Private assert handling

    [Conditional("DEBUG")]
    public static void Assert(bool f)
    {
        if (!f)
            DbgFail();
    }
    [Conditional("DEBUG")]
    public static void Assert(this IExceptionContext ctx, bool f)
    {
        if (!f)
            DbgFail(ctx);
    }

    [Conditional("DEBUG")]
    public static void Assert(bool f, string msg)
    {
        if (!f)
            DbgFail(msg);
    }
    [Conditional("DEBUG")]
    public static void Assert(this IExceptionContext ctx, bool f, string msg)
    {
        if (!f)
            DbgFail(ctx, msg);
    }

    [Conditional("DEBUG")]
    public static void AssertValue<T>(T val) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue();
    }
    [Conditional("DEBUG")]
    public static void AssertValue<T>(this IExceptionContext ctx, T val) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue(ctx);
    }

    [Conditional("DEBUG")]
    public static void AssertValue<T>(T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue(paramName);
    }
    [Conditional("DEBUG")]
    public static void AssertValue<T>(this IExceptionContext ctx, T val, string paramName) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue(ctx, paramName);
    }

    [Conditional("DEBUG")]
    public static void AssertValue<T>(T val, string name, string msg) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue(name, msg);
    }
    [Conditional("DEBUG")]
    public static void AssertValue<T>(this IExceptionContext ctx, T val, string name, string msg) where T : class
    {
        if (object.ReferenceEquals(val, null))
            DbgFailValue(ctx, name, msg);
    }

    [Conditional("DEBUG")]
    public static void AssertNonEmpty(string s)
    {
        if (string.IsNullOrEmpty(s))
            DbgFailEmpty();
    }
    [Conditional("DEBUG")]
    public static void AssertNonEmpty(this IExceptionContext ctx, string s)
    {
        if (string.IsNullOrEmpty(s))
            DbgFailEmpty(ctx);
    }

    [Conditional("DEBUG")]
    public static void AssertNonWhiteSpace(string s)
    {
        if (string.IsNullOrWhiteSpace(s))
            DbgFailEmpty();
    }
    [Conditional("DEBUG")]
    public static void AssertNonWhiteSpace(this IExceptionContext ctx, string s)
    {
        if (string.IsNullOrWhiteSpace(s))
            DbgFailEmpty(ctx);
    }

    [Conditional("DEBUG")]
    public static void AssertNonEmpty(string s, string msg)
    {
        if (string.IsNullOrEmpty(s))
            DbgFailEmpty(msg);
    }
    [Conditional("DEBUG")]
    public static void AssertNonEmpty(this IExceptionContext ctx, string s, string msg)
    {
        if (string.IsNullOrEmpty(s))
            DbgFailEmpty(ctx, msg);
    }

    [Conditional("DEBUG")]
    public static void AssertNonWhiteSpace(string s, string msg)
    {
        if (string.IsNullOrWhiteSpace(s))
            DbgFailEmpty(msg);
    }
    [Conditional("DEBUG")]
    public static void AssertNonWhiteSpace(this IExceptionContext ctx, string s, string msg)
    {
        if (string.IsNullOrWhiteSpace(s))
            DbgFailEmpty(ctx, msg);
    }

    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(ReadOnlySpan<T> args)
    {
        if (args.IsEmpty)
            DbgFail();
    }
    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(Span<T> args)
    {
        if (args.IsEmpty)
            DbgFail();
    }

    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(ICollection<T> args)
    {
        if (Size(args) == 0)
            DbgFail();
    }
    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(this IExceptionContext ctx, ICollection<T> args)
    {
        if (Size(args) == 0)
            DbgFail(ctx);
    }

    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(ICollection<T> args, string msg)
    {
        if (Size(args) == 0)
            DbgFail(msg);
    }
    [Conditional("DEBUG")]
    public static void AssertNonEmpty<T>(this IExceptionContext ctx, ICollection<T> args, string msg)
    {
        if (Size(args) == 0)
            DbgFail(ctx, msg);
    }

    /// <summary>
    /// This documents that the parameter can legally be null.
    /// </summary>
    [Conditional("INVARIANT_CHECKS")]
    public static void AssertValueOrNull<T>(T val) where T : class
    {
    }
    [Conditional("INVARIANT_CHECKS")]
    public static void AssertValueOrNull<T>(this IExceptionContext ctx, T val) where T : class
    {
    }
}
}
