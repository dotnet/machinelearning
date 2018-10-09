// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Linq;

// We use this TF_Xxx as the native "TF_Xxx *" as those are opaque
using TF_Status = System.IntPtr;
using TF_SessionOptions = System.IntPtr;
using TF_Graph = System.IntPtr;
using TF_OperationDescription = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Session = System.IntPtr;
using TF_DeprecatedSession = System.IntPtr;
using TF_Tensor = System.IntPtr;
using TF_ImportGraphDefOptions = System.IntPtr;
using TF_Library = System.IntPtr;
using TF_BufferPtr = System.IntPtr;
using TF_Function = System.IntPtr;
using TF_DeviceList = System.IntPtr;

using size_t = System.UIntPtr;
using System.Collections.Generic;
using System.Collections;

#pragma warning disable MSML_GeneralName
#pragma warning disable MSML_PrivateFieldName
#pragma warning disable MSML_ParameterLocalVarName

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal static partial class NativeBinding
    {
        public const string TensorFlowLibrary = "tensorflow";
        public const string TensorFlowLibraryGPU = "libtensorflowgpu";

        internal static string GetStr(this IntPtr x) => Marshal.PtrToStringAnsi(x);
    }

    /// <summary>
    /// Contains TensorFlow fundamental methods and utility functions.
    /// </summary>
    internal static class TFCore
    {
        internal static bool UseCPU = true;

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe IntPtr TF_Version();

        static TFCore()
        {
            Init();
        }

        internal static void Init()
        {
            CheckSize();
        }

        /// <summary>
        /// Returns the version of the TensorFlow runtime in use.
        /// </summary>
        /// <value>The version.</value>
        public static string Version => TF_Version().GetStr();

        // extern size_t TF_DataTypeSize (TF_DataType dt);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern IntPtr TF_DataTypeSize(TFDataType dt);

        /// <summary>
        /// Gets the size in bytes of the specified TensorFlow data type.
        /// </summary>
        /// <returns>The data type size.</returns>
        /// <param name="dt">Dt.</param>
        public static long GetDataTypeSize(TFDataType dt) => (long)TF_DataTypeSize(dt);

        // extern TF_Buffer * TF_GetAllOpList ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe IntPtr TF_GetAllOpList();

        /// <summary>
        /// Retrieves the ProtocolBuffer describing all of the available operations in
        /// the TensorFlow library in current use.
        /// </summary>
        /// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
        public static TFBuffer GetAllOpList()
        {
            return new TFBuffer(TF_GetAllOpList());
        }

        private static void CheckSize()
        {
            unsafe
            {
                if (sizeof(IntPtr) == 4)
                {
                    Console.Error.WriteLine(
                        "The TensorFlow native libraries were compiled in 64 bit mode, you must run in 64 bit mode\n" +
                        "With Mono, do that with mono --arch=64 executable.exe, if using an IDE like MonoDevelop,\n" +
                        "Xamarin Studio or Visual Studio for Mac, Build/Compiler settings, make sure that " +
                        "\"Platform Target\" has x64 selected.");
                    throw new Exception();

                }
            }
        }
    }

    /// <summary>
    /// Base class for many TensorFlow data types that provides a common idiom to dispose and
    /// release resources associated with the native data types.   Generally, you do not need to use this.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implements the Dispose pattern in a reusable form for TensorFlow types.
    /// </para>
    /// <para>
    /// Subclasses invoke the constructor with the handle that this will wrap, and must
    /// override the NativeDispose method (internal) to release the associated resource.
    /// </para>
    /// </remarks>
    internal abstract class TFDisposable : IDisposable
    {
        internal IntPtr handle;

        /// <summary>
        /// Returns the opaque handle to the object that this TFDisposable owns.
        /// </summary>
        /// <value>The handle.</value>
        public IntPtr Handle => handle;

        static TFDisposable()
        {
            TFCore.Init();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposable"/> class.
        /// </summary>
        public TFDisposable()
        { }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposable"/> class
        /// from the handle that it will wrap.
        /// </summary>
        public TFDisposable(IntPtr handle)
        {
            this.handle = handle;
        }

        /// <summary>
        /// Releases all resource used by the <see cref="T:TensorFlow.TFDisposable"/> object.
        /// </summary>
        /// <remarks>Call Dispose when you are finished using the <see cref="T:TensorFlow.TFDisposable"/>. The
        /// Dispose method leaves the <see cref="T:TensorFlow.TFDisposable"/> in an unusable state. After
        /// calling Dispose, you must release all references to the <see cref="T:TensorFlow.TFDisposable"/> so
        /// the garbage collector can reclaim the memory that the <see cref="T:TensorFlow.TFDisposable"/> was occupying.</remarks>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~TFDisposable()
        {
            Dispose(false);
        }

        // Must be implemented in subclasses to dispose the unmanaged object, it does
        // not need to take care of zeroing out the handle, that is done by the Dispose
        // method inherited from TFDisposable
        internal abstract void NativeDispose(IntPtr handle);

        /// <summary>
        /// Dispose the specified object
        /// </summary>
        /// <param name="disposing">If set to <c>true</c> it means that this method was called from Dispose, otherwise from the finalizer.</param>
        public virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (handle != IntPtr.Zero)
                    NativeDispose(handle);
                handle = IntPtr.Zero;
            }
        }

        internal static void ObjectDisposedException()
        {
            throw new ObjectDisposedException("The object was disposed");
        }
    }

    /// <summary>
    /// ase class for many TensorFlow data types that provides a common idiom to dispose and
    /// release resources associated with the native data types and whose unmanaged resource
    /// disposing can be called from a background thread (the finalizer).   Users do not
    /// need to deal with this class.
    /// </summary>
    /// <remarks>
    /// Some object deletion APIs in TensorFlow can be invoked from a background thread,
    /// so the release methods are suitable to be invoked from the Finalizer thread, in
    /// those scenarios, subclass from this class rather than the TFDisposable class.
    /// </remarks>
    internal abstract class TFDisposableThreadSafe : TFDisposable
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposable"/> class
        /// from the handle that it will wrap.
        /// </summary>
        public TFDisposableThreadSafe(IntPtr handle) : base(handle)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposableThreadSafe"/> class.
        /// </summary>
        public TFDisposableThreadSafe()
        { }

        /// <summary>
        /// Dispose the object, unlike the default implementat in TFDisposable,
        /// this will release the unmanaged resources from a background thread.
        /// </summary>
        /// <param name="disposing">If set to <c>true</c> disposing.</param>
        public override void Dispose(bool disposing)
        {
            if (handle != IntPtr.Zero)
                NativeDispose(handle);
            handle = IntPtr.Zero;
        }
    }

    /// <summary>
    /// TensorFlow Exception
    /// </summary>
    internal class TFException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFException"/> class with a message.
        /// </summary>
        /// <param name="message">Message.</param>
        public TFException(string message) : base(message) { }
    }

    /// <summary>
    /// Used to track the result of TensorFlow operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TFStatus is used to track the status of a call to some TensorFlow
    /// operations.   Instances of this object are passed to various
    /// TensorFlow operations and you can use the <see cref="P:TensorFlow.TFStatus.Ok"/>
    /// to quickly check if the operation succeeded, or get more detail from the
    /// <see cref="P:TensorFlow.TFStatus.StatusCode"/> and a human-readable text
    /// using the <see cref="P:TensorFlow.TFStatus.StatusMessage"/> property.
    /// </para>
    /// <para>
    /// The convenience <see cref="M:TensorFlow.TFStatus.Raise"/> can be used
    /// to raise a <see cref="P:TensorFlow.TFException"/> if the status of the
    /// operation did not succeed.
    /// </para>
    /// </remarks>
    internal class TFStatus : TFDisposable
    {
        // extern TF_Status * TF_NewStatus ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_Status TF_NewStatus();

        /// <summary>
        /// Per-thread global status that you can use if you do not need to create a new instance of this object.
        /// </summary>
        /// <remarks>
        /// This is provided as a convenience for APIs that take a TFStatus.   While the TFStatus is usually an
        /// optional parameter, when it is made optional, API calls that fail raise an exception.   Use this
        /// property to pass a TFStatus without having to allocate a new one.   The problem with this of course
        /// is that you risk having multiple parts of your code override this thread-global variable.
        /// </remarks>
        [ThreadStatic] public static TFStatus Default = new TFStatus();

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFStatus"/> class.
        /// </summary>
        public TFStatus() : base(TF_NewStatus())
        {
        }

        // extern void TF_DeleteStatus (TF_Status *);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteStatus(TF_Status status);

        internal override void NativeDispose(IntPtr handle)
        {
            TF_DeleteStatus(handle);
        }

        // extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_SetStatus(TF_Status s, TFCode code, string msg);

        /// <summary>
        /// Sets the status code on this TFStatus.
        /// </summary>
        /// <param name="code">Code.</param>
        /// <param name="msg">Message.</param>
        public void SetStatusCode(TFCode code, string msg)
        {
            TF_SetStatus(handle, code, msg);
        }

        // extern TF_Code TF_GetCode (const TF_Status *s);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TFCode TF_GetCode(TF_Status s);

        /// <summary>
        /// Gets the status code for the status code.
        /// </summary>
        /// <value>The status code as an enumeration.</value>
        public TFCode StatusCode
        {
            get
            {
                if (handle == IntPtr.Zero)
                    throw new ObjectDisposedException("TFStatus");
                return TF_GetCode(handle);
            }
        }

        // extern const char * TF_Message (const TF_Status *s);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe IntPtr TF_Message(TF_Status s);

        /// <summary>
        /// Gets a human-readable status message.
        /// </summary>
        /// <value>The status message.</value>
        public string StatusMessage => TF_Message(handle).GetStr();

        /// <summary>
        /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.
        /// </summary>
        /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.</returns>
        public override string ToString()
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException("TFStatus");

            return string.Format("[TFStatus: StatusCode={0}, StatusMessage={1}]", StatusCode, StatusMessage);
        }

        /// <summary>
        /// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to ok.
        /// </summary>
        /// <value><c>true</c> if ok; otherwise, <c>false</c>.</value>
        public bool Ok => StatusCode == TFCode.Ok;

        /// <summary>
        /// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to an error.
        /// </summary>
        /// <value><c>true</c> if error; otherwise, <c>false</c>.</value>
        public bool Error => StatusCode != TFCode.Ok;

        /// <summary>
        /// Convenience method that raises an exception if the current status is an error.
        /// </summary>
        /// <remarks>
        /// You can use this method as a convenience to raise an exception after you
        /// invoke an operation if the operation did not succeed.
        /// </remarks>
        public void Raise()
        {
            if (TF_GetCode(handle) != TFCode.Ok)
                throw new TFException(StatusMessage);
        }

        //
        // Utility function used to simplify implementing the idiom
        // where the user optionally provides a TFStatus, if it is provided,
        // the error is returned there;   If it is not provided, then an
        // exception is raised.
        //

        internal bool CheckMaybeRaise(TFStatus incomingStatus, bool last = true)
        {
            if (incomingStatus == null)
            {
                if (handle == IntPtr.Zero)
                    Console.WriteLine("oops");
                if (StatusCode != TFCode.Ok)
                {
                    var e = new TFException(StatusMessage);
                    if (last)
                        Dispose();
                    throw e;
                }
                if (last)
                    Dispose();
                return true;
            }
            return StatusCode == TFCode.Ok;
        }

        internal static TFStatus Setup(TFStatus incoming)
        {
            return incoming == null ? new TFStatus() : incoming;
        }
    }

    /// <summary>
    /// The session options object holds configuration options that you want to use during your session, like the TensorFlow target or the configuration.
    /// </summary>
    internal class TFSessionOptions : TFDisposable
    {
        // extern TF_SessionOptions * TF_NewSessionOptions ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_SessionOptions TF_NewSessionOptions();

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFSessionOptions"/> class.
        /// </summary>
        public TFSessionOptions() : base(TF_NewSessionOptions()) { }

        // extern void TF_DeleteSessionOptions (TF_SessionOptions *);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteSessionOptions(TF_SessionOptions options);
        internal override void NativeDispose(IntPtr handle)
        {
            TF_DeleteSessionOptions(handle);
        }

        // extern void TF_SetTarget (TF_SessionOptions *options, const char *target);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_SetTarget(TF_SessionOptions options, string target);

        /// <summary>
        /// Sets the target in options.
        /// </summary>
        /// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
        /// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
        ///
        public void SetTarget(string target)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();

            TF_SetTarget(handle, target);
        }

        // extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_SetConfig(TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status);

        /// <summary>
        /// Sets the configuration information for the session.
        /// </summary>
        /// <param name="protoData">Serialized protocol buffer for the tensorflow.ConfigProto message.</param>
        /// <param name="length">Length of the buffer.</param>
        /// <param name="status">If config was not parsed successfully as a ConfigProto, the error is recorded here.</param>
        /// <remarks>
        /// The configuration option is a Protocol Buffer representing the tensorflow.ConfigProto
        /// </remarks>
        public void SetConfig(IntPtr protoData, int length, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();

            var cstatus = TFStatus.Setup(status);

            TF_SetConfig(handle, protoData, (UIntPtr)length, cstatus.handle);
            cstatus.CheckMaybeRaise(status);
        }

    }

    /// <summary>
    /// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Graphs consist of operations (represented by TFOperation objects), these can be named, or
    /// the runtime will automatically assign a name.
    /// </para>
    /// <para>
    /// For debugging purposes, you might want to group operations together, for this, call the
    /// WithScope method with your new scope, which will create a new namespace for your object names.
    /// </para>
    /// <para>
    /// For example, if you call WithScope ("demo"), and add an operation named "add" inside the
    /// scope, the full name of the operation will be "demo/add", if you create a new scope inside, say
    /// "hot", and add a "sub" operation there the result will be "demo/hot/sub".
    /// </para>
    /// </remarks>
    internal partial class TFGraph : TFDisposableThreadSafe, IEnumerable<TFOperation>
    {
        // extern TF_Graph * TF_NewGraph ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_Graph TF_NewGraph();

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
        /// </summary>
        public TFGraph() : base(TF_NewGraph())
        {
        }

        // extern void TF_DeleteGraph (TF_Graph *);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteGraph(TF_Graph graph);
        internal override void NativeDispose(IntPtr handle)
        {
            TF_DeleteGraph(handle);
        }

        // extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe int TF_GraphGetTensorNumDims(TF_Graph graph, TFOutput output, TF_Status status);

        // extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_GraphGetTensorShape(TF_Graph graph, TFOutput output, long[] dims, int num_dims, TF_Status status);

        /// <summary>
        /// Returns the shape of a tensor specified in <paramref name="output"/>.
        /// </summary>
        ///
        /// <returns>The tensor shape.    If the number of dimensions in the shape is unknown or the shape is, a scalar, the values in the array will be zero. Otherwise, each element of will be set corresponding to the size of the dimension. An  unknown dimension is represented by -1.</returns>
        /// <param name="output">The tensor that you want to look up.  </param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public TFShape GetTensorShape(TFOutput output, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            var cstatus = TFStatus.Setup(status);
            var n = TF_GraphGetTensorNumDims(handle, output, cstatus.handle);
            if (!cstatus.CheckMaybeRaise(status, last: false))
                return TFShape.Unknown;
            if (n == -1)
                return TFShape.Unknown;

            var dims = new long[n];
            TF_GraphGetTensorShape(handle, output, dims, dims.Length, cstatus.handle);
            cstatus.CheckMaybeRaise(status);
            return new TFShape(dims);
        }

        // extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_GraphToGraphDef(TF_Graph graph, LLBuffer* output_graph_def, TF_Status status);

        /// <summary>
        /// Write out a serialized representation of the graph (as a GraphDef protocol buffer message) into <paramref name="outputGraphDef"/>.
        /// </summary>
        /// <param name="outputGraphDef">Target buffer where the graphs is serialized into.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public void ToGraphDef(TFBuffer outputGraphDef, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (outputGraphDef == null)
                throw new ArgumentNullException(nameof(outputGraphDef));

            var cstatus = TFStatus.Setup(status);
            unsafe
            {
                TF_GraphToGraphDef(handle, outputGraphDef.LLBuffer, cstatus.handle);
            }
            cstatus.CheckMaybeRaise(status);
        }

        // extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_GraphImportGraphDef(TF_Graph graph, LLBuffer* graph_def, TF_ImportGraphDefOptions options, TF_Status status);

        /// <summary>
        /// Import a serialized graph into this graph, using the specified prefix.
        /// </summary>
        /// <returns>The import.</returns>
        /// <param name="graphDef">A buffer containing the serialized graph.</param>
        /// <param name="prefix">A prefix that will be prepended to names of nodes in the <paramref name="graphDef"/> when they are imported into the graph.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public void Import(TFBuffer graphDef, string prefix = "", TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (graphDef == null)
                throw new ArgumentNullException(nameof(graphDef));
            if (prefix == null)
                throw new ArgumentNullException(nameof(prefix));

            using (var options = new TFImportGraphDefOptions())
            {
                options.SetPrefix(prefix);
                Import(graphDef, options, status);
            }
        }

        /// <summary>
        /// Import a serialized graph into this graph, using the specified importing options.
        /// </summary>
        /// <returns>The import.</returns>
        /// <param name="graphDef">A buffer containing the serialized graph.</param>
        /// <param name="options">Importing graph options.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public void Import(TFBuffer graphDef, TFImportGraphDefOptions options, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (graphDef == null)
                throw new ArgumentNullException(nameof(graphDef));
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            var cstatus = TFStatus.Setup(status);
            unsafe
            {
                TF_GraphImportGraphDef(handle, graphDef.LLBuffer, options.handle, cstatus.handle);
            }
            cstatus.CheckMaybeRaise(status);
        }

        /// <summary>
        /// Import a serialized graph held in a byte array into this graph, using the specified prefix.
        /// </summary>
        /// <returns>The import.</returns>
        /// <param name="buffer">A byte array containing the serialized graph.</param>
        /// <param name="prefix">A prefix that will be prepended to names of nodes in the graph when they are imported into the graph.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public void Import(byte[] buffer, string prefix = "", TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (prefix == null)
                throw new ArgumentNullException(nameof(prefix));
            using (var options = new TFImportGraphDefOptions())
            {
                options.SetPrefix(prefix);
                Import(buffer, options, status);
            }
        }

        /// <summary>
        /// Import a serialized graph held in a byte array into this graph, using the specified import options.
        /// </summary>
        /// <returns>The import.</returns>
        /// <param name="buffer">A byte array containing the serialized graph.</param>
        /// <param name="options">Importing graph options.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        ///   If you are tryig to load a file stored using the SavedModel file format, you should use the <see cref="T:TensorFlow.TFSession.FromSavedModel"/> API instead.
        /// </remarks>
        public void Import(byte[] buffer, TFImportGraphDefOptions options, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (options == null)
                throw new ArgumentNullException(nameof(options));
            var cstatus = TFStatus.Setup(status);
            using (var tb = new TFBuffer(buffer, 0, buffer.Length))
                Import(tb, options, status);

            cstatus.CheckMaybeRaise(cstatus);
        }

        // extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_Operation TF_GraphOperationByName(TF_Graph graph, string oper_name);

        /// <summary>
        /// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name, or null if the named operation does not exist in the graph.
        /// </summary>
        /// <param name="name">Name to lookup.</param>
        public TFOperation this[string name]
        {
            get
            {
                if (handle == IntPtr.Zero)
                    ObjectDisposedException();
                var h = TF_GraphOperationByName(handle, name);
                if (h == IntPtr.Zero)
                    return null;
                return new TFOperation(this, h);
            }
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern string TF_GraphDebugString(TF_Graph graph, out IntPtr len);

        public override string ToString()
        {
            IntPtr len;
            return TF_GraphDebugString(Handle, out len);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static unsafe extern TF_Operation TF_GraphNextOperation(TF_Graph graph, ref IntPtr pos);

        /// <summary>
        /// Returns the enumerator that returns all the TFOperations in a graph.
        /// </summary>
        /// <returns>The enumerator.</returns>
        private IEnumerable<TFOperation> GetEnumerable()
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            IntPtr token = IntPtr.Zero;
            IntPtr operll;
            while ((operll = TF_GraphNextOperation(handle, ref token)) != IntPtr.Zero)
                yield return new TFOperation(this, operll);
        }

        public IEnumerator<TFOperation> GetEnumerator()
        {
            return GetEnumerable().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Represents a computation node in the graph.  Tensorflow operations are attached to a <see cref="T:Tensorflow.TFGraph"/>.
    /// </summary>
    /// <remarks>
    /// TFOperations are usually created by  invoking one of the methods in
    /// <see cref="T:Tensorflow.TFGraph"/>, but they can also be constructed
    /// manually using the low-level <see cref="T:Tensorflow.TFOperationDesc"/> API.
    /// </remarks>
    internal partial class TFOperation
    {
        internal IntPtr handle;

        /// <summary>
        /// Gets the handle to the unmanaged TF_Operation object.
        /// </summary>
        /// <value>The handle.</value>
        public IntPtr Handle => handle;

        // Pointer to the graph, to keep it from collecting if there are TFOperations alive.
        internal TFGraph graph;

        internal TFOperation(TFGraph graph, IntPtr handle)
        {
            this.handle = handle;
            this.graph = graph;
        }

        /// <summary>
        /// Returns the handle to the idx-th output of the operation.
        /// </summary>
        /// <param name="idx">Index of the output in the operation.</param>
        public TFOutput this[int idx]
        {
            get
            {
                return new TFOutput(this, idx);
            }
        }

        // extern TF_Output TF_OperationInput (TF_Input oper_in);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern TFOutput TF_OperationInput(TFInput oper_in);

        public TFOutput GetInput(int idx)
        {
            return TF_OperationInput(new TFInput() { Operation = handle, Index = idx });
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern IntPtr TF_OperationName(TF_Operation oper);

        /// <summary>
        /// The name for this operation/
        /// </summary>
        /// <value>The name.</value>
        public string Name => handle == IntPtr.Zero ? "<ObjectDisposed>" : TF_OperationName(handle).GetStr();

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern IntPtr TF_OperationOpType(TF_Operation oper);

        public string OpType => handle == IntPtr.Zero ? "<ObjectDisposedException>" : TF_OperationOpType(handle).GetStr();

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern int TF_OperationNumOutputs(TF_Operation oper);

        /// <summary>
        /// Gets the number of outputs on this operation.
        /// </summary>
        /// <value>The number outputs.</value>
        public int NumOutputs => handle == IntPtr.Zero ? -1 : TF_OperationNumOutputs(handle);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern int TF_OperationNumInputs(TF_Operation oper);

        /// <summary>
        /// Gets the number of inputs for this operation.
        /// Import a serialized graph into this graph, using the specified importing options.
        /// </summary>
        /// <value>The number inputs.</value>
        public int NumInputs => TF_OperationNumInputs(handle);
    }

    /// <summary>
    /// Device type
    /// </summary>
    internal enum DeviceType
    {
        /// <summary>
        /// The device is the Central Processing Unit (CPU)
        /// </summary>
        CPU,

        /// <summary>
        /// The device is a Graphics Processing Unit (GPU)
        /// </summary>
        GPU,

        /// <summary>
        /// The device is a Tensor Processing Unit (TPU)
        /// </summary>
        TPU
    }

    /// <summary>
    /// Describes the device attributes
    /// </summary>
    internal class DeviceAttributes
    {
        internal DeviceAttributes(string name, DeviceType deviceType, long memoryLimitBytes)
        {
            Name = name;
            DeviceType = deviceType;
            MemoryLimitBytes = memoryLimitBytes;
        }

        /// <summary>
        /// The full name of the device (for example, /job:worker/replica:0/...)
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// Gets the type of the device.
        /// </summary>
        /// <value>The type of the device.</value>
        public DeviceType DeviceType { get; private set; }

        /// <summary>
        /// The amount of memory associated with a given device.
        /// </summary>
        /// <value>The memory limit bytes.</value>
        public long MemoryLimitBytes { get; private set; }
    }

    /// <summary>
    /// Contains options that are used to control how graph importing works.
    /// </summary>
    internal class TFImportGraphDefOptions : TFDisposable
    {
        // extern TF_ImportGraphDefOptions * TF_NewImportGraphDefOptions ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_ImportGraphDefOptions TF_NewImportGraphDefOptions();

        public TFImportGraphDefOptions() : base(TF_NewImportGraphDefOptions())
        {
        }

        // extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions *opts);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions opts);

        internal override void NativeDispose(IntPtr handle)
        {
            TF_DeleteImportGraphDefOptions(handle);
        }

        // extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions *opts, const char *prefix);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions opts, string prefix);

        public void SetPrefix(string prefix)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            TF_ImportGraphDefOptionsSetPrefix(handle, prefix);
        }

        // extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions *opts, const char* src_name, int src_index, TF_Output dst);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_ImportGraphDefOptionsAddInputMapping(TF_ImportGraphDefOptions opts, string src_name, int src_index, TFOutput dst);

        /// <summary>
        /// Adds an input mapping from a source name and index to a destination output
        /// </summary>
        /// <param name="srcName">Source name.</param>
        /// <param name="srcIndex">Source index (in the source).</param>
        /// <param name="dst">Replacement value for the srcName:srcIndex.</param>
        /// <remarks>
        /// Set any imported nodes with input `src_name:src_index` to have that input
        /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
        /// `dst` references a node already existing in the graph being imported into.
        /// </remarks>
        public void AddInputMapping(string srcName, int srcIndex, TFOutput dst)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            TF_ImportGraphDefOptionsAddInputMapping(handle, srcName, srcIndex, dst);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern void TF_ImportGraphDefOptionsAddControlDependency(TF_ImportGraphDefOptions opts, TF_Operation oper);

        /// <summary>
        /// Cause the imported graph to have a control dependency on the provided operation.
        /// </summary>
        /// <param name="operation">This operation should exist in the graph being imported to.</param>
        public void AddControlDependency(TFOperation operation)
        {
            if (operation == null)
                throw new ArgumentNullException(nameof(operation));
            if (handle == IntPtr.Zero)
                ObjectDisposedException();

            TF_ImportGraphDefOptionsAddControlDependency(handle, operation.handle);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern void TF_ImportGraphDefOptionsAddReturnOutput(TF_ImportGraphDefOptions opts, string oper_name, int index);

        /// <summary>
        /// Add an output in the graph definition to be returned via the return outputs parameter.
        /// </summary>
        /// <param name="operName">Operation name.</param>
        /// <param name="index">Operation index.</param>
        /// <remarks>
        /// If the output is remapped via an input
        /// mapping, the corresponding existing tensor in graph will be returned.
        /// </remarks>
        public void AddReturnOutput(string operName, int index)
        {
            if (operName == null)
                throw new ArgumentNullException(nameof(operName));
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            TF_ImportGraphDefOptionsAddReturnOutput(handle, operName, index);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern int TF_ImportGraphDefOptionsNumReturnOutputs(TF_ImportGraphDefOptions opts);

        /// <summary>
        /// Gets the number return outputs added via AddReturnOutput.
        /// </summary>
        /// <value>The number return outputs.</value>
        public int NumReturnOutputs
        {
            get
            {
                if (handle == IntPtr.Zero)
                    ObjectDisposedException();
                return TF_ImportGraphDefOptionsNumReturnOutputs(handle);
            }
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern void TF_ImportGraphDefOptionsRemapControlDependency(TF_ImportGraphDefOptions opts, string srcName, TF_Operation dst);

        /// <summary>
        /// Sets any imported nodes with a given control input to have it replaced with an operation
        /// </summary>
        /// <param name="srcName">Node in the graph to be imported.</param>
        /// <param name="destination">References an operation that already exists in the graph being imported.</param>
        /// <remarks>
        /// Set any imported nodes with control input <paramref name="srcName"/> to have that input
        /// replaced with <paramref name="destination"/>.
        /// </remarks>
        public void RemapControlDependency(string srcName, TFOperation destination)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (srcName == null)
                throw new ArgumentNullException(nameof(srcName));
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (destination.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(destination));
            TF_ImportGraphDefOptionsRemapControlDependency(handle, srcName, destination.Handle);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern void TF_ImportGraphDefOptionsSetUniquifyNames(TF_ImportGraphDefOptions opts, byte uniquify);

        /// <summary>
        /// Set whether to uniquify imported operation names.
        /// </summary>
        /// <param name="uniquifyNames">If set to <c>true</c> imported operation names will be modified if their name already exists in the graph.
        /// If set to <c>false</c> conflicting names will be treated as an error.
        /// </param>
        /// <remarks>
        ///  Note that this option has no effect if a prefix is set, since the prefix will guarantee all names are
        ///  Defaults to false.
        /// </remarks>
        public void SetUniquifyNames(bool uniquifyNames)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();

            TF_ImportGraphDefOptionsSetUniquifyNames(handle, uniquifyNames ? (byte)1 : (byte)0);
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern void TF_ImportGraphDefOptionsSetUniquifyPrefix(TF_ImportGraphDefOptions opts, byte uniquify_prefix);

        /// <summary>
        /// Sets the uniquify prefix.  This option has no effect if no prefix is specified.
        /// </summary>
        /// <param name="uniquifyPrefix">If set to <c>true</c> the specified prefix will be modified if it already exists as an
        /// operation name or prefix in the graph.
        /// If set to <c>false</c> a conflicting prefix will be treated as an error.
        /// </param>
        public void SetUniquifyPrefix(bool uniquifyPrefix)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            TF_ImportGraphDefOptionsSetUniquifyPrefix(handle, uniquifyPrefix ? (byte)1 : (byte)0);
        }
    }

    /// <summary>
    /// Drives the execution of a graph
    /// </summary>
    /// <remarks>
    /// <para>
    /// This creates a new context to execute a TFGraph.   You can use the
    /// constructor to create an empty session, or you can load an existing
    /// model using the <see cref="FromSavedModel"/> static method in this class.
    /// </para>
    /// <para>
    /// To execute operations with the graph, call the <see cref="GetRunner"/>  method
    /// which returns an object that you can use to build the operation by providing
    /// the inputs, requesting the operations that you want to execute and the desired outputs.
    /// </para>
    /// <para>
    /// The <see cref="GetRunner"/> method is a high-level helper function that wraps a
    /// call to the <see cref="Run"/> method which just takes too many parameters that must
    /// be kept in sync.
    /// </para>
    /// </remarks>
    internal class TFSession : TFDisposableThreadSafe
    {
        // extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_Session TF_NewSession(TF_Graph graph, TF_SessionOptions opts, TF_Status status);

        /// <summary>
        /// Gets the graph associated with this TensorFlow session.
        /// </summary>
        /// <value>The graph.</value>
        public TFGraph Graph { get; private set; }

        private TFSession(IntPtr handle, TFGraph graph) : base(handle)
        {
            Graph = graph;
        }

        /// <summary>
        /// Creates a new execution session associated with the specified session graph with some configuration options.
        /// </summary>
        /// <param name="graph">The Graph to which this session is associated.</param>
        /// <param name="sessionOptions">Session options.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public TFSession(TFGraph graph, TFSessionOptions sessionOptions, TFStatus status = null) : base(IntPtr.Zero)
        {
            Graph = graph;
            var cstatus = TFStatus.Setup(status);
            var h = TF_NewSession(graph.handle, sessionOptions.handle, cstatus.handle);
            cstatus.CheckMaybeRaise(status);
            handle = h;
        }

        /// <summary>
        /// Creates a new execution session associated with the specified session graph.
        /// </summary>
        /// <param name="graph">The Graph to which this session is associated.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public TFSession(TFGraph graph, TFStatus status = null) : base(IntPtr.Zero)
        {
            Graph = graph;
            var cstatus = TFStatus.Setup(status);
            TF_Status h;
            using (var empty = new TFSessionOptions())
            {
                h = TF_NewSession(graph.handle, empty.Handle, cstatus.handle);
            }
            cstatus.CheckMaybeRaise(status);
            handle = h;
        }

        /// <summary>
        /// Creates a new execution session with an empty graph
        /// </summary>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        /// The created graph can be retrieved using the Graph property on the session.
        /// </remarks>
        public TFSession(TFStatus status = null) : this(new TFGraph(), status)
        {
        }

        // extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_Session TF_LoadSessionFromSavedModel(TF_SessionOptions session_options, LLBuffer* run_options, string export_dir, string[] tags, int tags_len, TF_Graph graph, LLBuffer* meta_graph_def, TF_Status status);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe TF_DeviceList TF_SessionListDevices(TF_Session session, TF_Status status);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe int TF_DeviceListCount(TF_DeviceList list);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe IntPtr TF_DeviceListName(TF_DeviceList list, int index, TF_Status status);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe IntPtr TF_DeviceListType(TF_DeviceList list, int index, TF_Status status);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe long TF_DeviceListMemoryBytes(TF_DeviceList list, int index, TF_Status status);

        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteDeviceList(TF_DeviceList list);

        /// <summary>
        /// Lists available devices in this session.
        /// </summary>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public IEnumerable<DeviceAttributes> ListDevices(TFStatus status = null)
        {
            var cstatus = TFStatus.Setup(status);
            var rawDeviceList = TF_SessionListDevices(Handle, cstatus.handle);
            var size = TF_DeviceListCount(rawDeviceList);

            var list = new List<DeviceAttributes>();
            for (var i = 0; i < size; i++)
            {
                var name = Marshal.PtrToStringAnsi(TF_DeviceListName(rawDeviceList, i, cstatus.handle));
                var deviceType = (DeviceType)Enum.Parse(typeof(DeviceType), Marshal.PtrToStringAnsi(TF_DeviceListType(rawDeviceList, i, cstatus.handle)));
                var memory = TF_DeviceListMemoryBytes(rawDeviceList, i, cstatus.handle);

                list.Add(new DeviceAttributes(name, deviceType, memory));
            }

            TF_DeleteDeviceList(rawDeviceList);

            return list;
        }

        /// <summary>
        /// Creates a session and graph from a model stored in the SavedModel file format.
        /// </summary>
        /// <returns>On success, this populates the provided <paramref name="graph"/> with the contents of the graph stored in the specified model and <paramref name="metaGraphDef"/> with the MetaGraphDef of the loaded model.</returns>
        /// <param name="sessionOptions">Session options to use for the new session.</param>
        /// <param name="runOptions">Options to use to initialize the state (can be null).</param>
        /// <param name="exportDir">must be set to the path of the exported SavedModel.</param>
        /// <param name="tags">must include the set of tags used to identify one MetaGraphDef in the SavedModel.</param>
        /// <param name="graph">This must be a newly created graph.</param>
        /// <param name="metaGraphDef">On success, this will be populated on return with the contents of the MetaGraphDef (can be null).</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        /// <para>
        /// This function creates a new session using the specified <paramref name="sessionOptions"/> and then initializes
        /// the state (restoring tensors and other assets) using <paramref name="runOptions"/>.
        /// </para>
        /// <para>
        /// This function loads the data that was saved using the SavedModel file format, as described
        /// here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
        /// </para>
        /// </remarks>
        public static TFSession FromSavedModel(TFSessionOptions sessionOptions, TFBuffer runOptions, string exportDir, string[] tags, TFGraph graph, TFBuffer metaGraphDef, TFStatus status = null)
        {
            if (graph == null)
                throw new ArgumentNullException(nameof(graph));
            if (tags == null)
                throw new ArgumentNullException(nameof(tags));
            if (exportDir == null)
                throw new ArgumentNullException(nameof(exportDir));
            if (metaGraphDef == null)
                throw new ArgumentNullException(nameof(metaGraphDef));
            var cstatus = TFStatus.Setup(status);
            unsafe
            {
                var h = TF_LoadSessionFromSavedModel(sessionOptions.handle, runOptions == null ? null : runOptions.LLBuffer, exportDir, tags, tags.Length, graph.handle, metaGraphDef == null ? null : metaGraphDef.LLBuffer, cstatus.handle);

                if (cstatus.CheckMaybeRaise(status))
                {
                    return new TFSession(h, graph);
                }
            }
            return null;
        }

        // extern void TF_CloseSession (TF_Session *, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_CloseSession(TF_Session session, TF_Status status);

        /// <summary>
        /// Closes the session.  Contacts any other processes associated with the session, if applicable.
        /// </summary>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        /// Can not be called after calling DeleteSession.
        /// </remarks>
        public void CloseSession(TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            var cstatus = TFStatus.Setup(status);
            TF_CloseSession(handle, cstatus.handle);
            cstatus.CheckMaybeRaise(status);
        }

        // extern void TF_DeleteSession (TF_Session *, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteSession(TF_Session session, TF_Status status);

        /// <summary>
        /// Deletes the session.
        /// </summary>
        /// <param name="status">Status.</param>
        public void DeleteSession(TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            var cstatus = TFStatus.Setup(status);
            TF_DeleteSession(handle, cstatus.handle);
            cstatus.CheckMaybeRaise(status);
        }

        internal override void NativeDispose(IntPtr handle)
        {
            using (var s = new TFStatus())
            {
                TF_DeleteSession(handle, s.handle);
            }
        }

        // extern void TF_SessionRun (TF_Session *session, const TF_Buffer *run_options, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Buffer *run_metadata, TF_Status *);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_SessionRun(TF_Session session, LLBuffer* run_options, TFOutput[] inputs, TF_Tensor[] input_values, int ninputs, TFOutput[] outputs, TF_Tensor[] output_values, int noutputs, TF_Operation[] target_opers, int ntargets, LLBuffer* run_metadata, TF_Status status);

        /// <summary>
        /// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
        /// to construct the parameters that will be passed to the TFSession.Run method.
        /// </para>
        /// <para>
        /// Instances of this class are created by calling the GetRunner method on the TFSession.
        /// </para>
        /// <para>
        /// The various methods in this class return an instance to the Runner itsel, to allow
        /// to easily construct chains of execution like this:
        /// </para>
        /// <code>
        /// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ();
        /// </code>
        /// <para>
        /// You do not need to chain the operations, this works just the same:
        /// </para>
        /// <code>
        /// runner = session.GetRunner ();
        /// runner.AddInput(myInput);
        /// runner.Fetch(myOutput);
        /// var results = runner.Run();
        /// </code>
        /// </remarks>
        public class Runner
        {
            private List<TFOutput> inputs;
            private List<TFOutput> outputs;
            private List<TFTensor> inputValues;
            private List<TFOperation> targets;
            private TFSession session;

            internal Runner(TFSession session)
            {
                inputs = new List<TFOutput>();
                outputs = new List<TFOutput>();
                inputValues = new List<TFTensor>();
                targets = new List<TFOperation>();
                this.session = session;
                RunMetadata = null;
                RunOptions = null;
            }

            /// <summary>
            /// Adds an input to the session
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="input">Incoming port.</param>
            /// <param name="value">Value to assing to the incoming port.</param>
            public Runner AddInput(TFOutput input, TFTensor value)
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));
                inputs.Add(input);
                inputValues.Add(value);
                return this;
            }

            /// <summary>
            /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
            /// <param name="value">Value to assing to the incoming port.</param>
            public Runner AddInput(string input, TFTensor value)
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));
                inputs.Add(ParseOutput(input));
                inputValues.Add(value);
                return this;
            }

            /// <summary>
            /// Adds the specified operations as the ones to be retrieved.
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="targets">One or more targets.</param>
            public Runner AddTarget(params TFOperation[] targets)
            {
                foreach (var t in targets)
                    this.targets.Add(t);
                return this;
            }

            // Parses user strings that contain both the operation name and an index.
            private TFOutput ParseOutput(string operation)
            {
                var p = operation.IndexOf(':');
                if (p != -1 && p != operation.Length - 1)
                {
                    var op = operation.Substring(0, p);
                    if (int.TryParse(operation.Substring(p + 1), out var idx))
                    {
                        return session.Graph[op][idx];
                    }
                }
                return session.Graph[operation][0];
            }

            /// <summary>
            /// Adds the specified operation names as the ones to be retrieved.
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="targetNames">One or more target names.</param>
            public Runner AddTarget(params string[] targetNames)
            {
                foreach (var tn in targetNames)
                    targets.Add(session.Graph[tn]);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the index-th output of the tensor referenced by operation.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="operation">The name of the operation in the graph.</param>
            /// <param name="index">The index of the output in the operation.</param>
            public Runner Fetch(string operation, int index)
            {
                var op = session.Graph[operation];
                outputs.Add(op[index]);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="operation">The name of the operation in the graph, which might be a simple name, or it might be name:index,
            /// where the index is the .</param>
            public Runner Fetch(string operation)
            {
                var op = ParseOutput(operation);
                outputs.Add(op);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of the tensor referenced by output
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="output">The output referencing a specified tensor.</param>
            public Runner Fetch(TFOutput output)
            {
                outputs.Add(output);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of all the tensor referenced by outputs.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="outputs">The outputs referencing a specified tensor.</param>
            public Runner Fetch(params TFOutput[] outputs)
            {
                foreach (var output in outputs)
                    this.outputs.Add(output);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of all the tensor referenced by outputs.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="outputs">The output sreferencing a specified tensor.</param>
            public Runner Fetch(params string[] outputs)
            {
                foreach (var output in outputs)
                    this.outputs.Add(ParseOutput(output));
                return this;
            }

            /// <summary>
            /// Protocol buffer encoded block containing the metadata passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
            /// </summary>
            public TFBuffer RunMetadata;

            /// <summary>
            /// Protocol buffer encoded block containing the run options passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
            /// </summary>
            public TFBuffer RunOptions;

            /// <summary>
            ///  Execute the graph fragments necessary to compute all requested fetches.
            /// </summary>
            /// <returns>One TFTensor for each call to Fetch that you made, in the order that you made them.</returns>
            /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
            public TFTensor[] Run(TFStatus status = null)
            {
                return session.Run(inputs.ToArray(), inputValues.ToArray(), outputs.ToArray(), targets.ToArray(), RunMetadata, RunOptions, status);
            }

            /// <summary>
            /// Run the specified operation, by adding it implicity to the output, single return value
            /// </summary>
            /// <param name="operation">The output of the operation.</param>
            /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
            /// <remarks>
            /// This method is a convenience method, and when you call it, it will clear any
            /// calls that you might have done to Fetch() and use the specified operation to Fetch
            /// instead.
            /// </remarks>
            public TFTensor Run(TFOutput operation, TFStatus status = null)
            {
                outputs.Clear();
                Fetch(operation);
                return Run(status)[0];
            }

        }

        /// <summary>
        /// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
        /// </summary>
        /// <returns>The runner.</returns>
        /// <remarks>
        /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
        /// to construct the parameters that will be passed to the TFSession.Run method.
        ///
        /// The Run method will return an array of TFTensor values, one for each invocation to the Fetch method.
        /// </remarks>
        public Runner GetRunner()
        {
            return new Runner(this);
        }

        /// <summary>
        /// Executes a pipeline given the specified inputs, inputValues, outputs, targetOpers, runMetadata and runOptions.
        /// A simpler API is available by calling the <see cref="M:GetRunner"/> method which performs all the bookkeeping
        /// necessary.
        /// </summary>
        /// <returns>An array of tensors fetched from the requested outputs.</returns>
        /// <param name="inputs">Inputs nodes.</param>
        /// <param name="inputValues">Input values.</param>
        /// <param name="outputs">Output nodes.</param>
        /// <param name="targetOpers">Target operations to execute.</param>
        /// <param name="runMetadata">Run metadata, a buffer containing the protocol buffer encoded value for https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/protobuf/config.proto.</param>
        /// <param name="runOptions">Run options, a buffer containing the protocol buffer encoded value for https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/protobuf/config.proto.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public TFTensor[] Run(TFOutput[] inputs, TFTensor[] inputValues, TFOutput[] outputs, TFOperation[] targetOpers = null, TFBuffer runMetadata = null, TFBuffer runOptions = null, TFStatus status = null)
        {
            if (handle == IntPtr.Zero)
                ObjectDisposedException();
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (inputValues == null)
                throw new ArgumentNullException(nameof(inputValues));
            if (outputs == null)
                throw new ArgumentNullException(nameof(outputs));
            int iLen = inputs.Length;
            if (iLen != inputValues.Length)
                throw new ArgumentException("inputs and inputValues have different lengths", "inputs");
            int oLen = outputs.Length;

            // runOptions and runMetadata might be null
            var cstatus = TFStatus.Setup(status);

            // Create arrays for the unmanaged versions
            var ivals = new IntPtr[iLen];
            for (int i = 0; i < iLen; i++)
                ivals[i] = inputValues[i].handle;

            // I believe this might not be necessary, the output values in TF_SessionRun looks like a write-only result
            var ovals = new IntPtr[outputs.Length];
            IntPtr[] topers = null;
            int tLen = 0;
            if (targetOpers != null)
            {
                tLen = targetOpers.Length;
                topers = new IntPtr[tLen];
                for (int i = 0; i < tLen; i++)
                    topers[i] = targetOpers[i].Handle;
            }

            unsafe
            {
                TF_SessionRun(handle, runOptions == null ? null : runOptions.LLBuffer, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, runMetadata == null ? null : runMetadata.LLBuffer, cstatus.handle);
            }
            cstatus.CheckMaybeRaise(status);

            // Ensure that the input tensors remain rooted, so that the GC won't collect & run finalizers between
            // when they are copied to ivals and TF_SessionRun is called.
            GC.KeepAlive(inputValues);

            var result = new TFTensor[oLen];
            for (int i = 0; i < oLen; i++)
            {
                result[i] = new TFTensor(ovals[i]);
            }
            return result;
        }
    }

    /// <summary>
    /// The data type for a specific tensor.
    /// </summary>
    /// <remarks>
    /// Tensors have uniform data types, all the elements of the tensor are of this
    /// type and they dictate how TensorFlow will treat the data stored.
    /// </remarks>
    internal enum TFDataType : uint
    {
        /// <summary>
        /// The TFDataType has not been set
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// Single precission floatint point, 32-bits (C# float)
        /// </summary>
        Float = 1,
        /// <summary>
        /// Double precission floatint point, 64-bits (C# double)
        /// </summary>
        Double = 2,
        /// <summary>
        /// 32-bit signed integers (C# int)
        /// </summary>
        Int32 = 3,
        /// <summary>
        /// 8 bit unsigned integers (C# byte)
        /// </summary>
        UInt8 = 4,
        /// <summary>
        /// 16-bit signed integers (C# short)
        /// </summary>
        Int16 = 5,
        /// <summary>
        /// 8-bit signed integers (C# sbyte)
        /// </summary>
        Int8 = 6,
        /// <summary>
        /// Binary blob
        /// </summary>
        String = 7,
        /// <summary>
        /// Single precission complex numbers (32-bit floats)
        /// </summary>
        Complex64 = 8,
        /// <summary>
        /// 32-bit float based complex numbers
        /// </summary>
        Complex = 8,
        /// <summary>
        /// 64-bit signed integers (C# long)
        /// </summary>
        Int64 = 9,
        /// <summary>
        /// 8-bit boolean (C# bool)
        /// </summary>
        Bool = 10,
        /// <summary>
        /// Quantized 8-bit signed integer
        /// </summary>
        QInt8 = 11,
        /// <summary>
        /// Quantized 8-bit unsigned integer
        /// </summary>
        QUInt8 = 12,
        /// <summary>
        /// Quantized 32-bit signed integer
        /// </summary>
        QInt32 = 13,
        /// <summary>
        /// Float32 truncated to 16 bits.  Only for cast operations.
        /// </summary>
        BFloat16 = 14,
        /// <summary>
        /// Quantized 16-bit signed integer
        /// </summary>
        QInt16 = 15,
        /// <summary>
        /// Quantized 16-bit unsigned integer
        /// </summary>
        QUInt16 = 16,
        /// <summary>
        /// 16-bit unsigned integers (C# long)
        /// </summary>
        UInt16 = 17,
        /// <summary>
        /// Double precission complex numbers (32-bit floats)
        /// </summary>
        Complex128 = 18,

        /// <summary>
        /// Half floats - 16-bit half precision floating point.
        /// </summary>
        Half = 19,

        /// <summary>
        /// Handle to a mutable resource.
        /// </summary>
        Resource = 20,

        /// <summary>
        /// Variant data type
        /// </summary>
        Variant = 21,

        /// <summary>
        /// 32-bit unsigned integers
        /// </summary>
        UInt32 = 22,

        /// <summary>
        /// 64-bit unsigned integers
        /// </summary>
        UInt64 = 23,

        /// <summary>
        /// Float reference type. It used for defining types of Variables.
        /// Please https://www.tensorflow.org/api_docs/python/tf/DType for more details.
        /// </summary>
        Float_ref = 101
    }

    /// <summary>
    /// Status code for invoking a tensorflow operation.
    /// </summary>
    internal enum TFCode : uint
    {
        /// <summary>
        /// Not an error; returned on success
        /// </summary>
        Ok = 0,
        /// <summary>
        /// The operation was cancelled (typically by the caller).
        /// </summary>
        Cancelled = 1,
        /// <summary>
        /// Unknown error.  An example of where this error may be returned is
        /// if a Status value received from another address space belongs to
        /// an error-space that is not known in this address space.  Also
        /// errors raised by APIs that do not return enough error information
        /// may be converted to this error.
        /// </summary>
        Unknown = 2,

        /// <summary>
        /// Client specified an invalid argument.  Note that this differs
        /// from FailedPrecondition.  InvalidArgumentindicates arguments
        /// that are problematic regardless of the state of the system
        /// (for example, a malformed file name).
        /// </summary>
        InvalidArgument = 3,

        /// <summary>
        /// Deadline expired before operation could complete.  For operations
        /// that change the state of the system, this error may be returned
        /// even if the operation has completed successfully.  For example, a
        /// successful response from a server could have been delayed long
        /// enough for the deadline to expire.
        /// </summary>
        DeadlineExceeded = 4,

        /// <summary>
        /// Some requested entity (for example, file or directory) was not found.
        /// For privacy reasons, this code may be returned when the client
        /// does not have the access right to the entity.
        /// </summary>
        NotFound = 5,

        /// <summary>
        /// Some entity that we attempted to create (for example, file or directory) already exists.
        /// </summary>
        AlreadyExists = 6,

        /// <summary>
        /// The caller does not have permission to execute the specified
        /// operation.  PermissionDenied must not be used for rejections
        /// caused by exhausting some resource (use ResourceExhausted
        /// instead for those errors).  PermissionDeniedmust not be
        /// used if the caller can not be identified (use Unauthenticated
        /// instead for those errors).
        /// </summary>
        PermissionDenied = 7,

        /// <summary>
        /// The request does not have valid authentication credentials for the
        /// operation.
        /// </summary>
        Unauthenticated = 16,

        /// <summary>
        /// Some resource has been exhausted, perhaps a per-user quota, or
        /// perhaps the entire file system is out of space.
        /// </summary>
        ResourceExhausted = 8,

        /// <summary>
        /// Operation was rejected because the system is not in a state
        /// required for the operation's execution.  For example, directory
        /// to be deleted may be non-empty, an rmdir operation is applied to
        /// a non-directory, etc.
        ///
        /// A litmus test that may help a service implementor in deciding
        /// between FailedPrecondition, Aborted, and Unavailable:
        ///
        ///  (a) Use Unavailableif the client can retry just the failing call.
        ///  (b) Use Aborted if the client should retry at a higher-level
        ///      (for example, restarting a read-modify-write sequence).
        ///  (c) Use FailedPrecondition if the client should not retry until
        ///      the system state has been explicitly fixed. For example, if an "rmdir"
        ///      fails because the directory is non-empty, FailedPrecondition
        ///      should be returned since the client should not retry unless
        ///      they have first fixed up the directory by deleting files from it.
        ///  (d) Use FailedPrecondition if the client performs conditional
        ///      REST Get/Update/Delete on a resource and the resource on the
        ///      server does not match the condition. For example, conflicting
        ///      read-modify-write on the same resource.
        /// </summary>
        FailedPrecondition = 9,

        /// <summary>
        /// The operation was aborted, typically due to a concurrency issue
        /// like sequencer check failures, transaction aborts, etc.
        ///
        /// See litmus test above for deciding between FailedPrecondition,
        /// Aborted and Unavailable
        /// </summary>
        Aborted = 10,

        /// <summary>
        /// Operation tried to iterate past the valid input range. For example, seeking or
        /// reading past end of file.
        ///
        /// Unlike InvalidArgument, this error indicates a problem that may
        /// be fixed if the system state changes. For example, a 32-bit file
        /// system will generate InvalidArgument if asked to read at an
        /// offset that is not in the range [0,2^32-1], but it will generate
        /// OutOfRange if asked to read from an offset past the current
        /// file size.
        ///
        /// There is a fair bit of overlap between FailedPrecondition and
        /// OutOfRange.  We recommend using OutOfRane (the more specific
        /// error) when it applies so that callers who are iterating through
        /// a space can easily look for an OutOfRange error to detect when
        /// they are done.
        /// </summary>
        OutOfRange = 11,

        /// <summary>
        /// Operation is not implemented or not supported/enabled in this service.
        /// </summary>
        Unimplemented = 12,

        /// <summary>
        /// Internal errors.  Means some invariants expected by underlying
        /// system has been broken.  If you see one of these errors,
        /// something is very broken.
        /// </summary>
        Internal = 13,

        /// <summary>
        /// The service is currently unavailable.  This is a most likely a
        /// transient condition and may be corrected by retrying with
        /// a backoff.
        ///
        /// See litmus test above for deciding between FailedPrecondition,
        /// Aborted, and Unavailable.
        /// </summary>
        Unavailable = 14,

        /// <summary>
        /// Unrecoverable data loss or corruption.
        /// </summary>
        DataLoss = 15
    }

    /// <summary>
    /// Represents a specific input of an operation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct TFInput
    {
        /// <summary>
        /// The operation that this input is for
        /// </summary>
        public unsafe TF_Operation Operation;

        /// <summary>
        /// The index of the output within the Operation
        /// </summary>
        public int Index;

        // extern TF_DataType TF_OperationInputType (TF_Input oper_in);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern TFDataType TF_OperationInputType(TFInput oper_in);

        public TFDataType InputType => TF_OperationInputType(this);

    }

    /// <summary>
    /// Represents a specific output of an operation on a tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TFOutput objects represent one of the outputs of an operation in the graph
    /// (TFGraph).  Outputs have a data type, and eventually a shape that you can
    /// retrieve by calling the <see cref="M:TensorFlow.TFGraph.GetShape"/> method.
    /// </para>
    /// <para>
    /// These can be passed as an input argument to a function for adding operations
    /// to a graph, or to the TFSession's Run and GetRunner method as values to be
    /// fetched.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal struct TFOutput
    {
        private unsafe TF_Operation LLOperation;

        /// <summary>
        /// The index of the output within the operation.
        /// </summary>
        public int Index;

        // extern int TF_OperationOutputNumConsumers (TF_Output oper_out);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern int TF_OperationOutputNumConsumers(TFOutput oper_out);

        /// <summary>
        /// Gets the number consumers.
        /// </summary>
        /// <value>The number consumers.</value>
        /// <remarks>
        /// This number can change when new operations are added to the graph.
        /// </remarks>
        public int NumConsumers => TF_OperationOutputNumConsumers(this);

        // extern TF_DataType TF_OperationOutputType (TF_Output oper_out);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern TFDataType TF_OperationOutputType(TFOutput oper_out);

        /// <summary>
        /// Gets the type of the output.
        /// </summary>
        /// <value>The type of the output.</value>
        public TFDataType OutputType => LLOperation == IntPtr.Zero ? TFDataType.Unknown : TF_OperationOutputType(this);

        /// <summary>
        /// Initializes a new TFOutput instance.
        /// </summary>
        /// <param name="operation">The operation to which to attach the output.</param>
        /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
        public TFOutput(TFOperation operation, int index = 0)
        {
            if (operation == null)
                throw new ArgumentNullException(nameof(operation));
            LLOperation = operation.Handle;
            Index = index;
        }

        /// <summary>
        /// Initializes a new TFOutput instance from another TFOutput
        /// </summary>
        /// <param name="output">The other TFOutput that is having its operation attached.</param>
        /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
        public TFOutput(TFOutput output, int index = 0)
        {
            if (output.LLOperation == null)
                throw new ArgumentNullException("Outputs does not have a valid operation pointer");
            LLOperation = output.LLOperation;
            Index = index;
        }

        // extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input *consumers, int max_consumers);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe int TF_OperationOutputConsumers(TFOutput oper_out, TFInput* consumers, int max_consumers);

        /// <summary>
        /// Get list of all current consumers of a specific output of an operation
        /// </summary>
        /// <value>The output consumers.</value>
        /// <remarks>
        /// A concurrent modification of the graph can increase the number of consumers of
        /// an operation.
        /// This can return null if the TFOutput does not point to a valid object.
        /// </remarks>
        public TFInput[] OutputConsumers
        {
            get
            {
                var result = new TFInput[NumConsumers];
                unsafe
                {
                    fixed (TFInput* first = &result[0])
                        TF_OperationOutputConsumers(this, first, result.Length);
                }
                return result;
            }
        }

        /// <summary>
        /// The associated operation.
        /// </summary>
        /// <value>The operation.</value>
        public TFOperation Operation => new TFOperation(null, LLOperation);

        /// <summary>
        /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFOutput"/>.
        /// </summary>
        /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFOutput"/>.</returns>
        public override string ToString()
        {
            return string.Format("[{3} Index={1} Operation={2} (0x{0:X})]", (long)LLOperation, Index, Operation, OutputType);
        }
    }

    /// <summary>
    /// Low-level: Enumeration describing the types of a metadata attribute
    /// </summary>
    internal enum TFAttributeType : uint
    {
        /// <summary>
        /// The type of the attribute is a string
        /// </summary>
        String = 0,

        /// <summary>
        /// The type of the attribute is an int.
        /// </summary>
        Int = 1,

        /// <summary>
        /// The type of the attribute is a float
        /// </summary>
        Float = 2,

        /// <summary>
        /// The type of the attribute is a bool.
        /// </summary>
        Bool = 3,

        /// <summary>
        /// The type of the attribute is a type.
        /// </summary>
        Type = 4,

        /// <summary>
        /// The type of the attribute is a tensor shape
        /// </summary>
        Shape = 5,

        /// <summary>
        /// The type of the attribute is a tensor
        /// </summary>
        Tensor = 6,

        /// <summary>
        /// The type of the attribute is a placeholder
        /// </summary>
        Placeholder = 7,

        /// <summary>
        /// The type of the attribute is a function
        /// </summary>
        Func = 8
    }

    /// <summary>
    /// Low-level: this describes the tensorflow type information for an attribute in the low-level attributes used by operations.
    /// </summary>
    /// <remarks>
    /// This is a low-level operation returned by the <see cref="M:TensorFlow.TFOperation.GetAttributeMetadata"/>.
    /// This is included for completeness, but is not generally used from C#, as you have access to the high-level
    /// bindings in the <see cref="T:TensorFlow.TFGraph"/> type.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal struct TFAttributeMetadata
    {
        private byte isList;
        public bool IsList => isList != 0;
        public long ListSize;
        public TFAttributeType Type;
        public long TotalSize;

        /// <summary>
        /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFAttributeMetadata"/>.
        /// </summary>
        /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFAttributeMetadata"/>.</returns>
        public override string ToString()
        {
            return string.Format($"[TFAttributeMetadata IsList={IsList} ListSize={ListSize} Type={Type} TotalSize={TotalSize}]");
        }
    }

    /// <summary>
    /// Represents the shape of a tensor, it describes how many dimensions the tensor has in a given axis
    /// </summary>
    /// <remarks>
    /// <para>
    /// The shapes can be created by calling the constructor with the number of dimensions
    /// in the shape.   The null value is used to specify that the shape is unknown,
    /// an empty array is used to create a scalar, and other values are used to specify
    /// the number of dimensions.
    /// </para>
    /// <para>
    /// For the Unknown case, you can use <see cref="P:TensorFlor.TFShape.Unknown"/>, for
    /// scalars, you can use the <see cref="P:TensorFlor.TFShape.Scalar"/> shape.
    /// </para>
    /// <para>
    /// To create a 2-element vector, use:
    /// new TFShape (2)
    /// </para>
    /// <para>
    /// To create a 2x3 matrix, use:
    /// new TFShape (2, 3)
    /// </para>
    /// <para>
    /// To create a shape with an unknown number of elements, you can pass the value
    /// -1.  This is typically used to indicate the shape of tensors that represent a
    /// variable-sized batch of values.
    /// </para>
    /// <para>
    /// To create a matrix with 4 columns and an unknown number of rows:
    /// var batch = new TFShape (-1, 4)
    /// </para>
    /// </remarks>
    internal class TFShape
    {
        /// <summary>
        /// Represents an unknown number of dimensions in the tensor.
        /// </summary>
        /// <value>The unknown.</value>
        public static TFShape Unknown => new TFShape(null);

        /// <summary>
        /// This shape is used to represent scalar values.
        /// </summary>
        /// <value>The scalar.</value>
        public static TFShape Scalar => new TFShape(new long[0]);

        internal long[] dims;

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFShape"/> class.
        /// </summary>
        /// <param name="args">This is a params argument, so you can provide multiple values to it.
        /// A null value means that this is an unknown shape, a single value is used to create a vector,
        /// two values are used to create a 2-D matrix and so on.
        /// </param>
        /// <remarks>
        ///
        /// </remarks>
        public TFShape(params long[] args)
        {
            dims = args;
        }

        /// <summary>
        /// Gets the length of the specified dimension in the tensor
        /// </summary>
        /// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
        /// <param name="dimension">Dimension.</param>
        public int GetLength(int dimension) => dims == null ? -1 : dims.GetLength(dimension);

        /// <summary>
        /// Number of dimensions represented by this shape.
        /// </summary>
        /// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
        public int NumDimensions => dims == null ? -1 : dims.Length;

        /// <summary>
        /// Gets a value indicating whether all the dimensions in the <see cref="T:TensorFlow.TFShape"/> are fully specified.
        /// </summary>
        /// <value><c>true</c> if is fully specified; otherwise, <c>false</c>.</value>
        public bool IsFullySpecified
        {
            get
            {
                if (dims == null)
                    return false;
                foreach (var j in dims)
                    if (j == -1)
                        return false;
                return true;
            }
        }

        /// <summary>
        /// Returns the shape as an array
        /// </summary>
        /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
        public long[] ToArray()
        {
            if (dims == null)
                return null;

            var ret = (long[])dims.Clone();
            return ret;
        }

        /// <summary>
        /// Returns the shape as an array
        /// </summary>
        /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
        public int[] ToIntArray()
        {
            if (dims == null)
                return null;

            var ret = new int[dims.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                checked
                {
                    ret[i] = (int)dims[i];
                }
            }
            return ret;
        }

        /// <summary>
        /// Gets a value indicating whether one of the dimensions <see cref="T:TensorFlow.TFShape"/> in the shape is larger than Int32.MaxValue.
        /// </summary>
        /// <value><c>true</c> if is long array; otherwise, <c>false</c>.</value>
        public bool IsLongArray
        {
            get
            {
                foreach (var l in dims)
                    if (l > Int32.MaxValue)
                        return true;

                return false;
            }
        }

        /// <summary>
        /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.
        /// </summary>
        /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.</returns>
        public override string ToString()
        {
            if (dims == null)
                return "unknown";
            return "[" + String.Join(", ", dims.Select(x => x == -1 ? "?" : x.ToString())) + "]";
        }

        /// <summary>
        /// Gets the dimensions for the specified index.
        /// </summary>
        /// <param name="idx">Index.</param>
        public long this[int idx] => dims[idx];

        /// <summary>
        /// Returns the shape as a 1-dimensional tensor with each element corresponding to the specified shape dimension.
        /// </summary>
        /// <returns>The tensor.</returns>
        public TFTensor AsTensor()
        {
            return new TFTensor(ToIntArray());
        }

        /// <summary>
        /// Adds a <see cref="TensorFlow.TFShape"/> to a <see cref="TensorFlow.TFShape"/>, yielding a shape made up of the concatenation of the first and the second shapes.
        /// </summary>
        /// <param name="left">The first <see cref="TensorFlow.TFShape"/> to add.</param>
        /// <param name="right">The second <see cref="TensorFlow.TFShape"/> to add.</param>
        /// <returns>The <see cref="T:TensorFlow.TFShape"/> that is the sum of the values of <c>left</c> and <c>right</c>.</returns>
        public static TFShape operator +(TFShape left, TFShape right)
        {
            if (left == null)
                return right;
            if (right == null)
                return left;

            var full = new long[left.dims.Length + right.dims.Length];
            Array.Copy(left.dims, full, left.dims.Length);
            Array.Copy(right.dims, 0, full, left.dims.Length, right.dims.Length);
            return new TFShape(full);
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="TFShape"/> to <see cref="TFTensor"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The result of the conversion.</returns>
        public static implicit operator TFTensor(TFShape shape)
        {
            return shape.AsTensor();
        }
    }
}
