// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Text;
using size_t = System.UIntPtr;

#pragma warning disable MSML_GeneralName
#pragma warning disable MSML_ParameterLocalVarName

namespace Microsoft.ML.Transforms.TensorFlow
{
    /// <summary>
    /// This attribute can be applied to callback functions that will be invoked
    /// from unmanaged code to managed code.
    /// </summary>
    /// <remarks>
    /// <code>
    /// [TensorFlow.MonoPInvokeCallback (typeof (BufferReleaseFunc))]
    /// internal static void MyFreeFunc (IntPtr data, IntPtr length){..}
    /// </code>
    /// </remarks>
    internal sealed class MonoPInvokeCallbackAttribute : Attribute
    {
        /// <summary>
        /// Use this constructor to annotate the type of the callback function that
        /// will be invoked from unmanaged code.
        /// </summary>
        /// <param name="t">T.</param>
        public MonoPInvokeCallbackAttribute(Type t) { }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LLBuffer
    {
        internal IntPtr data;
        internal size_t length;
        internal IntPtr data_deallocator;
    }

    /// <summary>
    /// Holds a block of data, suitable to pass, or retrieve from TensorFlow.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use the TFBuffer to blobs of data into TensorFlow, or to retrieve blocks
    /// of data out of TensorFlow.
    /// </para>
    /// <para>
    /// There are two constructors to wrap existing data, one to wrap blocks that are
    /// pointed to by an IntPtr and one that takes a byte array that we want to wrap.
    /// </para>
    /// <para>
    /// The empty constructor can be used to create a new TFBuffer that can be populated
    /// by the TensorFlow library and returned to user code.
    /// </para>
    /// <para>
    /// Typically, the data consists of a serialized protocol buffer, but other data
    /// may also be held in a buffer.
    /// </para>
    /// </remarks>
    // TODO: the string ctor
    // TODO: perhaps we should have an implicit byte [] conversion that just calls ToArray?
    internal class TFBuffer : TFDisposable
    {
        // extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe LLBuffer* TF_NewBufferFromString(IntPtr proto, IntPtr proto_len);

        // extern TF_Buffer * TF_NewBuffer ();
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe LLBuffer* TF_NewBuffer();

        internal TFBuffer(IntPtr handle) : base(handle) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> class.
        /// </summary>
        public unsafe TFBuffer() : base((IntPtr)TF_NewBuffer())
        {
        }

        /// <summary>
        /// Signature of the method that is invoked to release the data.
        /// </summary>
        /// <remarks>
        /// Methods of this signature are invoked with the data pointer and the
        /// lenght pointer when then TFBuffer no longer needs to hold on to the
        /// data.  If you are using this on platforms with static compilation
        /// like iOS, you need to annotate your callback with the MonoPInvokeCallbackAttribute,
        /// like this:
        ///
        /// <code>
        /// [TensorFlow.MonoPInvokeCallback (typeof (BufferReleaseFunc))]
        /// internal static void MyFreeFunc (IntPtr data, IntPtr length){..}
        /// </code>
        /// </remarks>
        public delegate void BufferReleaseFunc(IntPtr data, IntPtr lenght);

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by wrapping the unmanaged resource pointed by the buffer.
        /// </summary>
        /// <param name="buffer">Pointer to the data that will be wrapped.</param>
        /// <param name="size">The size of the buffer to wrap.</param>
        /// <param name="release">Optional, if not null, this method will be invoked to release the block.</param>
        /// <remarks>
        /// This constructor wraps the buffer as a the data to be held by the <see cref="T:TensorFlow.TFBuffer"/>,
        /// if the release parameter is null, then you must ensure that the data is not released before the TFBuffer
        /// is no longer in use.   If the value is not null, the provided method will be invoked to release
        /// the data when the TFBuffer is disposed, or the contents of the buffer replaced.
        /// </remarks>
        public unsafe TFBuffer(IntPtr buffer, long size, BufferReleaseFunc release) : base((IntPtr)TF_NewBuffer())
        {
            LLBuffer* buf = (LLBuffer*)handle;
            buf->data = buffer;
            buf->length = (size_t)size;
            if (release == null)
                buf->data_deallocator = IntPtr.Zero;
            else
                buf->data_deallocator = Marshal.GetFunctionPointerForDelegate(release);
        }

        [MonoPInvokeCallback(typeof(BufferReleaseFunc))]
        internal static void FreeBlock(IntPtr data, IntPtr length)
        {
            Marshal.FreeHGlobal(data);
        }

        internal static IntPtr FreeBufferFunc;
        internal static BufferReleaseFunc FreeBlockDelegate;

        static TFBuffer()
        {
            FreeBlockDelegate = FreeBlock;
            FreeBufferFunc = Marshal.GetFunctionPointerForDelegate<BufferReleaseFunc>(FreeBlockDelegate);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
        /// </summary>
        /// <param name="buffer">Buffer of data that will be wrapped.</param>
        /// <remarks>
        /// This constructor makes a copy of the data into an unmanaged buffer,
        /// so the byte array is not pinned.
        /// </remarks>
        public TFBuffer(byte[] buffer) : this(buffer, 0, buffer.Length) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
        /// </summary>
        /// <param name="buffer">Buffer of data that will be wrapped.</param>
        /// <param name="start">Starting offset into the buffer to wrap.</param>
        /// <param name="count">Number of bytes from the buffer to keep.</param>
        /// <remarks>
        /// This constructor makes a copy of the data into an unmanaged buffer,
        /// so the byte array is not pinned.
        /// </remarks>
        public TFBuffer(byte[] buffer, int start, int count) : this()
        {
            if (start < 0 || start >= buffer.Length)
                throw new ArgumentException("start");
            if (count < 0 || count > buffer.Length - start)
                throw new ArgumentException("count");
            unsafe
            {
                LLBuffer* buf = LLBuffer;
                buf->data = Marshal.AllocHGlobal(count);
                Marshal.Copy(buffer, start, buf->data, count);
                buf->length = (size_t)count;
                buf->data_deallocator = FreeBufferFunc;
            }
        }

        internal unsafe LLBuffer* LLBuffer => (LLBuffer*)handle;

        // extern void TF_DeleteBuffer (TF_Buffer *);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe void TF_DeleteBuffer(LLBuffer* buffer);

        internal override void NativeDispose(IntPtr handle)
        {
            unsafe { TF_DeleteBuffer((LLBuffer*)handle); }
        }

        // extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        private static extern unsafe LLBuffer TF_GetBuffer(LLBuffer* buffer);

        /// <summary>
        /// Returns a byte array representing the data wrapped by this buffer.
        /// </summary>
        /// <returns>The array.</returns>
        public byte[] ToArray()
        {
            if (handle == IntPtr.Zero)
                return null;

            unsafe
            {
                var lb = (LLBuffer*)handle;

                var result = new byte[(int)lb->length];
                Marshal.Copy(lb->data, result, 0, (int)lb->length);

                return result;
            }
        }
    }
}
