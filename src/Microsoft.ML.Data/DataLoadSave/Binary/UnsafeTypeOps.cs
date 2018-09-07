// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    /// <summary>
    /// Represents some common global operations over a type
    /// including many unsafe operations.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal abstract class UnsafeTypeOps<T>
    {
        public abstract int Size { get; }
        public abstract void Apply(T[] array, Action<IntPtr> func);
        public abstract void Write(T a, BinaryWriter writer);
        public abstract T Read(BinaryReader reader);
    }

    internal static class UnsafeTypeOpsFactory
    {
        private static Dictionary<Type, object> _type2ops;

        static UnsafeTypeOpsFactory()
        {
            _type2ops = new Dictionary<Type, object>();
            _type2ops[typeof(sbyte)] = new SByteUnsafeTypeOps();
            _type2ops[typeof(Byte)] = new ByteUnsafeTypeOps();
            _type2ops[typeof(short)] = new Int16UnsafeTypeOps();
            _type2ops[typeof(UInt16)] = new UInt16UnsafeTypeOps();
            _type2ops[typeof(int)] = new Int32UnsafeTypeOps();
            _type2ops[typeof(UInt32)] = new UInt32UnsafeTypeOps();
            _type2ops[typeof(long)] = new Int64UnsafeTypeOps();
            _type2ops[typeof(UInt64)] = new UInt64UnsafeTypeOps();
            _type2ops[typeof(Single)] = new SingleUnsafeTypeOps();
            _type2ops[typeof(Double)] = new DoubleUnsafeTypeOps();
            _type2ops[typeof(TimeSpan)] = new TimeSpanUnsafeTypeOps();
            _type2ops[typeof(UInt128)] = new UgUnsafeTypeOps();
        }

        public static UnsafeTypeOps<T> Get<T>()
        {
            return (UnsafeTypeOps<T>)_type2ops[typeof(T)];
        }

        private sealed class SByteUnsafeTypeOps : UnsafeTypeOps<sbyte>
        {
            public override int Size { get { return sizeof(sbyte); } }
            public override unsafe void Apply(sbyte[] array, Action<IntPtr> func)
            {
                fixed (sbyte* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(sbyte a, BinaryWriter writer) { writer.Write(a); }
            public override sbyte Read(BinaryReader reader) { return reader.ReadSByte(); }
        }

        private sealed class ByteUnsafeTypeOps : UnsafeTypeOps<Byte>
        {
            public override int Size { get { return sizeof(Byte); } }
            public override unsafe void Apply(Byte[] array, Action<IntPtr> func)
            {
                fixed (Byte* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(Byte a, BinaryWriter writer) { writer.Write(a); }
            public override Byte Read(BinaryReader reader) { return reader.ReadByte(); }
        }

        private sealed class Int16UnsafeTypeOps : UnsafeTypeOps<short>
        {
            public override int Size { get { return sizeof(short); } }
            public override unsafe void Apply(short[] array, Action<IntPtr> func)
            {
                fixed (short* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(short a, BinaryWriter writer) { writer.Write(a); }
            public override short Read(BinaryReader reader) { return reader.ReadInt16(); }
        }

        private sealed class UInt16UnsafeTypeOps : UnsafeTypeOps<UInt16>
        {
            public override int Size { get { return sizeof(UInt16); } }
            public override unsafe void Apply(UInt16[] array, Action<IntPtr> func)
            {
                fixed (UInt16* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(UInt16 a, BinaryWriter writer) { writer.Write(a); }
            public override UInt16 Read(BinaryReader reader) { return reader.ReadUInt16(); }
        }

        private sealed class Int32UnsafeTypeOps : UnsafeTypeOps<int>
        {
            public override int Size { get { return sizeof(int); } }
            public override unsafe void Apply(int[] array, Action<IntPtr> func)
            {
                fixed (int* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(int a, BinaryWriter writer) { writer.Write(a); }
            public override int Read(BinaryReader reader) { return reader.ReadInt32(); }
        }

        private sealed class UInt32UnsafeTypeOps : UnsafeTypeOps<UInt32>
        {
            public override int Size { get { return sizeof(UInt32); } }
            public override unsafe void Apply(UInt32[] array, Action<IntPtr> func)
            {
                fixed (UInt32* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(UInt32 a, BinaryWriter writer) { writer.Write(a); }
            public override UInt32 Read(BinaryReader reader) { return reader.ReadUInt32(); }
        }

        private sealed class Int64UnsafeTypeOps : UnsafeTypeOps<long>
        {
            public override int Size { get { return sizeof(long); } }
            public override unsafe void Apply(long[] array, Action<IntPtr> func)
            {
                fixed (long* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(long a, BinaryWriter writer) { writer.Write(a); }
            public override long Read(BinaryReader reader) { return reader.ReadInt64(); }
        }

        private sealed class UInt64UnsafeTypeOps : UnsafeTypeOps<UInt64>
        {
            public override int Size { get { return sizeof(UInt64); } }
            public override unsafe void Apply(UInt64[] array, Action<IntPtr> func)
            {
                fixed (UInt64* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(UInt64 a, BinaryWriter writer) { writer.Write(a); }
            public override UInt64 Read(BinaryReader reader) { return reader.ReadUInt64(); }
        }

        private sealed class SingleUnsafeTypeOps : UnsafeTypeOps<Single>
        {
            public override int Size { get { return sizeof(Single); } }
            public override unsafe void Apply(Single[] array, Action<IntPtr> func)
            {
                fixed (Single* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(Single a, BinaryWriter writer) { writer.Write(a); }
            public override Single Read(BinaryReader reader) { return reader.ReadSingle(); }
        }

        private sealed class DoubleUnsafeTypeOps : UnsafeTypeOps<Double>
        {
            public override int Size { get { return sizeof(Double); } }
            public override unsafe void Apply(Double[] array, Action<IntPtr> func)
            {
                fixed (Double* pArray = array)
                    func(new IntPtr(pArray));
            }
            public override void Write(Double a, BinaryWriter writer) { writer.Write(a); }
            public override Double Read(BinaryReader reader) { return reader.ReadDouble(); }
        }

        private sealed class TimeSpanUnsafeTypeOps : UnsafeTypeOps<TimeSpan>
        {
            public override int Size { get { return sizeof(long); } }
            public override unsafe void Apply(TimeSpan[] array, Action<IntPtr> func)
            {
                fixed (TimeSpan* pArray = array)
                    func(new IntPtr(pArray));
            }

            public override void Write(TimeSpan a, BinaryWriter writer) { writer.Write(a.Ticks); }
            public override TimeSpan Read(BinaryReader reader)
            {
                var ticks = reader.ReadInt64();
                return new TimeSpan(ticks == long.MinValue ? default : ticks);
            }
        }

        private sealed class UgUnsafeTypeOps : UnsafeTypeOps<UInt128>
        {
            public override int Size { get { return 2 * sizeof(ulong); } }
            public override unsafe void Apply(UInt128[] array, Action<IntPtr> func)
            {
                fixed (UInt128* pArray = array)
                    func(new IntPtr(pArray));
            }

            public override void Write(UInt128 a, BinaryWriter writer) { writer.Write(a.Lo); writer.Write(a.Hi); }
            public override UInt128 Read(BinaryReader reader)
            {
                ulong lo = reader.ReadUInt64();
                ulong hi = reader.ReadUInt64();
                return new UInt128(lo, hi);
            }
        }
    }
}
