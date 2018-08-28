// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data.Conversion
{
    using BL = Boolean;
    using DT = DvDateTime;
    using DZ = DvDateTimeZone;
    using I1 = DvInt1;
    using I2 = DvInt2;
    using I4 = DvInt4;
    using I8 = DvInt8;
    using R4 = Single;
    using R8 = Double;
    using RawI1 = SByte;
    using RawI2 = Int16;
    using RawI4 = Int32;
    using RawI8 = Int64;
    using SB = StringBuilder;
    using TS = DvTimeSpan;
    using TX = DvText;
    using U1 = Byte;
    using U2 = UInt16;
    using U4 = UInt32;
    using U8 = UInt64;
    using UG = UInt128;

    public delegate bool TryParseMapper<T>(ref TX src, out T dst);

    /// <summary>
    /// This type exists to provide efficient delegates for conversion between standard ColumnTypes,
    /// as discussed in the IDataView Type System Specification. This is a singleton class.
    /// Some conversions are "standard" conversions, conforming to the details in the spec.
    /// Others are auxilliary conversions. The use of auxilliary conversions should be limited to
    /// situations that genuinely require them and have been well designed in the particular context.
    /// For example, this contains non-standard conversions from the standard primitive types to
    /// text (and StringBuilder). These are needed by the standard TextSaver, which handles
    /// differences between sparse and dense inputs in a semantically invariant way.
    /// </summary>
    public sealed class Conversions
    {
        // REVIEW: Reconcile implementations with TypeUtils, and clarify the distinction.

        // Singleton pattern.
        private static volatile Conversions _instance;
        public static Conversions Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new Conversions(), null);
                return _instance;
            }
        }

        private const DataKind _kindStringBuilder = (DataKind)100;
        private readonly Dictionary<Type, DataKind> _kinds;

        // Maps from {src,dst} pair of DataKind to ValueMapper. The {src,dst} pair is
        // the two byte values packed into the low two bytes of an int, with src the lsb.
        private readonly Dictionary<int, Delegate> _delegatesStd;

        // Maps from {src,dst} pair of DataKind to ValueMapper. The {src,dst} pair is
        // the two byte values packed into the low two bytes of an int, with src the lsb.
        private readonly Dictionary<int, Delegate> _delegatesAll;

        // This has RefPredicate<T> delegates for determining whether a value is NA.
        private readonly Dictionary<DataKind, Delegate> _isNADelegates;

        // This has RefPredicate<VBuffer<T>> delegates for determining whether a buffer contains any NA values.
        private readonly Dictionary<DataKind, Delegate> _hasNADelegates;

        // This has RefPredicate<T> delegates for determining whether a value is default.
        private readonly Dictionary<DataKind, Delegate> _isDefaultDelegates;

        // This has RefPredicate<VBuffer<T>> delegates for determining whether a buffer contains any zero values.
        // The supported types are unsigned signed integer values (for determining whether a key type is NA).
        private readonly Dictionary<DataKind, Delegate> _hasZeroDelegates;

        // This has ValueGetter<T> delegates for producing an NA value of the given type.
        private readonly Dictionary<DataKind, Delegate> _getNADelegates;

        // This has TryParseMapper<T> delegates for parsing values from text.
        private readonly Dictionary<DataKind, Delegate> _tryParseDelegates;

        private Conversions()
        {
            // We fabricate a DataKind value for StringBuilder.
            Contracts.Assert(!Enum.IsDefined(typeof(DataKind), _kindStringBuilder));

            _kinds = new Dictionary<Type, DataKind>();
            for (DataKind kind = DataKindExtensions.KindMin; kind < DataKindExtensions.KindLim; kind++)
                _kinds.Add(kind.ToType(), kind);

            // We don't put StringBuilder in _kinds, but there are conversions to StringBuilder.
            Contracts.Assert(!_kinds.ContainsKey(typeof(StringBuilder)));
            Contracts.Assert(_kinds.Count == 16);

            _delegatesStd = new Dictionary<int, Delegate>();
            _delegatesAll = new Dictionary<int, Delegate>();
            _isNADelegates = new Dictionary<DataKind, Delegate>();
            _hasNADelegates = new Dictionary<DataKind, Delegate>();
            _isDefaultDelegates = new Dictionary<DataKind, Delegate>();
            _hasZeroDelegates = new Dictionary<DataKind, Delegate>();
            _getNADelegates = new Dictionary<DataKind, Delegate>();
            _tryParseDelegates = new Dictionary<DataKind, Delegate>();

            // !!! WARNING !!!: Do NOT add any standard conversions without clearing from the IDV Type System
            // design committee. Any changes also require updating the IDV Type System Specification.

            AddStd<I1, I1>(Convert);
            AddStd<I1, I2>(Convert);
            AddStd<I1, I4>(Convert);
            AddStd<I1, I8>(Convert);
            AddStd<I1, R4>(Convert);
            AddStd<I1, R8>(Convert);
            AddAux<I1, SB>(Convert);

            AddStd<I2, I1>(Convert);
            AddStd<I2, I2>(Convert);
            AddStd<I2, I4>(Convert);
            AddStd<I2, I8>(Convert);
            AddStd<I2, R4>(Convert);
            AddStd<I2, R8>(Convert);
            AddAux<I2, SB>(Convert);

            AddStd<I4, I1>(Convert);
            AddStd<I4, I2>(Convert);
            AddStd<I4, I4>(Convert);
            AddStd<I4, I8>(Convert);
            AddStd<I4, R4>(Convert);
            AddStd<I4, R8>(Convert);
            AddAux<I4, SB>(Convert);

            AddStd<I8, I1>(Convert);
            AddStd<I8, I2>(Convert);
            AddStd<I8, I4>(Convert);
            AddStd<I8, I8>(Convert);
            AddStd<I8, R4>(Convert);
            AddStd<I8, R8>(Convert);
            AddAux<I8, SB>(Convert);

            AddStd<U1, U1>(Convert);
            AddStd<U1, U2>(Convert);
            AddStd<U1, U4>(Convert);
            AddStd<U1, U8>(Convert);
            AddStd<U1, UG>(Convert);
            AddStd<U1, R4>(Convert);
            AddStd<U1, R8>(Convert);
            AddAux<U1, SB>(Convert);

            AddStd<U2, U1>(Convert);
            AddStd<U2, U2>(Convert);
            AddStd<U2, U4>(Convert);
            AddStd<U2, U8>(Convert);
            AddStd<U2, UG>(Convert);
            AddStd<U2, R4>(Convert);
            AddStd<U2, R8>(Convert);
            AddAux<U2, SB>(Convert);

            AddStd<U4, U1>(Convert);
            AddStd<U4, U2>(Convert);
            AddStd<U4, U4>(Convert);
            AddStd<U4, U8>(Convert);
            AddStd<U4, UG>(Convert);
            AddStd<U4, R4>(Convert);
            AddStd<U4, R8>(Convert);
            AddAux<U4, SB>(Convert);

            AddStd<U8, U1>(Convert);
            AddStd<U8, U2>(Convert);
            AddStd<U8, U4>(Convert);
            AddStd<U8, U8>(Convert);
            AddStd<U8, UG>(Convert);
            AddStd<U8, R4>(Convert);
            AddStd<U8, R8>(Convert);
            AddAux<U8, SB>(Convert);

            AddStd<UG, U1>(Convert);
            AddStd<UG, U2>(Convert);
            AddStd<UG, U4>(Convert);
            AddStd<UG, U8>(Convert);
            // REVIEW: Conversion from UG to R4/R8, should we?
            AddAux<UG, SB>(Convert);

            AddStd<R4, R4>(Convert);
            AddStd<R4, R8>(Convert);
            AddAux<R4, SB>(Convert);

            AddStd<R8, R4>(Convert);
            AddStd<R8, R8>(Convert);
            AddAux<R8, SB>(Convert);

            AddStd<TX, I1>(Convert);
            AddStd<TX, U1>(Convert);
            AddStd<TX, I2>(Convert);
            AddStd<TX, U2>(Convert);
            AddStd<TX, I4>(Convert);
            AddStd<TX, U4>(Convert);
            AddStd<TX, I8>(Convert);
            AddStd<TX, U8>(Convert);
            AddStd<TX, UG>(Convert);
            AddStd<TX, R4>(Convert);
            AddStd<TX, R8>(Convert);
            AddStd<TX, TX>(Convert);
            AddStd<TX, BL>(Convert);
            AddAux<TX, SB>(Convert);
            AddStd<TX, TS>(Convert);
            AddStd<TX, DT>(Convert);
            AddStd<TX, DZ>(Convert);

            AddStd<BL, I1>(Convert);
            AddStd<BL, I2>(Convert);
            AddStd<BL, I4>(Convert);
            AddStd<BL, I8>(Convert);
            AddStd<BL, R4>(Convert);
            AddStd<BL, R8>(Convert);
            AddStd<BL, BL>(Convert);
            AddAux<BL, SB>(Convert);

            AddStd<TS, I8>(Convert);
            AddStd<TS, R4>(Convert);
            AddStd<TS, R8>(Convert);
            AddAux<TS, SB>(Convert);

            AddStd<DT, I8>(Convert);
            AddStd<DT, R4>(Convert);
            AddStd<DT, R8>(Convert);
            AddAux<DT, SB>(Convert);

            AddStd<DZ, I8>(Convert);
            AddStd<DZ, R4>(Convert);
            AddStd<DZ, R8>(Convert);
            AddAux<DZ, SB>(Convert);

            AddIsNA<I1>(IsNA);
            AddIsNA<I2>(IsNA);
            AddIsNA<I4>(IsNA);
            AddIsNA<I8>(IsNA);
            AddIsNA<R4>(IsNA);
            AddIsNA<R8>(IsNA);
            AddIsNA<TX>(IsNA);
            AddIsNA<TS>(IsNA);
            AddIsNA<DT>(IsNA);
            AddIsNA<DZ>(IsNA);

            AddGetNA<I1>(GetNA);
            AddGetNA<I2>(GetNA);
            AddGetNA<I4>(GetNA);
            AddGetNA<I8>(GetNA);
            AddGetNA<R4>(GetNA);
            AddGetNA<R8>(GetNA);
            AddGetNA<TX>(GetNA);
            AddGetNA<TS>(GetNA);
            AddGetNA<DT>(GetNA);
            AddGetNA<DZ>(GetNA);

            AddHasNA<I1>(HasNA);
            AddHasNA<I2>(HasNA);
            AddHasNA<I4>(HasNA);
            AddHasNA<I8>(HasNA);
            AddHasNA<R4>(HasNA);
            AddHasNA<R8>(HasNA);
            AddHasNA<TX>(HasNA);
            AddHasNA<TS>(HasNA);
            AddHasNA<DT>(HasNA);
            AddHasNA<DZ>(HasNA);

            AddIsDef<I1>(IsDefault);
            AddIsDef<I2>(IsDefault);
            AddIsDef<I4>(IsDefault);
            AddIsDef<I8>(IsDefault);
            AddIsDef<R4>(IsDefault);
            AddIsDef<R8>(IsDefault);
            AddIsDef<BL>(IsDefault);
            AddIsDef<TX>(IsDefault);
            AddIsDef<U1>(IsDefault);
            AddIsDef<U2>(IsDefault);
            AddIsDef<U4>(IsDefault);
            AddIsDef<U8>(IsDefault);
            AddIsDef<UG>(IsDefault);
            AddIsDef<TS>(IsDefault);
            AddIsDef<DT>(IsDefault);
            AddIsDef<DZ>(IsDefault);

            AddHasZero<U1>(HasZero);
            AddHasZero<U2>(HasZero);
            AddHasZero<U4>(HasZero);
            AddHasZero<U8>(HasZero);

            AddTryParse<I1>(TryParse);
            AddTryParse<I2>(TryParse);
            AddTryParse<I4>(TryParse);
            AddTryParse<I8>(TryParse);
            AddTryParse<U1>(TryParse);
            AddTryParse<U2>(TryParse);
            AddTryParse<U4>(TryParse);
            AddTryParse<U8>(TryParse);
            AddTryParse<UG>(TryParse);
            AddTryParse<R4>(TryParse);
            AddTryParse<R8>(TryParse);
            AddTryParse<BL>(TryParse);
            AddTryParse<TX>(TryParse);
            AddTryParse<TS>(TryParse);
            AddTryParse<DT>(TryParse);
            AddTryParse<DZ>(TryParse);
        }

        private static int GetKey(DataKind kindSrc, DataKind kindDst)
        {
            Contracts.Assert(Enum.IsDefined(typeof(DataKind), kindSrc));
            Contracts.Assert(Enum.IsDefined(typeof(DataKind), kindDst) || kindDst == _kindStringBuilder);
            Contracts.Assert(0 <= _kindStringBuilder && (int)_kindStringBuilder < (1 << 8));
            return ((int)kindSrc << 8) | (int)kindDst;
        }

        // Add a standard conversion to the lookup tables.
        private void AddStd<TSrc, TDst>(ValueMapper<TSrc, TDst> fn)
        {
            var kindSrc = _kinds[typeof(TSrc)];
            var kindDst = _kinds[typeof(TDst)];
            var key = GetKey(kindSrc, kindDst);
            _delegatesStd.Add(key, fn);
            _delegatesAll.Add(key, fn);
        }

        // Add a non-standard conversion to the lookup table.
        private void AddAux<TSrc, TDst>(ValueMapper<TSrc, TDst> fn)
        {
            var kindSrc = _kinds[typeof(TSrc)];
            var kindDst = typeof(TDst) == typeof(SB) ? _kindStringBuilder : _kinds[typeof(TDst)];
            _delegatesAll.Add(GetKey(kindSrc, kindDst), fn);
        }

        private void AddIsNA<T>(RefPredicate<T> fn)
        {
            var kind = _kinds[typeof(T)];
            _isNADelegates.Add(kind, fn);
        }

        private void AddGetNA<T>(ValueGetter<T> fn)
        {
            var kind = _kinds[typeof(T)];
            _getNADelegates.Add(kind, fn);
        }

        private void AddHasNA<T>(RefPredicate<VBuffer<T>> fn)
        {
            var kind = _kinds[typeof(T)];
            _hasNADelegates.Add(kind, fn);
        }

        private void AddIsDef<T>(RefPredicate<T> fn)
        {
            var kind = _kinds[typeof(T)];
            _isDefaultDelegates.Add(kind, fn);
        }

        private void AddHasZero<T>(RefPredicate<VBuffer<T>> fn)
        {
            var kind = _kinds[typeof(T)];
            _hasZeroDelegates.Add(kind, fn);
        }

        private void AddTryParse<T>(TryParseMapper<T> fn)
        {
            var kind = _kinds[typeof(T)];
            _tryParseDelegates.Add(kind, fn);
        }

        /// <summary>
        /// Return a standard conversion delegate from typeSrc to typeDst. If there is no such standard
        /// conversion, this throws an exception.
        /// </summary>
        public ValueMapper<TSrc, TDst> GetStandardConversion<TSrc, TDst>(ColumnType typeSrc, ColumnType typeDst,
            out bool identity)
        {
            ValueMapper<TSrc, TDst> conv;
            if (!TryGetStandardConversion(typeSrc, typeDst, out conv, out identity))
                throw Contracts.Except("No standard conversion from '{0}' to '{1}'", typeSrc, typeDst);
            return conv;
        }

        /// <summary>
        /// Determine whether there is a standard conversion from typeSrc to typeDst and if so,
        /// set conv to the conversion delegate. The type parameters TSrc and TDst must match
        /// the raw types of typeSrc and typeDst.
        /// </summary>
        public bool TryGetStandardConversion<TSrc, TDst>(ColumnType typeSrc, ColumnType typeDst,
            out ValueMapper<TSrc, TDst> conv, out bool identity)
        {
            Contracts.CheckValue(typeSrc, nameof(typeSrc));
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.Check(typeSrc.RawType == typeof(TSrc));
            Contracts.Check(typeDst.RawType == typeof(TDst));

            Delegate del;
            if (!TryGetStandardConversion(typeSrc, typeDst, out del, out identity))
            {
                conv = null;
                return false;
            }
            conv = (ValueMapper<TSrc, TDst>)del;
            return true;
        }

        /// <summary>
        /// Return a standard conversion delegate from typeSrc to typeDst. If there is no such standard
        /// conversion, this throws an exception.
        /// </summary>
        public Delegate GetStandardConversion(ColumnType typeSrc, ColumnType typeDst)
        {
            bool identity;
            Delegate conv;
            if (!TryGetStandardConversion(typeSrc, typeDst, out conv, out identity))
                throw Contracts.Except("No standard conversion from '{0}' to '{1}'", typeSrc, typeDst);
            return conv;
        }

        /// <summary>
        /// Determine whether there is a standard conversion from typeSrc to typeDst and if so,
        /// set conv to the conversion delegate.
        /// </summary>
        public bool TryGetStandardConversion(ColumnType typeSrc, ColumnType typeDst,
            out Delegate conv, out bool identity)
        {
            Contracts.CheckValue(typeSrc, nameof(typeSrc));
            Contracts.CheckValue(typeDst, nameof(typeDst));

            conv = null;
            identity = false;
            if (typeSrc.IsKey)
            {
                var keySrc = typeSrc.AsKey;

                // Key types are only convertable to compatible key types or unsigned integer
                // types that are large enough.
                if (typeDst.IsKey)
                {
                    var keyDst = typeDst.AsKey;
                    // We allow the Min value to shift. We currently don't allow the counts to vary.
                    // REVIEW: Should we allow the counts to vary? Allowing the dst to be bigger is trivial.
                    // Smaller dst means mapping values to NA.
                    if (keySrc.Count != keyDst.Count)
                        return false;
                    if (keySrc.Count == 0 && keySrc.RawKind > keyDst.RawKind)
                        return false;
                    // REVIEW: Should we allow contiguous to be changed when Count is zero?
                    if (keySrc.Contiguous != keyDst.Contiguous)
                        return false;
                }
                else
                {
                    // Technically there is no standard conversion from a key type to an unsigned integer type,
                    // but it's very convenient for client code, so we allow it here. Note that ConvertTransform
                    // does not allow this.
                    if (!KeyType.IsValidDataKind(typeDst.RawKind))
                        return false;
                    if (keySrc.RawKind > typeDst.RawKind)
                    {
                        if (keySrc.Count == 0)
                            return false;
                        if ((ulong)keySrc.Count > typeDst.RawKind.ToMaxInt())
                            return false;
                    }
                }

                // REVIEW: Should we look for illegal values and force them to zero? If so, then
                // we'll need to set identity to false.
            }
            else if (typeDst.IsKey)
            {
                if (!typeSrc.IsText)
                    return false;
                conv = GetKeyParse(typeDst.AsKey);
                return true;
            }
            else if (!typeDst.IsStandardScalar)
                return false;

            Contracts.Assert(typeSrc.RawKind != 0);
            Contracts.Assert(typeDst.RawKind != 0);

            int key = GetKey(typeSrc.RawKind, typeDst.RawKind);
            identity = typeSrc.RawKind == typeDst.RawKind;
            return _delegatesStd.TryGetValue(key, out conv);
        }

        public ValueMapper<TSrc, SB> GetStringConversion<TSrc>(ColumnType type)
        {
            ValueMapper<TSrc, SB> conv;
            if (TryGetStringConversion(type, out conv))
                return conv;
            throw Contracts.Except($"No conversion from '{type}' to {nameof(StringBuilder)}");
        }

        public ValueMapper<TSrc, SB> GetStringConversion<TSrc>()
        {
            ValueMapper<TSrc, SB> conv;
            if (TryGetStringConversion(out conv))
                return conv;
            throw Contracts.Except($"No conversion from '{typeof(TSrc)}' to {nameof(StringBuilder)}");
        }

        public bool TryGetStringConversion<TSrc>(ColumnType type, out ValueMapper<TSrc, SB> conv)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.Check(type.RawType == typeof(TSrc), "Wrong TSrc type argument");

            if (type.IsKey)
            {
                // Key string conversion always works.
                conv = GetKeyStringConversion<TSrc>(type.AsKey);
                return true;
            }
            return TryGetStringConversion(out conv);
        }

        private bool TryGetStringConversion<TSrc>(out ValueMapper<TSrc, SB> conv)
        {
            DataKind kindSrc;
            if (!_kinds.TryGetValue(typeof (TSrc), out kindSrc))
            {
                conv = null;
                return false;
            }
            int key = GetKey(kindSrc, _kindStringBuilder);
            Delegate del;
            if (_delegatesAll.TryGetValue(key, out del))
            {
                conv = (ValueMapper<TSrc, SB>)del;
                return true;
            }
            conv = null;
            return false;
        }

        public ValueMapper<TSrc, SB> GetKeyStringConversion<TSrc>(KeyType key)
        {
            Contracts.Check(key.RawType == typeof(TSrc));

            // For key types, first convert to ulong, then do the range check,
            // then convert to StringBuilder.
            U8 min = key.Min;
            int count = key.Count;
            Contracts.Assert(count >= 0 && (U8)count <= U8.MaxValue - min);

            bool identity;
            var convSrc = GetStandardConversion<TSrc, U8>(key, NumberType.U8, out identity);
            var convU8 = GetStringConversion<U8>(NumberType.U8);
            if (count > 0)
            {
                return
                    (ref TSrc src, ref SB dst) =>
                    {
                        ulong tmp = 0;
                        convSrc(ref src, ref tmp);
                        if (tmp == 0 || tmp > (ulong)count)
                            ClearDst(ref dst);
                        else
                        {
                            tmp = tmp + min - 1;
                            convU8(ref tmp, ref dst);
                        }
                    };
            }
            else
            {
                return
                    (ref TSrc src, ref SB dst) =>
                    {
                        U8 tmp = 0;
                        convSrc(ref src, ref tmp);
                        if (tmp == 0 || min > 1 && tmp > U8.MaxValue - min + 1)
                            ClearDst(ref dst);
                        else
                        {
                            tmp = tmp + min - 1;
                            convU8(ref tmp, ref dst);
                        }
                    };
            }
        }

        public TryParseMapper<TDst> GetParseConversion<TDst>(ColumnType typeDst)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.IsStandardScalar || typeDst.IsKey, nameof(typeDst),
                "Parse conversion only supported for standard types");
            Contracts.Check(typeDst.RawType == typeof(TDst), "Wrong TDst type parameter");

            if (typeDst.IsKey)
                return GetKeyTryParse<TDst>(typeDst.AsKey);

            Contracts.Assert(_tryParseDelegates.ContainsKey(typeDst.RawKind));
            return (TryParseMapper<TDst>)_tryParseDelegates[typeDst.RawKind];
        }

        private TryParseMapper<TDst> GetKeyTryParse<TDst>(KeyType key)
        {
            Contracts.Assert(key.RawType == typeof(TDst));

            // First parse as ulong, then convert to T.
            ulong min = key.Min;
            ulong max;

            ulong count = DataKindExtensions.ToMaxInt(key.RawKind);
            if (key.Count > 0)
                max = min - 1 + (ulong)key.Count;
            else if (min == 0)
                max = count - 1;
            else if (key.RawKind == DataKind.U8)
                max = ulong.MaxValue;
            else if (min - 1 > ulong.MaxValue - count)
                max = ulong.MaxValue;
            else
                max = min - 1 + count;

            bool identity;
            var fnConv = GetStandardConversion<U8, TDst>(NumberType.U8, NumberType.FromKind(key.RawKind), out identity);
            return
                (ref TX src, out TDst dst) =>
                {
                    ulong uu;
                    dst = default(TDst);
                    if (!TryParseKey(ref src, min, max, out uu))
                        return false;
                    // REVIEW: This call to fnConv should never need range checks, so could be made faster.
                    // Also, it would be nice to be able to assert that it doesn't overflow....
                    fnConv(ref uu, ref dst);
                    return true;
                };
        }

        private Delegate GetKeyParse(KeyType key)
        {
            Func<KeyType, ValueMapper<TX, int>> del = GetKeyParse<int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(key.RawType);
            return (Delegate)meth.Invoke(this, new object[] { key });
        }

        private ValueMapper<TX, TDst> GetKeyParse<TDst>(KeyType key)
        {
            Contracts.Assert(key.RawType == typeof(TDst));

            // First parse as ulong, then convert to T.
            ulong min = key.Min;
            ulong max;

            ulong count = DataKindExtensions.ToMaxInt(key.RawKind);
            if (key.Count > 0)
                max = min - 1 + (ulong)key.Count;
            else if (min == 0)
                max = count - 1;
            else if (key.RawKind == DataKind.U8)
                max = ulong.MaxValue;
            else if (min - 1 > ulong.MaxValue - count)
                max = ulong.MaxValue;
            else
                max = min - 1 + count;

            bool identity;
            var fnConv = GetStandardConversion<U8, TDst>(NumberType.U8, NumberType.FromKind(key.RawKind), out identity);
            return
                (ref TX src, ref TDst dst) =>
                {
                    ulong uu;
                    dst = default(TDst);
                    if (!TryParseKey(ref src, min, max, out uu))
                    {
                        dst = default(TDst);
                        return;
                    }
                    // REVIEW: This call to fnConv should never need range checks, so could be made faster.
                    // Also, it would be nice to be able to assert that it doesn't overflow....
                    fnConv(ref uu, ref dst);
                };
        }

        private static StringBuilder ClearDst(ref StringBuilder dst)
        {
            if (dst == null)
                dst = new StringBuilder();
            else
                dst.Clear();
            return dst;
        }

        public RefPredicate<T> GetIsDefaultPredicate<T>(ColumnType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(!type.IsVector, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            var t = type;
            Delegate del;
            if (!t.IsStandardScalar && !t.IsKey || !_isDefaultDelegates.TryGetValue(t.RawKind, out del))
                throw Contracts.Except("No IsDefault predicate for '{0}'", type);

            return (RefPredicate<T>)del;
        }

        public RefPredicate<T> GetIsNAPredicate<T>(ColumnType type)
        {
            RefPredicate<T> pred;
            if (TryGetIsNAPredicate(type, out pred))
                return pred;
            throw Contracts.Except("No IsNA predicate for '{0}'", type);
        }

        public bool TryGetIsNAPredicate<T>(ColumnType type, out RefPredicate<T> pred)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!TryGetIsNAPredicate(type, out del))
            {
                pred = null;
                return false;
            }

            Contracts.Assert(del is RefPredicate<T>);
            pred = (RefPredicate<T>)del;
            return true;
        }

        public bool TryGetIsNAPredicate(ColumnType type, out Delegate del)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(!type.IsVector, nameof(type));

            var t = type;
            if (t.IsKey)
            {
                // REVIEW: Should we test for out of range when KeyCount > 0?
                Contracts.Assert(_isDefaultDelegates.ContainsKey(t.RawKind));
                del = _isDefaultDelegates[t.RawKind];
            }
            else if (!t.IsStandardScalar || !_isNADelegates.TryGetValue(t.RawKind, out del))
            {
                del = null;
                return false;
            }

            Contracts.Assert(del != null);
            return true;
        }

        public RefPredicate<VBuffer<T>> GetHasMissingPredicate<T>(VectorType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.ItemType.RawType == typeof(T), nameof(type));

            var t = type.ItemType;
            Delegate del;
            if (t.IsKey)
            {
                // REVIEW: Should we test for out of range when KeyCount > 0?
                Contracts.Assert(_hasZeroDelegates.ContainsKey(t.RawKind));
                del = _hasZeroDelegates[t.RawKind];
            }
            else if (!t.IsStandardScalar || !_hasNADelegates.TryGetValue(t.RawKind, out del))
                throw Contracts.Except("No HasMissing predicate for '{0}'", type);

            return (RefPredicate<VBuffer<T>>)del;
        }

        /// <summary>
        /// Returns the NA value of the given type, if it has one, otherwise, it returns
        /// default of the type. This only knows about NA values of standard scalar types
        /// and key types.
        /// </summary>
        public T GetNAOrDefault<T>(ColumnType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawKind, out del))
                return default(T);
            T res = default(T);
            ((ValueGetter<T>)del)(ref res);
            return res;
        }

        /// <summary>
        /// Returns the NA value of the given type, if it has one, otherwise, it returns
        /// default of the type. This only knows about NA values of standard scalar types
        /// and key types. Returns whether the returned value is the default value or not.
        /// </summary>
        public T GetNAOrDefault<T>(ColumnType type, out bool isDefault)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawKind, out del))
            {
                isDefault = true;
                return default(T);
            }

            T res = default(T);
            ((ValueGetter<T>)del)(ref res);
            isDefault = false;

#if DEBUG
            Delegate isDefPred;
            if (_isDefaultDelegates.TryGetValue(type.RawKind, out isDefPred))
                Contracts.Assert(!((RefPredicate<T>)isDefPred)(ref res));
#endif

            return res;
        }

        /// <summary>
        /// Returns a ValueGetter{T} that produces the NA value of the given type, if it has one,
        /// otherwise, it produces default of the type. This only knows about NA values of standard
        /// scalar types and key types.
        /// </summary>
        public ValueGetter<T> GetNAOrDefaultGetter<T>(ColumnType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawKind, out del))
                return (ref T res) => res = default(T);
            return (ValueGetter<T>)del;
        }

        // The IsNA methods are for efficient delegates (instance instead of static).
        #region IsNA
        private bool IsNA(ref I1 src) => src.IsNA;
        private bool IsNA(ref I2 src) => src.IsNA;
        private bool IsNA(ref I4 src) => src.IsNA;
        private bool IsNA(ref I8 src) => src.IsNA;
        private bool IsNA(ref R4 src) => src.IsNA();
        private bool IsNA(ref R8 src) => src.IsNA();
        private bool IsNA(ref TS src) => src.IsNA;
        private bool IsNA(ref DT src) => src.IsNA;
        private bool IsNA(ref DZ src) => src.IsNA;
        private bool IsNA(ref TX src) => src.IsNA;
        #endregion IsNA

        #region HasNA
        private bool HasNA(ref VBuffer<I1> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<I2> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<I4> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<I8> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<R4> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA()) return true; } return false; }
        private bool HasNA(ref VBuffer<R8> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA()) return true; } return false; }
        private bool HasNA(ref VBuffer<TS> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<DT> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<DZ> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        private bool HasNA(ref VBuffer<TX> src) { for (int i = 0; i < src.Count; i++) { if (src.Values[i].IsNA) return true; } return false; }
        #endregion HasNA

        #region IsDefault
        private bool IsDefault(ref I1 src) => src.RawValue == 0;
        private bool IsDefault(ref I2 src) => src.RawValue == 0;
        private bool IsDefault(ref I4 src) => src.RawValue == 0;
        private bool IsDefault(ref I8 src) => src.RawValue == 0;
        private bool IsDefault(ref R4 src) => src == 0;
        private bool IsDefault(ref R8 src) => src == 0;
        private bool IsDefault(ref TX src) => src.IsEmpty;
        private bool IsDefault(ref BL src) => !src;
        private bool IsDefault(ref U1 src) => src == 0;
        private bool IsDefault(ref U2 src) => src == 0;
        private bool IsDefault(ref U4 src) => src == 0;
        private bool IsDefault(ref U8 src) => src == 0;
        private bool IsDefault(ref UG src) => src.Equals(default(UG));
        private bool IsDefault(ref TS src) => src.Equals(default(TS));
        private bool IsDefault(ref DT src) => src.Equals(default(DT));
        private bool IsDefault(ref DZ src) => src.Equals(default(DZ));
        #endregion IsDefault

        #region HasZero
        private bool HasZero(ref VBuffer<U1> src) { if (!src.IsDense) return true; for (int i = 0; i < src.Count; i++) { if (src.Values[i] == 0) return true; } return false; }
        private bool HasZero(ref VBuffer<U2> src) { if (!src.IsDense) return true; for (int i = 0; i < src.Count; i++) { if (src.Values[i] == 0) return true; } return false; }
        private bool HasZero(ref VBuffer<U4> src) { if (!src.IsDense) return true; for (int i = 0; i < src.Count; i++) { if (src.Values[i] == 0) return true; } return false; }
        private bool HasZero(ref VBuffer<U8> src) { if (!src.IsDense) return true; for (int i = 0; i < src.Count; i++) { if (src.Values[i] == 0) return true; } return false; }
        #endregion HasZero

        #region GetNA
        private void GetNA(ref I1 value) => value = I1.NA;
        private void GetNA(ref I2 value) => value = I2.NA;
        private void GetNA(ref I4 value) => value = I4.NA;
        private void GetNA(ref I8 value) => value = I8.NA;
        private void GetNA(ref R4 value) => value = R4.NaN;
        private void GetNA(ref R8 value) => value = R8.NaN;
        private void GetNA(ref TS value) => value = TS.NA;
        private void GetNA(ref DT value) => value = DT.NA;
        private void GetNA(ref DZ value) => value = DZ.NA;
        private void GetNA(ref TX value) => value = TX.NA;
        #endregion GetNA

        #region ToI1
        public void Convert(ref I1 src, ref I1 dst) => dst = src;
        public void Convert(ref I2 src, ref I1 dst) => dst = (I1)src;
        public void Convert(ref I4 src, ref I1 dst) => dst = (I1)src;
        public void Convert(ref I8 src, ref I1 dst) => dst = (I1)src;
        #endregion ToI1

        #region ToI2
        public void Convert(ref I1 src, ref I2 dst) => dst = src;
        public void Convert(ref I2 src, ref I2 dst) => dst = src;
        public void Convert(ref I4 src, ref I2 dst) => dst = (I2)src;
        public void Convert(ref I8 src, ref I2 dst) => dst = (I2)src;
        #endregion ToI2

        #region ToI4
        public void Convert(ref I1 src, ref I4 dst) => dst = src;
        public void Convert(ref I2 src, ref I4 dst) => dst = src;
        public void Convert(ref I4 src, ref I4 dst) => dst = src;
        public void Convert(ref I8 src, ref I4 dst) => dst = (I4)src;
        #endregion ToI4

        #region ToI8
        public void Convert(ref I1 src, ref I8 dst) => dst = src;
        public void Convert(ref I2 src, ref I8 dst) => dst = src;
        public void Convert(ref I4 src, ref I8 dst) => dst = src;
        public void Convert(ref I8 src, ref I8 dst) => dst = src;

        public void Convert(ref TS src, ref I8 dst) => dst = (I8)src.Ticks;
        public void Convert(ref DT src, ref I8 dst) => dst = (I8)src.Ticks;
        public void Convert(ref DZ src, ref I8 dst) => dst = (I8)src.UtcDateTime.Ticks;
        #endregion ToI8

        #region ToU1
        public void Convert(ref U1 src, ref U1 dst) => dst = src;
        public void Convert(ref U2 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(ref U4 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(ref U8 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(ref UG src, ref U1 dst) => dst = src.Hi == 0 && src.Lo <= U1.MaxValue ? (U1)src.Lo : (U1)0;
        #endregion ToU1

        #region ToU2
        public void Convert(ref U1 src, ref U2 dst) => dst = src;
        public void Convert(ref U2 src, ref U2 dst) => dst = src;
        public void Convert(ref U4 src, ref U2 dst) => dst = src <= U2.MaxValue ? (U2)src : (U2)0;
        public void Convert(ref U8 src, ref U2 dst) => dst = src <= U2.MaxValue ? (U2)src : (U2)0;
        public void Convert(ref UG src, ref U2 dst) => dst = src.Hi == 0 && src.Lo <= U2.MaxValue ? (U2)src.Lo : (U2)0;
        #endregion ToU2

        #region ToU4
        public void Convert(ref U1 src, ref U4 dst) => dst = src;
        public void Convert(ref U2 src, ref U4 dst) => dst = src;
        public void Convert(ref U4 src, ref U4 dst) => dst = src;
        public void Convert(ref U8 src, ref U4 dst) => dst = src <= U4.MaxValue ? (U4)src : (U4)0;
        public void Convert(ref UG src, ref U4 dst) => dst = src.Hi == 0 && src.Lo <= U4.MaxValue ? (U4)src.Lo : (U4)0;
        #endregion ToU4

        #region ToU8
        public void Convert(ref U1 src, ref U8 dst) => dst = src;
        public void Convert(ref U2 src, ref U8 dst) => dst = src;
        public void Convert(ref U4 src, ref U8 dst) => dst = src;
        public void Convert(ref U8 src, ref U8 dst) => dst = src;
        public void Convert(ref UG src, ref U8 dst) => dst = src.Hi == 0 ? src.Lo : (U8)0;
        #endregion ToU8

        #region ToUG
        public void Convert(ref U1 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(ref U2 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(ref U4 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(ref U8 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(ref UG src, ref UG dst) => dst = src;
        #endregion ToUG

        #region ToR4
        public void Convert(ref I1 src, ref R4 dst) => dst = (R4)src;
        public void Convert(ref I2 src, ref R4 dst) => dst = (R4)src;
        public void Convert(ref I4 src, ref R4 dst) => dst = (R4)src;
        public void Convert(ref I8 src, ref R4 dst) => dst = (R4)src;
        public void Convert(ref U1 src, ref R4 dst) => dst = src;
        public void Convert(ref U2 src, ref R4 dst) => dst = src;
        public void Convert(ref U4 src, ref R4 dst) => dst = src;
        // REVIEW: The 64-bit JIT has a bug in that it rounds incorrectly from ulong
        // to floating point when the high bit of the ulong is set. Should we work around the bug
        // or just live with it? See the DoubleParser code for details.
        public void Convert(ref U8 src, ref R4 dst) => dst = src;

        public void Convert(ref TS src, ref R4 dst) => dst = (R4)src.Ticks;
        public void Convert(ref DT src, ref R4 dst) => dst = (R4)src.Ticks;
        public void Convert(ref DZ src, ref R4 dst) => dst = (R4)src.UtcDateTime.Ticks;
        #endregion ToR4

        #region ToR8
        public void Convert(ref I1 src, ref R8 dst) => dst = (R8)src;
        public void Convert(ref I2 src, ref R8 dst) => dst = (R8)src;
        public void Convert(ref I4 src, ref R8 dst) => dst = (R8)src;
        public void Convert(ref I8 src, ref R8 dst) => dst = (R8)src;
        public void Convert(ref U1 src, ref R8 dst) => dst = src;
        public void Convert(ref U2 src, ref R8 dst) => dst = src;
        public void Convert(ref U4 src, ref R8 dst) => dst = src;
        // REVIEW: The 64-bit JIT has a bug in that it rounds incorrectly from ulong
        // to floating point when the high bit of the ulong is set. Should we work around the bug
        // or just live with it? See the DoubleParser code for details.
        public void Convert(ref U8 src, ref R8 dst) => dst = src;

        public void Convert(ref TS src, ref R8 dst) => dst = (R8)src.Ticks;
        public void Convert(ref DT src, ref R8 dst) => dst = (R8)src.Ticks;
        public void Convert(ref DZ src, ref R8 dst) => dst = (R8)src.UtcDateTime.Ticks;
        #endregion ToR8

        #region ToStringBuilder
        public void Convert(ref I1 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.Append(src.RawValue); }
        public void Convert(ref I2 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.Append(src.RawValue); }
        public void Convert(ref I4 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.Append(src.RawValue); }
        public void Convert(ref I8 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.Append(src.RawValue); }
        public void Convert(ref U1 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(ref U2 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(ref U4 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(ref U8 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(ref UG src, ref SB dst) { ClearDst(ref dst); dst.AppendFormat("0x{0:x16}{1:x16}", src.Hi, src.Lo); }
        public void Convert(ref R4 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA()) dst.AppendFormat(CultureInfo.InvariantCulture, "{0:R}", src); }
        public void Convert(ref R8 src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA()) dst.AppendFormat(CultureInfo.InvariantCulture, "{0:G17}", src); }
        public void Convert(ref BL src, ref SB dst)
        {
            ClearDst(ref dst);
            if (!src)
                dst.Append("0");
            else
                dst.Append("1");
        }
        public void Convert(ref TS src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.AppendFormat("{0:c}", (TimeSpan)src); }
        public void Convert(ref DT src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.AppendFormat("{0:o}", (DateTime)src); }
        public void Convert(ref DZ src, ref SB dst) { ClearDst(ref dst); if (!src.IsNA) dst.AppendFormat("{0:o}", (DateTimeOffset)src); }
        #endregion ToStringBuilder

        #region FromR4
        public void Convert(ref R4 src, ref R4 dst) => dst = src;
        public void Convert(ref R4 src, ref R8 dst) => dst = src;
        #endregion FromR4

        #region FromR8
        public void Convert(ref R8 src, ref R4 dst) => dst = (R4)src;
        public void Convert(ref R8 src, ref R8 dst) => dst = src;
        #endregion FromR8

        #region FromTX

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// </summary>
        public bool TryParse(ref TX src, out U1 dst)
        {
            ulong res;
            if (!TryParse(ref src, out res) || res > U1.MaxValue)
            {
                dst = 0;
                return false;
            }
            dst = (U1)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// </summary>
        public bool TryParse(ref TX src, out U2 dst)
        {
            ulong res;
            if (!TryParse(ref src, out res) || res > U2.MaxValue)
            {
                dst = 0;
                return false;
            }
            dst = (U2)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// </summary>
        public bool TryParse(ref TX src, out U4 dst)
        {
            ulong res;
            if (!TryParse(ref src, out res) || res > U4.MaxValue)
            {
                dst = 0;
                return false;
            }
            dst = (U4)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// </summary>
        public bool TryParse(ref TX src, out U8 dst)
        {
            if (src.IsNA)
            {
                dst = 0;
                return false;
            }

            int ichMin;
            int ichLim;
            string text = src.GetRawUnderlyingBufferInfo(out ichMin, out ichLim);
            return TryParseCore(text, ichMin, ichLim, out dst);
        }

        /// <summary>
        /// A parse method that transforms a 34-length string into a <see cref="UInt128"/>.
        /// </summary>
        /// <param name="src">What should be a 34-length hexadecimal representation, including a 0x prefix,
        /// of the 128-bit number</param>
        /// <param name="dst">The result</param>
        /// <returns>Whether the input string was parsed successfully, that is, it was exactly length 32
        /// and had only digits and the letters 'a' through 'f' or 'A' through 'F' as characters</returns>
        public bool TryParse(ref TX src, out UG dst)
        {
            // REVIEW: Accomodate numeric inputs?
            if (src.Length != 34 || src[0] != '0' || (src[1] != 'x' && src[1] != 'X'))
            {
                dst = default(UG);
                return false;
            }
            int ichMin;
            int ichLim;
            string tx = src.GetRawUnderlyingBufferInfo(out ichMin, out ichLim);
            int offset = ichMin + 2;
            ulong hi = 0;
            ulong num = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int d = 0; d < 16; ++d)
                {
                    num <<= 4;
                    char c = tx[offset++];
                    // REVIEW: An exhaustive switch statement *might* be faster, maybe, at the
                    // cost of being significantly longer.
                    if ('0' <= c && c <= '9')
                        num |= (uint)(c - '0');
                    else if ('A' <= c && c <= 'F')
                        num |= (uint)(c - 'A' + 10);
                    else if ('a' <= c && c <= 'f')
                        num |= (uint)(c - 'a' + 10);
                    else
                    {
                        dst = default(UG);
                        return false;
                    }
                }
                if (i == 0)
                {
                    hi = num;
                    num = 0;
                }
            }
            Contracts.Assert(offset == ichLim);
            // The first read bits are the higher order bits, so they are listed second here.
            dst = new UG(num, hi);
            return true;
        }

        /// <summary>
        /// Return true if the span contains a standard text representation of NA
        /// other than the standard TX missing representation - callers should
        /// have already dealt with that case and the case of empty.
        /// The standard representations are any casing of:
        ///    ?  NaN  NA  N/A
        /// </summary>
        private bool IsStdMissing(ref TX src)
        {
            Contracts.Assert(src.HasChars);

            char ch;
            switch (src.Length)
            {
            default:
                return false;

            case 1:
                if (src[0] == '?')
                    return true;
                return false;
            case 2:
                if ((ch = src[0]) != 'N' && ch != 'n')
                    return false;
                if ((ch = src[1]) != 'A' && ch != 'a')
                    return false;
                return true;
            case 3:
                if ((ch = src[0]) != 'N' && ch != 'n')
                    return false;
                if ((ch = src[1]) == '/')
                {
                    // Check for N/A.
                    if ((ch = src[2]) != 'A' && ch != 'a')
                        return false;
                }
                else
                {
                    // Check for NaN.
                    if (ch != 'a' && ch != 'A')
                        return false;
                    if ((ch = src[2]) != 'N' && ch != 'n')
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Utility to assist in parsing key-type values. The min and max values define
        /// the legal input value bounds. The output dst value is "normalized" so min is
        /// mapped to 1, max is mapped to 1 + (max - min).
        /// Missing values are mapped to zero with a true return.
        /// Unparsable or out of range values are mapped to zero with a false return.
        /// </summary>
        public bool TryParseKey(ref TX src, U8 min, U8 max, out U8 dst)
        {
            Contracts.Assert(min <= max);

            // This simply ensures we don't have min == 0 and max == U8.MaxValue. This is illegal since
            // we map min to 1, which would cause max to overflow to zero. Specifically, it protects
            // against overflow in the expression uu - min + 1 below.
            Contracts.Assert((max - min) < U8.MaxValue);

            // Both empty and missing map to zero (NA for key values) and that mapping is valid,
            // hence the true return.
            if (!src.HasChars)
            {
                dst = 0;
                return true;
            }

            // Parse a ulong.
            int ichMin;
            int ichLim;
            string text = src.GetRawUnderlyingBufferInfo(out ichMin, out ichLim);
            ulong uu;
            if (!TryParseCore(text, ichMin, ichLim, out uu))
            {
                dst = 0;
                // Return true only for standard forms for NA.
                return IsStdMissing(ref src);
            }

            if (min > uu || uu > max)
            {
                dst = 0;
                return false;
            }

            dst = uu - min + 1;
            return true;
        }

        private bool TryParseCore(string text, int ich, int lim, out ulong dst)
        {
            Contracts.Assert(0 <= ich && ich <= lim && lim <= Utils.Size(text));

            ulong res = 0;
            while (ich < lim)
            {
                uint d = (uint)text[ich++] - (uint)'0';
                if (d >= 10)
                    goto LFail;

                // If any of the top three bits of prev are set, we're guaranteed to overflow.
                if ((res & 0xE000000000000000UL) != 0)
                    goto LFail;

                // Given that tmp = 8 * res doesn't overflow, if 10 * res + d overflows, then it overflows to
                // 10 * res + d - 2^n = tmp + (2 * res + d - 2^n). Clearly the paren group is negative,
                // so the new result (after overflow) will be less than tmp. The converse is also true.
                ulong tmp = res << 3;
                res = tmp + (res << 1) + d;
                if (res < tmp)
                    goto LFail;
            }
            dst = res;
            return true;

            LFail:
            dst = 0;
            return false;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out I1 dst)
        {
            long res;
            bool f = TryParseSigned(RawI1.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I1.RawNA);
            Contracts.Assert((RawI1)res == res);
            dst = (RawI1)res;
            return f;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out I2 dst)
        {
            long res;
            bool f = TryParseSigned(RawI2.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I2.RawNA);
            Contracts.Assert((RawI2)res == res);
            dst = (RawI2)res;
            return f;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out I4 dst)
        {
            long res;
            bool f = TryParseSigned(RawI4.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I4.RawNA);
            Contracts.Assert((RawI4)res == res);
            dst = (RawI4)res;
            return f;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out I8 dst)
        {
            long res;
            bool f = TryParseSigned(RawI8.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I8.RawNA);
            dst = res;
            return f;
        }

        /// <summary>
        /// Returns false if the text is not parsable as an non-negative long or overflows.
        /// </summary>
        private bool TryParseNonNegative(string text, int ich, int lim, out long result)
        {
            Contracts.Assert(0 <= ich && ich <= lim && lim <= Utils.Size(text));

            long res = 0;
            while (ich < lim)
            {
                Contracts.Assert(res >= 0);
                uint d = (uint)text[ich++] - (uint)'0';
                if (d >= 10)
                    goto LFail;

                // If any of the top four bits of prev are set, we're guaranteed to overflow.
                if (res >= 0x1000000000000000L)
                    goto LFail;

                // Given that tmp = 8 * res doesn't overflow, if 10 * res + d overflows, then it overflows to
                // a negative value. The converse is also true.
                res = (res << 3) + (res << 1) + d;
                if (res < 0)
                    goto LFail;
            }
            result = res;
            return true;

            LFail:
            result = 0;
            return false;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable as a signed integer
        /// or the result overflows. The min legal value is -max. The NA value is -max - 1.
        /// When it returns false, result is set to the NA value. The result can be NA on true return,
        /// since some representations of NA are not considered parse failure.
        /// </summary>
        private bool TryParseSigned(long max, ref TX span, out long result)
        {
            Contracts.Assert(max > 0);
            Contracts.Assert((max & (max + 1)) == 0);

            if (!span.HasChars)
            {
                if (span.IsNA)
                    result = -max - 1;
                else
                    result = 0;
                return true;
            }

            int ichMin;
            int ichLim;
            string text = span.GetRawUnderlyingBufferInfo(out ichMin, out ichLim);

            long val;
            if (span[0] == '-')
            {
                if (span.Length == 1 ||
                    !TryParseNonNegative(text, ichMin + 1, ichLim, out val) ||
                    val > max)
                {
                    result = -max - 1;
                    return false;
                }
                Contracts.Assert(val >= 0);
                result = -(long)val;
                Contracts.Assert(long.MinValue < result && result <= 0);
                return true;
            }

            if (!TryParseNonNegative(text, ichMin, ichLim, out val))
            {
                // Check for acceptable NA forms: ? NaN NA and N/A.
                result = -max - 1;
                return IsStdMissing(ref span);
            }

            Contracts.Assert(val >= 0);
            if (val > max)
            {
                result = -max - 1;
                return false;
            }

            result = (long)val;
            Contracts.Assert(0 <= result && result <= long.MaxValue);
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out R4 dst)
        {
            if (src.TryParse(out dst))
                return true;
            dst = R4.NaN;
            return IsStdMissing(ref src);
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(ref TX src, out R8 dst)
        {
            if (src.TryParse(out dst))
                return true;
            dst = R8.NaN;
            return IsStdMissing(ref src);
        }

        public bool TryParse(ref TX src, out TS dst)
        {
            if (!src.HasChars)
            {
                if (src.IsNA)
                    dst = TS.NA;
                else
                    dst = default(TS);
                return true;
            }
            TimeSpan res;
            if (TimeSpan.TryParse(src.ToString(), CultureInfo.InvariantCulture, out res))
            {
                dst = new TS(res);
                return true;
            }
            dst = TS.NA;
            return IsStdMissing(ref src);
        }

        public bool TryParse(ref TX src, out DT dst)
        {
            if (!src.HasChars)
            {
                if (src.IsNA)
                    dst = DvDateTime.NA;
                else
                    dst = default(DvDateTime);
                return true;
            }
            DateTime res;
            if (DateTime.TryParse(src.ToString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out res))
            {
                dst = new DT(res);
                return true;
            }
            dst = DvDateTime.NA;
            return IsStdMissing(ref src);
        }

        public bool TryParse(ref TX src, out DZ dst)
        {
            if (!src.HasChars)
            {
                if (src.IsNA)
                    dst = DvDateTimeZone.NA;
                else
                    dst = default(DvDateTimeZone);
                return true;
            }
            DateTimeOffset res;
            if (DateTimeOffset.TryParse(src.ToString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out res))
            {
                dst = new DZ(res);
                return true;
            }
            dst = DvDateTimeZone.NA;
            return IsStdMissing(ref src);
        }

        // These map unparsable and overflow values to "NA", which is the value Ix.MinValue. Note that this NA
        // value is the "evil" value - the non-zero value, x, such that x == -x. Note also, that for I4, this
        // matches R's representation of NA.
        private I1 ParseI1(ref TX src)
        {
            long res;
            bool f = TryParseSigned(RawI1.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I1.RawNA);
            Contracts.Assert((RawI1)res == res);
            return (RawI1)res;
        }

        private I2 ParseI2(ref TX src)
        {
            long res;
            bool f = TryParseSigned(RawI2.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I2.RawNA);
            Contracts.Assert((RawI2)res == res);
            return (RawI2)res;
        }

        private I4 ParseI4(ref TX src)
        {
            long res;
            bool f = TryParseSigned(RawI4.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I4.RawNA);
            Contracts.Assert((RawI4)res == res);
            return (RawI4)res;
        }

        private I8 ParseI8(ref TX src)
        {
            long res;
            bool f = TryParseSigned(RawI8.MaxValue, ref src, out res);
            Contracts.Assert(f || res == I8.RawNA);
            return res;
        }

        // These map unparsable and overflow values to zero. The unsigned integer types do not have an NA value.
        // Note that this matches the "bad" value for key-types, which will likely be the primary use for
        // unsigned integer types.
        private U1 ParseU1(ref TX span)
        {
            ulong res;
            if (!TryParse(ref span, out res))
                return 0;
            if (res > U1.MaxValue)
                return 0;
            return (U1)res;
        }

        private U2 ParseU2(ref TX span)
        {
            ulong res;
            if (!TryParse(ref span, out res))
                return 0;
            if (res > U2.MaxValue)
                return 0;
            return (U2)res;
        }

        private U4 ParseU4(ref TX span)
        {
            ulong res;
            if (!TryParse(ref span, out res))
                return 0;
            if (res > U4.MaxValue)
                return 0;
            return (U4)res;
        }

        private U8 ParseU8(ref TX span)
        {
            ulong res;
            if (!TryParse(ref span, out res))
                return 0;
            return res;
        }

        /// <summary>
        /// Try parsing a TX to a BL. This returns false for NA (span.IsMissing).
        /// Otherwise, it trims the span, then succeeds on all casings of the strings:
        /// * false, f, no, n, 0, -1, - => false
        /// * true, t, yes, y, 1, +1, + => true
        /// Empty string (but not missing string) succeeds and maps to false.
        /// </summary>
        public bool TryParse(ref TX src, out BL dst)
        {
            // NA text fails.
            Contracts.Check(!src.IsNA, "Missing text value cannot be converted to boolean value.");

            char ch;
            switch (src.Length)
            {
            case 0:
                // Empty succeeds and maps to false.
                dst = false;
                return true;

            case 1:
                switch (src[0])
                {
                case 'T':
                case 't':
                case 'Y':
                case 'y':
                case '1':
                case '+':
                    dst = true;
                    return true;
                case 'F':
                case 'f':
                case 'N':
                case 'n':
                case '0':
                case '-':
                    dst = false;
                    return true;
                }
                break;

            case 2:
                switch (src[0])
                {
                case 'N':
                case 'n':
                    if ((ch = src[1]) != 'O' && ch != 'o')
                        break;
                    dst = false;
                    return true;
                case '+':
                    if ((ch = src[1]) != '1')
                        break;
                    dst = true;
                    return true;
                case '-':
                    if ((ch = src[1]) != '1')
                        break;
                    dst = false;
                    return true;
                }
                break;

            case 3:
                switch (src[0])
                {
                case 'Y':
                case 'y':
                    if ((ch = src[1]) != 'E' && ch != 'e')
                        break;
                    if ((ch = src[2]) != 'S' && ch != 's')
                        break;
                    dst = true;
                    return true;
                }
                break;

            case 4:
                switch (src[0])
                {
                case 'T':
                case 't':
                    if ((ch = src[1]) != 'R' && ch != 'r')
                        break;
                    if ((ch = src[2]) != 'U' && ch != 'u')
                        break;
                    if ((ch = src[3]) != 'E' && ch != 'e')
                        break;
                    dst = true;
                    return true;
                }
                break;

            case 5:
                switch (src[0])
                {
                case 'F':
                case 'f':
                    if ((ch = src[1]) != 'A' && ch != 'a')
                        break;
                    if ((ch = src[2]) != 'L' && ch != 'l')
                        break;
                    if ((ch = src[3]) != 'S' && ch != 's')
                        break;
                    if ((ch = src[4]) != 'E' && ch != 'e')
                        break;
                    dst = false;
                    return true;
                }
                break;
            }

            dst = false;
            Contracts.Check(!IsStdMissing(ref src), "Missing text value cannot be converted to boolean value.");
            return true;
        }

        private bool TryParse(ref TX src, out TX dst)
        {
            dst = src;
            return true;
        }

        public void Convert(ref TX span, ref I1 value)
        {
            value = ParseI1(ref span);
        }
        public void Convert(ref TX span, ref U1 value)
        {
            value = ParseU1(ref span);
        }
        public void Convert(ref TX span, ref I2 value)
        {
            value = ParseI2(ref span);
        }
        public void Convert(ref TX span, ref U2 value)
        {
            value = ParseU2(ref span);
        }
        public void Convert(ref TX span, ref I4 value)
        {
            value = ParseI4(ref span);
        }
        public void Convert(ref TX span, ref U4 value)
        {
            value = ParseU4(ref span);
        }
        public void Convert(ref TX span, ref I8 value)
        {
            value = ParseI8(ref span);
        }
        public void Convert(ref TX span, ref U8 value)
        {
            value = ParseU8(ref span);
        }
        public void Convert(ref TX span, ref UG value)
        {
            if (!TryParse(ref span, out value))
                Contracts.Assert(value.Equals(default(UG)));
        }
        public void Convert(ref TX span, ref R4 value)
        {
            if (span.TryParse(out value))
                return;
            // Unparsable is mapped to NA.
            value = R4.NaN;
        }
        public void Convert(ref TX span, ref R8 value)
        {
            if (span.TryParse(out value))
                return;
            // Unparsable is mapped to NA.
            value = R8.NaN;
        }
        public void Convert(ref TX span, ref TX value)
        {
            value = span;
        }
        public void Convert(ref TX span, ref BL value)
        {
            // When TryParseBL returns false, it should have set value to false.
            if (!TryParse(ref span, out value))
                Contracts.Assert(!value);
        }
        public void Convert(ref TX src, ref SB dst)
        {
            ClearDst(ref dst);
            if (src.HasChars)
                src.AddToStringBuilder(dst);
        }

        public void Convert(ref TX span, ref TS value)
        {
            if (!TryParse(ref span, out value))
                Contracts.Assert(value.IsNA);
        }
        public void Convert(ref TX span, ref DT value)
        {
            if (!TryParse(ref span, out value))
                Contracts.Assert(value.IsNA);
        }
        public void Convert(ref TX span, ref DZ value)
        {
            if (!TryParse(ref span, out value))
                Contracts.Assert(value.IsNA);
        }
        #endregion FromTX

        #region FromBL
        public void Convert(ref BL src, ref I1 dst) => dst = (I1)src;
        public void Convert(ref BL src, ref I2 dst) => dst = (I2)src;
        public void Convert(ref BL src, ref I4 dst) => dst = (I4)src;
        public void Convert(ref BL src, ref I8 dst) => dst = (I8)src;
        public void Convert(ref BL src, ref R4 dst) => dst = System.Convert.ToSingle(src);
        public void Convert(ref BL src, ref R8 dst) => dst = System.Convert.ToDouble(src);
        public void Convert(ref BL src, ref BL dst) => dst = src;
        #endregion FromBL
    }
}
