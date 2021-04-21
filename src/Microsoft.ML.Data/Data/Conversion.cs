// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.Conversion
{
    using BL = Boolean;
    using DT = DateTime;
    using DZ = DateTimeOffset;
    using I1 = SByte;
    using I2 = Int16;
    using I4 = Int32;
    using I8 = Int64;
    using R4 = Single;
    using R8 = Double;
    using SB = StringBuilder;
    using TS = TimeSpan;
    using TX = ReadOnlyMemory<char>;
    using U1 = Byte;
    using U2 = UInt16;
    using U4 = UInt32;
    using U8 = UInt64;
    using UG = DataViewRowId;

    [BestFriend]
    internal delegate bool TryParseMapper<T>(in TX src, out T dst);

    /// <summary>
    /// This type exists to provide efficient delegates for conversion between standard ColumnTypes,
    /// as discussed in the IDataView Type System Specification. This is a singleton class.
    /// Some conversions are "standard" conversions, conforming to the details in the spec.
    /// Others are auxiliary conversions. The use of auxiliary conversions should be limited to
    /// situations that genuinely require them and have been well designed in the particular context.
    /// For example, this contains non-standard conversions from the standard primitive types to
    /// text (and StringBuilder). These are needed by the standard TextSaver, which handles
    /// differences between sparse and dense inputs in a semantically invariant way.
    /// </summary>
    [BestFriend]
    internal sealed class Conversions
    {
        private static readonly FuncInstanceMethodInfo1<Conversions, KeyDataViewType, Delegate> _getKeyParseMethodInfo
            = FuncInstanceMethodInfo1<Conversions, KeyDataViewType, Delegate>.Create(target => target.GetKeyParse<int>);

        // REVIEW: Reconcile implementations with TypeUtils, and clarify the distinction.

        // Default instance used by most of the codebase
        // Currently, only TextLoader would sometimes not use this instance
        private static volatile Conversions _defaultInstance;
        public static Conversions DefaultInstance
        {
            get
            {
                return _defaultInstance ??
                    Interlocked.CompareExchange(ref _defaultInstance, new Conversions(), null) ??
                    _defaultInstance;
            }
        }

        // Currently only TextLoader could create instances using non-default DoubleParser.OptionFlags
        private readonly DoubleParser.OptionFlags _doubleParserOptionFlags;
        public static Conversions CreateInstanceWithDoubleParserOptions(DoubleParser.OptionFlags doubleParserOptionFlags)
        {
            return new Conversions(doubleParserOptionFlags);
        }

        // Maps from {src,dst} pair of DataKind to ValueMapper. The {src,dst} pair is
        // the two byte values packed into the low two bytes of an int, with src the lsb.
        private readonly Dictionary<(Type src, Type dst), Delegate> _delegatesStd;

        // Maps from {src,dst} pair of DataKind to ValueMapper. The {src,dst} pair is
        // the two byte values packed into the low two bytes of an int, with src the lsb.
        private readonly Dictionary<(Type src, Type dst), Delegate> _delegatesAll;

        // This has RefPredicate<T> delegates for determining whether a value is NA.
        private readonly Dictionary<Type, Delegate> _isNADelegates;

        // This has RefPredicate<VBuffer<T>> delegates for determining whether a buffer contains any NA values.
        private readonly Dictionary<Type, Delegate> _hasNADelegates;

        // This has RefPredicate<T> delegates for determining whether a value is default.
        private readonly Dictionary<Type, Delegate> _isDefaultDelegates;

        // This has RefPredicate<VBuffer<T>> delegates for determining whether a buffer contains any zero values.
        // The supported types are unsigned signed integer values (for determining whether a key type is NA).
        private readonly Dictionary<Type, Delegate> _hasZeroDelegates;

        // This has ValueGetter<T> delegates for producing an NA value of the given type.
        private readonly Dictionary<Type, Delegate> _getNADelegates;

        // This has TryParseMapper<T> delegates for parsing values from text.
        private readonly Dictionary<Type, Delegate> _tryParseDelegates;

        private Conversions(DoubleParser.OptionFlags doubleParserOptionFlags = DoubleParser.OptionFlags.Default)
        {
            _delegatesStd = new Dictionary<(Type src, Type dst), Delegate>();
            _delegatesAll = new Dictionary<(Type src, Type dst), Delegate>();
            _isNADelegates = new Dictionary<Type, Delegate>();
            _hasNADelegates = new Dictionary<Type, Delegate>();
            _isDefaultDelegates = new Dictionary<Type, Delegate>();
            _hasZeroDelegates = new Dictionary<Type, Delegate>();
            _getNADelegates = new Dictionary<Type, Delegate>();
            _tryParseDelegates = new Dictionary<Type, Delegate>();
            _doubleParserOptionFlags = doubleParserOptionFlags;

            // !!! WARNING !!!: Do NOT add any standard conversions without clearing from the IDV Type System
            // design committee. Any changes also require updating the IDV Type System Specification.

            AddStd<I1, I1>(Convert);
            AddStd<I1, I2>(Convert);
            AddStd<I1, I4>(Convert);
            AddStd<I1, I8>(Convert);
            AddStd<I1, R4>(Convert);
            AddStd<I1, R8>(Convert);
            AddAux<I1, SB>(Convert);
            AddStd<I1, BL>(Convert);
            AddStd<I1, TX>(Convert);

            AddStd<I2, I1>(Convert);
            AddStd<I2, I2>(Convert);
            AddStd<I2, I4>(Convert);
            AddStd<I2, I8>(Convert);
            AddStd<I2, R4>(Convert);
            AddStd<I2, R8>(Convert);
            AddAux<I2, SB>(Convert);
            AddStd<I2, BL>(Convert);
            AddStd<I2, TX>(Convert);

            AddStd<I4, I1>(Convert);
            AddStd<I4, I2>(Convert);
            AddStd<I4, I4>(Convert);
            AddStd<I4, I8>(Convert);
            AddStd<I4, R4>(Convert);
            AddStd<I4, R8>(Convert);
            AddAux<I4, SB>(Convert);
            AddStd<I4, BL>(Convert);
            AddStd<I4, TX>(Convert);

            AddStd<I8, I1>(Convert);
            AddStd<I8, I2>(Convert);
            AddStd<I8, I4>(Convert);
            AddStd<I8, I8>(Convert);
            AddStd<I8, R4>(Convert);
            AddStd<I8, R8>(Convert);
            AddAux<I8, SB>(Convert);
            AddStd<I8, BL>(Convert);
            AddStd<I8, TX>(Convert);

            AddStd<U1, U1>(Convert);
            AddStd<U1, U2>(Convert);
            AddStd<U1, U4>(Convert);
            AddStd<U1, U8>(Convert);
            AddStd<U1, UG>(Convert);
            AddStd<U1, R4>(Convert);
            AddStd<U1, R8>(Convert);
            AddAux<U1, SB>(Convert);
            AddStd<U1, BL>(Convert);
            AddStd<U1, TX>(Convert);

            AddStd<U2, U1>(Convert);
            AddStd<U2, U2>(Convert);
            AddStd<U2, U4>(Convert);
            AddStd<U2, U8>(Convert);
            AddStd<U2, UG>(Convert);
            AddStd<U2, R4>(Convert);
            AddStd<U2, R8>(Convert);
            AddAux<U2, SB>(Convert);
            AddStd<U2, BL>(Convert);
            AddStd<U2, TX>(Convert);

            AddStd<U4, U1>(Convert);
            AddStd<U4, U2>(Convert);
            AddStd<U4, U4>(Convert);
            AddStd<U4, U8>(Convert);
            AddStd<U4, UG>(Convert);
            AddStd<U4, R4>(Convert);
            AddStd<U4, R8>(Convert);
            AddAux<U4, SB>(Convert);
            AddStd<U4, BL>(Convert);
            AddStd<U4, TX>(Convert);

            AddStd<U8, U1>(Convert);
            AddStd<U8, U2>(Convert);
            AddStd<U8, U4>(Convert);
            AddStd<U8, U8>(Convert);
            AddStd<U8, UG>(Convert);
            AddStd<U8, R4>(Convert);
            AddStd<U8, R8>(Convert);
            AddAux<U8, SB>(Convert);
            AddStd<U8, BL>(Convert);
            AddStd<U8, TX>(Convert);

            AddStd<UG, U1>(Convert);
            AddStd<UG, U2>(Convert);
            AddStd<UG, U4>(Convert);
            AddStd<UG, U8>(Convert);
            // REVIEW: Conversion from UG to R4/R8, should we?
            AddAux<UG, SB>(Convert);
            AddStd<UG, TX>(Convert);

            AddStd<R4, R4>(Convert);
            AddStd<R4, BL>(Convert);
            AddStd<R4, R8>(Convert);
            AddAux<R4, SB>(Convert);
            AddStd<R4, TX>(Convert);

            AddStd<R8, R4>(Convert);
            AddStd<R8, R8>(Convert);
            AddStd<R8, BL>(Convert);
            AddAux<R8, SB>(Convert);
            AddStd<R8, TX>(Convert);

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
            AddStd<BL, TX>(Convert);

            AddStd<TS, I8>(Convert);
            AddStd<TS, R4>(Convert);
            AddStd<TS, R8>(Convert);
            AddAux<TS, SB>(Convert);
            AddStd<TS, TX>(Convert);

            AddStd<DT, I8>(Convert);
            AddStd<DT, R4>(Convert);
            AddStd<DT, R8>(Convert);
            AddStd<DT, DT>(Convert);
            AddAux<DT, SB>(Convert);
            AddStd<DT, TX>(Convert);

            AddStd<DZ, I8>(Convert);
            AddStd<DZ, R4>(Convert);
            AddStd<DZ, R8>(Convert);
            AddAux<DZ, SB>(Convert);
            AddStd<DZ, TX>(Convert);

            AddIsNA<R4>(IsNA);
            AddIsNA<R8>(IsNA);

            AddGetNA<R4>(GetNA);
            AddGetNA<R8>(GetNA);

            AddHasNA<R4>(HasNA);
            AddHasNA<R8>(HasNA);

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

        // Add a standard conversion to the lookup tables.
        private void AddStd<TSrc, TDst>(ValueMapper<TSrc, TDst> fn)
        {
            var key = (typeof(TSrc), typeof(TDst));
            _delegatesStd.Add(key, fn);
            _delegatesAll.Add(key, fn);
        }

        // Add a non-standard conversion to the lookup table.
        private void AddAux<TSrc, TDst>(ValueMapper<TSrc, TDst> fn)
        {
            var key = (typeof(TSrc), typeof(TDst));
            _delegatesAll.Add(key, fn);
        }

        private void AddIsNA<T>(InPredicate<T> fn)
        {
            _isNADelegates.Add(typeof(T), fn);
        }

        private void AddGetNA<T>(ValueGetter<T> fn)
        {
            _getNADelegates.Add(typeof(T), fn);
        }

        private void AddHasNA<T>(InPredicate<VBuffer<T>> fn)
        {
            _hasNADelegates.Add(typeof(T), fn);
        }

        private void AddIsDef<T>(InPredicate<T> fn)
        {
            _isDefaultDelegates.Add(typeof(T), fn);
        }

        private void AddHasZero<T>(InPredicate<VBuffer<T>> fn)
        {
            _hasZeroDelegates.Add(typeof(T), fn);
        }

        private void AddTryParse<T>(TryParseMapper<T> fn)
        {
            _tryParseDelegates.Add(typeof(T), fn);
        }

        /// <summary>
        /// Return a standard conversion delegate from typeSrc to typeDst. If there is no such standard
        /// conversion, this throws an exception.
        /// </summary>
        public ValueMapper<TSrc, TDst> GetStandardConversion<TSrc, TDst>(DataViewType typeSrc, DataViewType typeDst,
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
        public bool TryGetStandardConversion<TSrc, TDst>(DataViewType typeSrc, DataViewType typeDst,
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
        public Delegate GetStandardConversion(DataViewType typeSrc, DataViewType typeDst)
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
        public bool TryGetStandardConversion(DataViewType typeSrc, DataViewType typeDst,
            out Delegate conv, out bool identity)
        {
            Contracts.CheckValue(typeSrc, nameof(typeSrc));
            Contracts.CheckValue(typeDst, nameof(typeDst));

            conv = null;
            identity = false;
            if (typeSrc is KeyDataViewType keySrc)
            {
                // Key types are only convertible to compatible key types or unsigned integer
                // types that are large enough.
                if (typeDst is KeyDataViewType keyDst)
                {
                    // REVIEW: Should we allow the counts to vary? Allowing the dst to be bigger is trivial.
                    // Smaller dst means mapping values to NA.
                    if (keySrc.Count != keyDst.Count)
                        return false;
                }
                else
                {
                    // Technically there is no standard conversion from a key type to an unsigned integer type,
                    // but it's very convenient for client code, so we allow it here. Note that ConvertTransform
                    // does not allow this.
                    if (!KeyDataViewType.IsValidDataType(typeDst.RawType))
                        return false;
                    if (Marshal.SizeOf(keySrc.RawType) > Marshal.SizeOf(typeDst.RawType))
                    {
                        if (keySrc.Count > typeDst.RawType.ToMaxInt())
                            return false;
                    }
                }

                // REVIEW: Should we look for illegal values and force them to zero? If so, then
                // we'll need to set identity to false.
            }
            else if (typeDst is KeyDataViewType keyDst)
            {
                if (!(typeSrc is TextDataViewType))
                    return false;
                conv = GetKeyParse(keyDst);
                return true;
            }
            else if (!typeDst.IsStandardScalar())
                return false;

            Contracts.Assert(typeSrc is KeyDataViewType || typeSrc.IsStandardScalar());
            Contracts.Assert(typeDst is KeyDataViewType || typeDst.IsStandardScalar());

            identity = typeSrc.RawType == typeDst.RawType;
            var key = (typeSrc.RawType, typeDst.RawType);
            return _delegatesStd.TryGetValue(key, out conv);
        }

        public ValueMapper<TSrc, SB> GetStringConversion<TSrc>(DataViewType type)
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

        public bool TryGetStringConversion<TSrc>(DataViewType type, out ValueMapper<TSrc, SB> conv)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.Check(type.RawType == typeof(TSrc), "Wrong TSrc type argument");

            if (type is KeyDataViewType keyType)
            {
                // Key string conversion always works.
                conv = GetKeyStringConversion<TSrc>(keyType);
                return true;
            }
            return TryGetStringConversion(out conv);
        }

        private bool TryGetStringConversion<TSrc>(out ValueMapper<TSrc, SB> conv)
        {
            var key = (typeof(TSrc), typeof(SB));
            Delegate del;
            if (_delegatesAll.TryGetValue(key, out del))
            {
                conv = (ValueMapper<TSrc, SB>)del;
                return true;
            }
            conv = null;
            return false;
        }

        public ValueMapper<TSrc, SB> GetKeyStringConversion<TSrc>(KeyDataViewType key)
        {
            Contracts.Check(key.RawType == typeof(TSrc));

            // For key types, first convert to ulong, then do the range check,
            // then convert to StringBuilder.
            ulong count = key.Count;
            bool identity;
            var convSrc = GetStandardConversion<TSrc, U8>(key, NumberDataViewType.UInt64, out identity);
            var convU8 = GetStringConversion<U8>(NumberDataViewType.UInt64);
            return
                (in TSrc src, ref SB dst) =>
                {
                    ulong tmp = 0;
                    convSrc(in src, ref tmp);
                    if (tmp == 0 || tmp > count)
                        ClearDst(ref dst);
                    else
                    {
                        tmp = tmp - 1;
                        convU8(in tmp, ref dst);
                    }
                };
        }

        public TryParseMapper<TDst> GetTryParseConversion<TDst>(DataViewType typeDst)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.IsStandardScalar() || typeDst is KeyDataViewType, nameof(typeDst),
                "Parse conversion only supported for standard types");
            Contracts.Check(typeDst.RawType == typeof(TDst), "Wrong TDst type parameter");

            if (typeDst is KeyDataViewType keyType)
                return GetKeyTryParse<TDst>(keyType);

            Contracts.Assert(_tryParseDelegates.ContainsKey(typeDst.RawType));
            return (TryParseMapper<TDst>)_tryParseDelegates[typeDst.RawType];
        }

        private TryParseMapper<TDst> GetKeyTryParse<TDst>(KeyDataViewType key)
        {
            Contracts.Assert(key.RawType == typeof(TDst));

            // First parse as ulong, then convert to T.
            ulong max = key.Count - 1;

            var fnConv = GetKeyStandardConversion<TDst>();
            return
                (in TX src, out TDst dst) =>
                {
                    ulong uu;
                    dst = default(TDst);
                    if (!TryParseKey(in src, max, out uu))
                        return false;
                    // REVIEW: This call to fnConv should never need range checks, so could be made faster.
                    // Also, it would be nice to be able to assert that it doesn't overflow....
                    fnConv(in uu, ref dst);
                    return true;
                };
        }

        private Delegate GetKeyParse(KeyDataViewType key)
        {
            return Utils.MarshalInvoke(_getKeyParseMethodInfo, this, key.RawType, key);
        }

        private ValueMapper<TX, TDst> GetKeyParse<TDst>(KeyDataViewType key)
        {
            Contracts.Assert(key.RawType == typeof(TDst));

            // First parse as ulong, then convert to T.
            ulong max = key.Count - 1;

            var fnConv = GetKeyStandardConversion<TDst>();
            return
                (in TX src, ref TDst dst) =>
                {
                    ulong uu;
                    dst = default(TDst);
                    if (!TryParseKey(in src, max, out uu))
                    {
                        dst = default(TDst);
                        return;
                    }
                    // REVIEW: This call to fnConv should never need range checks, so could be made faster.
                    // Also, it would be nice to be able to assert that it doesn't overflow....
                    fnConv(in uu, ref dst);
                };
        }

        private ValueMapper<U8, TDst> GetKeyStandardConversion<TDst>()
        {
            var delegatesKey = (typeof(U8), typeof(TDst));
            if (!_delegatesStd.TryGetValue(delegatesKey, out Delegate del))
                throw Contracts.Except("No standard conversion from '{0}' to '{1}'", typeof(U8), typeof(TDst));
            return (ValueMapper<U8, TDst>)del;
        }

        private static StringBuilder ClearDst(ref StringBuilder dst)
        {
            if (dst == null)
                dst = new StringBuilder();
            else
                dst.Clear();
            return dst;
        }

        public InPredicate<T> GetIsDefaultPredicate<T>(DataViewType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(!(type is VectorDataViewType), nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            var t = type;
            Delegate del;
            if (!t.IsStandardScalar() && !(t is KeyDataViewType) || !_isDefaultDelegates.TryGetValue(t.RawType, out del))
                throw Contracts.Except("No IsDefault predicate for '{0}'", type);

            return (InPredicate<T>)del;
        }

        public InPredicate<T> GetIsNAPredicate<T>(DataViewType type)
        {
            InPredicate<T> pred;
            if (TryGetIsNAPredicate(type, out pred))
                return pred;
            throw Contracts.Except("No IsNA predicate for '{0}'", type);
        }

        public bool TryGetIsNAPredicate<T>(DataViewType type, out InPredicate<T> pred)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!TryGetIsNAPredicate(type, out del))
            {
                pred = null;
                return false;
            }

            Contracts.Assert(del is InPredicate<T>);
            pred = (InPredicate<T>)del;
            return true;
        }

        public bool TryGetIsNAPredicate(DataViewType type, out Delegate del)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(!(type is VectorDataViewType), nameof(type));

            var t = type;
            if (t is KeyDataViewType)
            {
                // REVIEW: Should we test for out of range when KeyCount > 0?
                Contracts.Assert(_isDefaultDelegates.ContainsKey(t.RawType));
                del = _isDefaultDelegates[t.RawType];
            }
            else if (!t.IsStandardScalar() || !_isNADelegates.TryGetValue(t.RawType, out del))
            {
                del = null;
                return false;
            }

            Contracts.Assert(del != null);
            return true;
        }

        public InPredicate<VBuffer<T>> GetHasMissingPredicate<T>(VectorDataViewType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.ItemType.RawType == typeof(T), nameof(type));

            var t = type.ItemType;
            Delegate del;
            if (t is KeyDataViewType)
            {
                // REVIEW: Should we test for out of range when KeyCount > 0?
                Contracts.Assert(_hasZeroDelegates.ContainsKey(t.RawType));
                del = _hasZeroDelegates[t.RawType];
            }
            else if (!t.IsStandardScalar() || !_hasNADelegates.TryGetValue(t.RawType, out del))
                throw Contracts.Except("No HasMissing predicate for '{0}'", type);

            return (InPredicate<VBuffer<T>>)del;
        }

        /// <summary>
        /// Returns the NA value of the given type, if it has one, otherwise, it returns
        /// default of the type. This only knows about NA values of standard scalar types
        /// and key types.
        /// </summary>
        public T GetNAOrDefault<T>(DataViewType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawType, out del))
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
        public T GetNAOrDefault<T>(DataViewType type, out bool isDefault)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawType, out del))
            {
                isDefault = true;
                return default(T);
            }

            T res = default(T);
            ((ValueGetter<T>)del)(ref res);
            isDefault = false;

#if DEBUG
            Delegate isDefPred;
            if (_isDefaultDelegates.TryGetValue(type.RawType, out isDefPred))
                Contracts.Assert(!((InPredicate<T>)isDefPred)(in res));
#endif

            return res;
        }

        /// <summary>
        /// Returns a ValueGetter{T} that produces the NA value of the given type, if it has one,
        /// otherwise, it produces default of the type. This only knows about NA values of standard
        /// scalar types and key types.
        /// </summary>
        public ValueGetter<T> GetNAOrDefaultGetter<T>(DataViewType type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type));

            Delegate del;
            if (!_getNADelegates.TryGetValue(type.RawType, out del))
                return (ref T res) => res = default(T);
            return (ValueGetter<T>)del;
        }

        // The IsNA methods are for efficient delegates (instance instead of static).
        #region IsNA
        private bool IsNA(in R4 src) => R4.IsNaN(src);
        private bool IsNA(in R8 src) => R8.IsNaN(src);
        #endregion IsNA

        #region HasNA
        private bool HasNA(in VBuffer<R4> src) { var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (R4.IsNaN(srcValues[i])) return true; } return false; }
        private bool HasNA(in VBuffer<R8> src) { var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (R8.IsNaN(srcValues[i])) return true; } return false; }
        #endregion HasNA

        #region IsDefault
        private bool IsDefault(in I1 src) => src == default(I1);
        private bool IsDefault(in I2 src) => src == default(I2);
        private bool IsDefault(in I4 src) => src == default(I4);
        private bool IsDefault(in I8 src) => src == default(I8);
        private bool IsDefault(in R4 src) => src == 0;
        private bool IsDefault(in R8 src) => src == 0;
        private bool IsDefault(in TX src) => src.IsEmpty;
        private bool IsDefault(in BL src) => src == default;
        private bool IsDefault(in U1 src) => src == 0;
        private bool IsDefault(in U2 src) => src == 0;
        private bool IsDefault(in U4 src) => src == 0;
        private bool IsDefault(in U8 src) => src == 0;
        private bool IsDefault(in UG src) => src.Equals(default(UG));
        private bool IsDefault(in TS src) => src.Equals(default(TS));
        private bool IsDefault(in DT src) => src.Equals(default(DT));
        private bool IsDefault(in DZ src) => src.Equals(default(DZ));
        #endregion IsDefault

        #region HasZero
        private bool HasZero(in VBuffer<U1> src) { if (!src.IsDense) return true; var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (srcValues[i] == 0) return true; } return false; }
        private bool HasZero(in VBuffer<U2> src) { if (!src.IsDense) return true; var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (srcValues[i] == 0) return true; } return false; }
        private bool HasZero(in VBuffer<U4> src) { if (!src.IsDense) return true; var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (srcValues[i] == 0) return true; } return false; }
        private bool HasZero(in VBuffer<U8> src) { if (!src.IsDense) return true; var srcValues = src.GetValues(); for (int i = 0; i < srcValues.Length; i++) { if (srcValues[i] == 0) return true; } return false; }
        #endregion HasZero

        #region GetNA
        private void GetNA(ref R4 value) => value = R4.NaN;
        private void GetNA(ref R8 value) => value = R8.NaN;
        #endregion GetNA

        #region ToI1
        public void Convert(in I1 src, ref I1 dst) => dst = src;
        public void Convert(in I2 src, ref I1 dst) => dst = (I1)src;
        public void Convert(in I4 src, ref I1 dst) => dst = (I1)src;
        public void Convert(in I8 src, ref I1 dst) => dst = (I1)src;
        #endregion ToI1

        #region ToI2
        public void Convert(in I1 src, ref I2 dst) => dst = src;
        public void Convert(in I2 src, ref I2 dst) => dst = src;
        public void Convert(in I4 src, ref I2 dst) => dst = (I2)src;
        public void Convert(in I8 src, ref I2 dst) => dst = (I2)src;
        #endregion ToI2

        #region ToI4
        public void Convert(in I1 src, ref I4 dst) => dst = src;
        public void Convert(in I2 src, ref I4 dst) => dst = src;
        public void Convert(in I4 src, ref I4 dst) => dst = src;
        public void Convert(in I8 src, ref I4 dst) => dst = (I4)src;
        #endregion ToI4

        #region ToI8
        public void Convert(in I1 src, ref I8 dst) => dst = src;
        public void Convert(in I2 src, ref I8 dst) => dst = src;
        public void Convert(in I4 src, ref I8 dst) => dst = src;
        public void Convert(in I8 src, ref I8 dst) => dst = src;

        public void Convert(in TS src, ref I8 dst) => dst = (I8)src.Ticks;
        public void Convert(in DT src, ref I8 dst) => dst = (I8)src.Ticks;
        public void Convert(in DZ src, ref I8 dst) => dst = (I8)src.UtcDateTime.Ticks;
        #endregion ToI8

        #region ToU1
        public void Convert(in U1 src, ref U1 dst) => dst = src;
        public void Convert(in U2 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(in U4 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(in U8 src, ref U1 dst) => dst = src <= U1.MaxValue ? (U1)src : (U1)0;
        public void Convert(in UG src, ref U1 dst) => dst = src.High == 0 && src.Low <= U1.MaxValue ? (U1)src.Low : (U1)0;
        #endregion ToU1

        #region ToU2
        public void Convert(in U1 src, ref U2 dst) => dst = src;
        public void Convert(in U2 src, ref U2 dst) => dst = src;
        public void Convert(in U4 src, ref U2 dst) => dst = src <= U2.MaxValue ? (U2)src : (U2)0;
        public void Convert(in U8 src, ref U2 dst) => dst = src <= U2.MaxValue ? (U2)src : (U2)0;
        public void Convert(in UG src, ref U2 dst) => dst = src.High == 0 && src.Low <= U2.MaxValue ? (U2)src.Low : (U2)0;
        #endregion ToU2

        #region ToU4
        public void Convert(in U1 src, ref U4 dst) => dst = src;
        public void Convert(in U2 src, ref U4 dst) => dst = src;
        public void Convert(in U4 src, ref U4 dst) => dst = src;
        public void Convert(in U8 src, ref U4 dst) => dst = src <= U4.MaxValue ? (U4)src : (U4)0;
        public void Convert(in UG src, ref U4 dst) => dst = src.High == 0 && src.Low <= U4.MaxValue ? (U4)src.Low : (U4)0;
        #endregion ToU4

        #region ToU8
        public void Convert(in U1 src, ref U8 dst) => dst = src;
        public void Convert(in U2 src, ref U8 dst) => dst = src;
        public void Convert(in U4 src, ref U8 dst) => dst = src;
        public void Convert(in U8 src, ref U8 dst) => dst = src;
        public void Convert(in UG src, ref U8 dst) => dst = src.High == 0 ? src.Low : (U8)0;
        #endregion ToU8

        #region ToUG
        public void Convert(in U1 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(in U2 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(in U4 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(in U8 src, ref UG dst) => dst = new UG(src, 0);
        public void Convert(in UG src, ref UG dst) => dst = src;
        #endregion ToUG

        #region ToR4
        public void Convert(in I1 src, ref R4 dst) => dst = (R4)src;
        public void Convert(in I2 src, ref R4 dst) => dst = (R4)src;
        public void Convert(in I4 src, ref R4 dst) => dst = (R4)src;
        public void Convert(in I8 src, ref R4 dst) => dst = (R4)src;
        public void Convert(in U1 src, ref R4 dst) => dst = src;
        public void Convert(in U2 src, ref R4 dst) => dst = src;
        public void Convert(in U4 src, ref R4 dst) => dst = src;
        // REVIEW: The 64-bit JIT has a bug in that it rounds incorrectly from ulong
        // to floating point when the high bit of the ulong is set. Should we work around the bug
        // or just live with it? See the DoubleParser code for details.
        public void Convert(in U8 src, ref R4 dst) => dst = src;

        public void Convert(in TS src, ref R4 dst) => dst = (R4)src.Ticks;
        public void Convert(in DT src, ref R4 dst) => dst = (R4)src.Ticks;
        public void Convert(in DZ src, ref R4 dst) => dst = (R4)src.UtcDateTime.Ticks;
        #endregion ToR4

        #region ToR8
        public void Convert(in I1 src, ref R8 dst) => dst = (R8)src;
        public void Convert(in I2 src, ref R8 dst) => dst = (R8)src;
        public void Convert(in I4 src, ref R8 dst) => dst = (R8)src;
        public void Convert(in I8 src, ref R8 dst) => dst = (R8)src;
        public void Convert(in U1 src, ref R8 dst) => dst = src;
        public void Convert(in U2 src, ref R8 dst) => dst = src;
        public void Convert(in U4 src, ref R8 dst) => dst = src;
        // REVIEW: The 64-bit JIT has a bug in that it rounds incorrectly from ulong
        // to floating point when the high bit of the ulong is set. Should we work around the bug
        // or just live with it? See the DoubleParser code for details.
        public void Convert(in U8 src, ref R8 dst) => dst = src;

        public void Convert(in TS src, ref R8 dst) => dst = (R8)src.Ticks;
        public void Convert(in DT src, ref R8 dst) => dst = (R8)src.Ticks;
        public void Convert(in DZ src, ref R8 dst) => dst = (R8)src.UtcDateTime.Ticks;
        #endregion ToR8

        #region ToStringBuilder
        public void Convert(in I1 src, ref SB dst) { ClearDst(ref dst); dst.Append(src); }
        public void Convert(in I2 src, ref SB dst) { ClearDst(ref dst); dst.Append(src); }
        public void Convert(in I4 src, ref SB dst) { ClearDst(ref dst); dst.Append(src); }
        public void Convert(in I8 src, ref SB dst) { ClearDst(ref dst); dst.Append(src); }
        public void Convert(in U1 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(in U2 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(in U4 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(in U8 src, ref SB dst) => ClearDst(ref dst).Append(src);
        public void Convert(in UG src, ref SB dst) { ClearDst(ref dst); dst.AppendFormat("0x{0:x16}{1:x16}", src.High, src.Low); }
        public void Convert(in R4 src, ref SB dst) { ClearDst(ref dst); if (R4.IsNaN(src)) dst.AppendFormat(CultureInfo.InvariantCulture, "{0}", "?"); else dst.AppendFormat(CultureInfo.InvariantCulture, "{0:R}", src); }
        public void Convert(in R8 src, ref SB dst) { ClearDst(ref dst); if (R8.IsNaN(src)) dst.AppendFormat(CultureInfo.InvariantCulture, "{0}", "?"); else dst.AppendFormat(CultureInfo.InvariantCulture, "{0:G17}", src); }
        public void Convert(in BL src, ref SB dst)
        {
            ClearDst(ref dst);
            if (!src)
                dst.Append("0");
            else
                dst.Append("1");
        }
        public void Convert(in TS src, ref SB dst) { ClearDst(ref dst); dst.AppendFormat("{0:c}", src); }
        public void Convert(in DT src, ref SB dst) { ClearDst(ref dst); dst.AppendFormat("{0:o}", src); }
        public void Convert(in DZ src, ref SB dst) { ClearDst(ref dst); dst.AppendFormat("{0:o}", src); }
        #endregion ToStringBuilder

        #region ToTX
        public void Convert(in I1 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in I2 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in I4 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in I8 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in U1 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in U2 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in U4 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in U8 src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in UG src, ref TX dst) => dst = string.Format("0x{0:x16}{1:x16}", src.High, src.Low).AsMemory();
        public void Convert(in R4 src, ref TX dst) => dst = src.ToString("G7", CultureInfo.InvariantCulture).AsMemory();
        public void Convert(in R8 src, ref TX dst) => dst = src.ToString("G17", CultureInfo.InvariantCulture).AsMemory();
        public void Convert(in BL src, ref TX dst) => dst = src.ToString().AsMemory();
        public void Convert(in TS src, ref TX dst) => dst = string.Format("{0:c}", src).AsMemory();
        public void Convert(in DT src, ref TX dst) => dst = string.Format("{0:o}", src).AsMemory();
        public void Convert(in DZ src, ref TX dst) => dst = string.Format("{0:o}", src).AsMemory();
        #endregion ToTX

        #region ToBL
        public void Convert(in R8 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in R4 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in I1 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in I2 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in I4 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in I8 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in U1 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in U2 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in U4 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        public void Convert(in U8 src, ref BL dst) => dst = System.Convert.ToBoolean(src);
        #endregion

        #region FromR4
        public void Convert(in R4 src, ref R4 dst) => dst = src;
        public void Convert(in R4 src, ref R8 dst) => dst = src;
        #endregion FromR4

        #region FromR8
        public void Convert(in R8 src, ref R4 dst) => dst = (R4)src;
        public void Convert(in R8 src, ref R8 dst) => dst = src;
        #endregion FromR8

        #region FromTX

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// </summary>
        public bool TryParse(in TX src, out U1 dst)
        {
            ulong res;
            if (!TryParse(in src, out res) || res > U1.MaxValue)
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
        public bool TryParse(in TX src, out U2 dst)
        {
            ulong res;
            if (!TryParse(in src, out res) || res > U2.MaxValue)
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
        public bool TryParse(in TX src, out U4 dst)
        {
            ulong res;
            if (!TryParse(in src, out res) || res > U4.MaxValue)
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
        public bool TryParse(in TX src, out U8 dst)
        {
            if (src.IsEmpty)
            {
                dst = 0;
                return false;
            }

            return TryParseCore(src.Span, out dst);
        }

        /// <summary>
        /// A parse method that transforms a 34-length string into a <see cref="DataViewRowId"/>.
        /// </summary>
        /// <param name="src">What should be a 34-length hexadecimal representation, including a 0x prefix,
        /// of the 128-bit number</param>
        /// <param name="dst">The result</param>
        /// <returns>Whether the input string was parsed successfully, that is, it was exactly length 32
        /// and had only digits and the letters 'a' through 'f' or 'A' through 'F' as characters</returns>
        public bool TryParse(in TX src, out UG dst)
        {
            var span = src.Span;
            // REVIEW: Accommodate numeric inputs?
            if (src.Length != 34 || span[0] != '0' || (span[1] != 'x' && span[1] != 'X'))
            {
                dst = default(UG);
                return false;
            }

            int offset = 2;
            ulong hi = 0;
            ulong num = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int d = 0; d < 16; ++d)
                {
                    num <<= 4;
                    char c = span[offset++];
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
            Contracts.Assert(offset == src.Length);
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
        private bool IsStdMissing(ref ReadOnlySpan<char> span)
        {
            Contracts.Assert(!span.IsEmpty);

            char ch;
            switch (span.Length)
            {
                default:
                    return false;

                case 1:
                    if (span[0] == '?')
                        return true;
                    return false;
                case 2:
                    if ((ch = span[0]) != 'N' && ch != 'n')
                        return false;
                    if ((ch = span[1]) != 'A' && ch != 'a')
                        return false;
                    return true;
                case 3:
                    if ((ch = span[0]) != 'N' && ch != 'n')
                        return false;
                    if ((ch = span[1]) == '/')
                    {
                        // Check for N/A.
                        if ((ch = span[2]) != 'A' && ch != 'a')
                            return false;
                    }
                    else
                    {
                        // Check for NaN.
                        if (ch != 'a' && ch != 'A')
                            return false;
                        if ((ch = span[2]) != 'N' && ch != 'n')
                            return false;
                    }
                    return true;
            }
        }

        /// <summary>
        /// Utility to assist in parsing key-type values. The max value defines
        /// the legal input value bound. The output dst value is "normalized" by adding 1
        /// so max is mapped to 1 + max.
        /// Unparsable or out of range values are mapped to zero with a false return.
        /// </summary>
        public bool TryParseKey(in TX src, U8 max, out U8 dst)
        {
            var span = src.Span;
            // Both empty and missing map to zero (NA for key values) and that mapping is valid,
            // hence the true return.
            if (src.IsEmpty || IsStdMissing(ref span))
            {
                dst = 0;
                return true;
            }

            // This simply ensures we don't have max == U8.MaxValue. This is illegal since
            // it would cause max to overflow to zero. Specifically, it protects
            // against overflow in the expression uu + 1 below.
            Contracts.Assert(max < U8.MaxValue);

            // Parse a ulong.
            ulong uu;
            if (!TryParseCore(span, out uu))
            {
                dst = 0;
                // Return true only for standard forms for NA.
                return false;
            }

            if (uu > max)
            {
                dst = 0;
                return false;
            }

            dst = uu + 1;
            return true;
        }

        private bool TryParseCore(ReadOnlySpan<char> span, out ulong dst)
        {
            ulong res = 0;
            int ich = 0;
            while (ich < span.Length)
            {
                uint d = (uint)span[ich++] - (uint)'0';
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
        /// On failure, it sets dst to the default value.
        /// </summary>
        public bool TryParse(in TX src, out I1 dst)
        {
            dst = default;
            TryParseSigned(I1.MaxValue, in src, out long? res);
            if (res == null)
            {
                dst = default;
                return false;
            }
            Contracts.Assert(res.HasValue);
            Contracts.Check((I1)res == res, "Overflow or underflow occurred while converting value in text to sbyte.");
            dst = (I1)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the default value.
        /// </summary>
        public bool TryParse(in TX src, out I2 dst)
        {
            dst = default;
            TryParseSigned(I2.MaxValue, in src, out long? res);
            if (res == null)
            {
                dst = default;
                return false;
            }
            Contracts.Assert(res.HasValue);
            Contracts.Check((I2)res == res, "Overflow or underflow occurred while converting value in text to short.");
            dst = (I2)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the defualt value.
        /// </summary>
        public bool TryParse(in TX src, out I4 dst)
        {
            dst = default;
            TryParseSigned(I4.MaxValue, in src, out long? res);
            if (res == null)
            {
                dst = default;
                return false;
            }
            Contracts.Assert(res.HasValue);
            Contracts.Check((I4)res == res, "Overflow or underflow occurred while converting value in text to int.");
            dst = (I4)res;
            return true;
        }

        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable or overflows.
        /// On failure, it sets dst to the default value.
        /// </summary>
        public bool TryParse(in TX src, out I8 dst)
        {
            dst = default;
            TryParseSigned(I8.MaxValue, in src, out long? res);
            if (res == null)
            {
                dst = default;
                return false;
            }
            Contracts.Assert(res.HasValue);
            dst = (I8)res;
            return true;
        }

        /// <summary>
        /// Returns false if the text is not parsable as an non-negative long or overflows.
        /// </summary>
        private bool TryParseNonNegative(ReadOnlySpan<char> span, out long result)
        {
            long res = 0;
            int ich = 0;
            while (ich < span.Length)
            {
                Contracts.Assert(res >= 0);
                uint d = (uint)span[ich++] - (uint)'0';
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
        /// or the result overflows. The min legal value is -max. The NA value null.
        /// When it returns false, result is set to the NA value. The result can be NA on true return,
        /// since some representations of NA are not considered parse failure.
        /// </summary>
        private void TryParseSigned(long max, in TX text, out long? result)
        {
            Contracts.Assert(max > 0);
            Contracts.Assert((max & (max + 1)) == 0);

            if (text.IsEmpty)
            {
                result = default(long);
                return;
            }

            ulong val;
            var span = text.Span;
            if (span[0] == '-')
            {
                if (span.Length == 1 || !TryParseCore(span.Slice(1), out val) || (val > ((ulong)max + 1)))
                {
                    result = null;
                    return;
                }
                Contracts.Assert(val >= 0);
                result = -(long)val;
                Contracts.Assert(long.MinValue <= result && result <= 0);
                return;
            }

            long sVal;
            if (!TryParseNonNegative(span, out sVal))
            {
                result = null;
                return;
            }

            Contracts.Assert(sVal >= 0);
            if (sVal > max)
            {
                result = null;
                return;
            }

            result = (long)sVal;
            Contracts.Assert(0 <= result && result <= long.MaxValue);
            return;
        }

        /// <summary>
        /// This produces zero for empty, or NaN depending on the <see cref="DoubleParser.OptionFlags.EmptyAsNaN"/> used.
        /// It returns false if the text is not parsable.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(in TX src, out R4 dst)
        {
            var span = src.Span;
            if (DoubleParser.TryParse(span, out dst, _doubleParserOptionFlags))
                return true;
            dst = R4.NaN;
            return IsStdMissing(ref span);
        }

        /// <summary>
        /// This produces zero for empty, or NaN depending on the <see cref="DoubleParser.OptionFlags.EmptyAsNaN"/> used.
        /// It returns false if the text is not parsable.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public bool TryParse(in TX src, out R8 dst)
        {
            var span = src.Span;
            if (DoubleParser.TryParse(span, out dst, _doubleParserOptionFlags))
                return true;
            dst = R8.NaN;
            return IsStdMissing(ref span);
        }

        /// <summary>
        /// This produces default for empty.
        /// </summary>
        public bool TryParse(in TX src, out TS dst)
        {
            if (src.IsEmpty)
            {
                dst = default;
                return true;
            }

            if (TimeSpan.TryParse(src.ToString(), CultureInfo.InvariantCulture, out dst))
                return true;
            dst = default;
            return false;
        }

        /// <summary>
        /// This produces default for empty.
        /// </summary>
        public bool TryParse(in TX src, out DT dst)
        {
            if (src.IsEmpty)
            {
                dst = default;
                return true;
            }

            if (DateTime.TryParse(src.ToString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out dst))
                return true;
            dst = default;
            return false;
        }

        /// <summary>
        /// This produces default for empty.
        /// </summary>
        public bool TryParse(in TX src, out DZ dst)
        {
            if (src.IsEmpty)
            {
                dst = default;
                return true;
            }

            if (DateTimeOffset.TryParse(src.ToString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out dst))
                return true;

            dst = default;
            return false;
        }

        // These throw an exception for unparsable and overflow values.
        private I1 ParseI1(in TX src)
        {
            TryParseSigned(I1.MaxValue, in src, out long? res);
            Contracts.Check(res.HasValue, "Value could not be parsed from text to sbyte.");
            Contracts.Check((I1)res == res, "Overflow or underflow occurred while converting value in text to sbyte.");
            return (I1)res;
        }

        private I2 ParseI2(in TX src)
        {
            TryParseSigned(I2.MaxValue, in src, out long? res);
            Contracts.Check(res.HasValue, "Value could not be parsed from text to short.");
            Contracts.Check((I2)res == res, "Overflow or underflow occurred while converting value in text to short.");
            return (I2)res;
        }

        private I4 ParseI4(in TX src)
        {
            TryParseSigned(I4.MaxValue, in src, out long? res);
            Contracts.Check(res.HasValue, "Value could not be parsed from text to int.");
            Contracts.Check((I4)res == res, "Overflow or underflow occurred while converting value in text to int.");
            return (I4)res;
        }

        private I8 ParseI8(in TX src)
        {
            TryParseSigned(I8.MaxValue, in src, out long? res);
            Contracts.Check(res.HasValue, "Value could not be parsed from text to long.");
            return res.Value;
        }

        // These map unparsable and overflow values to zero. The unsigned integer types do not have an NA value.
        // Note that this matches the "bad" value for key-types, which will likely be the primary use for
        // unsigned integer types.
        private U1 ParseU1(in TX span)
        {
            ulong res;
            if (!TryParse(in span, out res))
                return 0;
            if (res > U1.MaxValue)
                return 0;
            return (U1)res;
        }

        private U2 ParseU2(in TX span)
        {
            ulong res;
            if (!TryParse(in span, out res))
                return 0;
            if (res > U2.MaxValue)
                return 0;
            return (U2)res;
        }

        private U4 ParseU4(in TX span)
        {
            ulong res;
            if (!TryParse(in span, out res))
                return 0;
            if (res > U4.MaxValue)
                return 0;
            return (U4)res;
        }

        private U8 ParseU8(in TX span)
        {
            ulong res;
            if (!TryParse(in span, out res))
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
        public bool TryParse(in TX src, out BL dst)
        {
            var span = src.Span;

            char ch;
            switch (src.Length)
            {
                case 0:
                    // Empty succeeds and maps to false.
                    dst = false;
                    return true;

                case 1:
                    switch (span[0])
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
                    switch (span[0])
                    {
                        case 'N':
                        case 'n':
                            if ((ch = span[1]) != 'O' && ch != 'o')
                                break;
                            dst = false;
                            return true;
                        case '+':
                            if ((ch = span[1]) != '1')
                                break;
                            dst = true;
                            return true;
                        case '-':
                            if ((ch = span[1]) != '1')
                                break;
                            dst = false;
                            return true;
                    }
                    break;

                case 3:
                    switch (span[0])
                    {
                        case 'Y':
                        case 'y':
                            if ((ch = span[1]) != 'E' && ch != 'e')
                                break;
                            if ((ch = span[2]) != 'S' && ch != 's')
                                break;
                            dst = true;
                            return true;
                    }
                    break;

                case 4:
                    switch (span[0])
                    {
                        case 'T':
                        case 't':
                            if ((ch = span[1]) != 'R' && ch != 'r')
                                break;
                            if ((ch = span[2]) != 'U' && ch != 'u')
                                break;
                            if ((ch = span[3]) != 'E' && ch != 'e')
                                break;
                            dst = true;
                            return true;
                    }
                    break;

                case 5:
                    switch (span[0])
                    {
                        case 'F':
                        case 'f':
                            if ((ch = span[1]) != 'A' && ch != 'a')
                                break;
                            if ((ch = span[2]) != 'L' && ch != 'l')
                                break;
                            if ((ch = span[3]) != 'S' && ch != 's')
                                break;
                            if ((ch = span[4]) != 'E' && ch != 'e')
                                break;
                            dst = false;
                            return true;
                    }
                    break;
            }

            dst = false;
            return false;
        }

        private bool TryParse(in TX src, out TX dst)
        {
            dst = src;
            return true;
        }

        public void Convert(in TX span, ref I1 value)
        {
            value = ParseI1(in span);
        }
        public void Convert(in TX span, ref U1 value)
        {
            value = ParseU1(in span);
        }
        public void Convert(in TX span, ref I2 value)
        {
            value = ParseI2(in span);
        }
        public void Convert(in TX span, ref U2 value)
        {
            value = ParseU2(in span);
        }
        public void Convert(in TX span, ref I4 value)
        {
            value = ParseI4(in span);
        }
        public void Convert(in TX span, ref U4 value)
        {
            value = ParseU4(in span);
        }
        public void Convert(in TX span, ref I8 value)
        {
            value = ParseI8(in span);
        }
        public void Convert(in TX span, ref U8 value)
        {
            value = ParseU8(in span);
        }
        public void Convert(in TX span, ref UG value)
        {
            if (!TryParse(in span, out value))
                Contracts.Assert(value.Equals(default(UG)));
        }
        public void Convert(in TX src, ref R4 value)
        {
            var span = src.Span;
            if (DoubleParser.TryParse(span, out value, _doubleParserOptionFlags))
                return;
            // Unparsable is mapped to NA.
            value = R4.NaN;
        }
        public void Convert(in TX src, ref R8 value)
        {
            var span = src.Span;
            if (DoubleParser.TryParse(span, out value, _doubleParserOptionFlags))
                return;
            // Unparsable is mapped to NA.
            value = R8.NaN;
        }
        public void Convert(in TX span, ref TX value)
        {
            value = span;
        }
        public void Convert(in TX src, ref BL value)
        {
            // When TryParseBL returns false, it should have set value to false.
            if (!TryParse(in src, out value))
                Contracts.Assert(!value);
        }
        public void Convert(in TX src, ref SB dst)
        {
            ClearDst(ref dst);
            if (!src.IsEmpty)
                dst.AppendMemory(src);
        }

        public void Convert(in TX span, ref TS value) => TryParse(in span, out value);
        public void Convert(in TX span, ref DT value) => TryParse(in span, out value);
        public void Convert(in TX span, ref DZ value) => TryParse(in span, out value);

        #endregion FromTX

        #region FromBL
        public void Convert(in BL src, ref I1 dst) => dst = System.Convert.ToSByte(src);
        public void Convert(in BL src, ref I2 dst) => dst = System.Convert.ToInt16(src);
        public void Convert(in BL src, ref I4 dst) => dst = System.Convert.ToInt32(src);
        public void Convert(in BL src, ref I8 dst) => dst = System.Convert.ToInt64(src);
        public void Convert(in BL src, ref R4 dst) => dst = System.Convert.ToSingle(src);
        public void Convert(in BL src, ref R8 dst) => dst = System.Convert.ToDouble(src);
        public void Convert(in BL src, ref BL dst) => dst = src;
        #endregion FromBL

        #region ToDT
        public void Convert(in DT src, ref DT dst) => dst = src;
        #endregion ToDT
    }
}
