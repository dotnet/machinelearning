// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{

    /// <summary>
    /// Represents a binned feature
    /// </summary>
    public abstract class Feature
    {
        private IntArray _bins;

        /// <summary>
        /// The type of the feature. This is serialized as part of the Bing extraction BIN file,
        /// so it should remain binary compatible from version to version.
        /// </summary>
        public enum FeatureType
        {
            Raw = 0,
            // There was a different feature type "derived" that actually was not serializable in
            // a bin file, that had value 1. So 1 is now reserved.
            Meta = 2
        }

#if !NO_STORE
        private long _binsOffset = -1;
        private int _memorySize = 0;
        private long _binsSize = 0;
#endif

        public bool IsTrivialFeature { get; private set; }
        public MD5Hash MD5Hash { get; private set; }
        public IntArrayType BinsType { get; private set; }
#if !NO_STORE
        public FileObjectStore<IntArrayFormatter> BinsCache { get; set; }
#endif
        protected Feature(IntArray bins)
        {
            Bins = bins;
#if !NO_STORE
            BinsCache = FileObjectStore<IntArrayFormatter>.GetDefaultInstance();
#endif
            IsTrivialFeature = bins.BitsPerItem == IntArrayBits.Bits0;

            if (!IsTrivialFeature && bins.Length > 0)
            {
                MD5Hash = bins.MD5Hash;
            }

            BinsType = bins.Type;
        }

#if !NO_STORE
        /// <summary>
        /// Retrieves the feature bins from the file object store
        /// </summary>
        /// <returns>The feature bins</returns>
        private IntArray RestoreBins()
        {
            return (IntArray)BinsCache.ReadObject(_binsOffset, _binsSize);
        }

        /// <summary>
        /// Saves the feature bins into the file object store
        /// </summary>
        /// <param name="bins">The feature bins to store</param>
        private void SaveBins(IntArray bins)
        {
            _binsSize = BinsCache.WriteObject(ref _binsOffset, bins);
            BinsType = bins.Type;
            _memorySize = bins.SizeInBytes();
        }
#endif

        public static Feature New(byte[] buffer, ref int position)
        {
            using (Timer.Time(TimerEvent.ConstructFromByteArray))
            {
                FeatureType type = (FeatureType)buffer.ToInt(ref position);

                switch (type)
                {
                case FeatureType.Raw:
                    TsvFeature tf = new TsvFeature(buffer, ref position);
#if !NO_STORE
                    tf.BinsCache = FileObjectStore<IntArrayFormatter>.GetDefaultInstance();
#endif
                    return tf;
                default:
                    throw Contracts.Except("Impossible!");
                }
            }
        }

        protected Feature(byte[] buffer, ref int position)
        {
            Bins = IntArray.New(buffer, ref position);
#if !NO_STORE
            BinsCache = FileObjectStore<IntArrayFormatter>.GetDefaultInstance();
#endif
        }

        public abstract string LookupName { get; }

        /// <summary>
        /// Returns the number of entires (documents) in the feature
        /// </summary>
        public int Length
        {
            get { return Bins.Length; }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public virtual int SizeInBytes()
        {
#if NO_STORE
            return sizeof(int) + _bins.SizeInBytes();
#else
            return sizeof(int) + ((this._bins != null) ? this._bins.SizeInBytes() : _memorySize);
#endif
        }

        public abstract FeatureType Type { get; }

        /// <summary>
        /// Writes a binary representation of this class to a byte buffer, at a given position.
        /// The position is incremented to the end of the representation
        /// </summary>
        /// <param name="buffer">a byte array where the binary represenaion is written</param>
        /// <param name="position">the position in the byte array</param>
        public virtual void ToByteArray(byte[] buffer, ref int position)
        {
            ((int)Type).ToByteArray(buffer, ref position);
            Bins.ToByteArray(buffer, ref position);
        }

        public byte[] ToByteArray()
        {
            int position = 0;
            byte[] buffer = new byte[SizeInBytes()];
            ToByteArray(buffer, ref position);
            return buffer;
        }

        /// <summary>
        /// Gets the compactIntArray of bin values.
        /// </summary>
        /// <value>The bin values.</value>
        public IntArray Bins
        {
            get
            {
#if !NO_STORE
                if (_bins == null &&
                    BinsCache != null &&
                    BinsCache.Initialized)
                {
                    lock (this)
                    {
                        if (_bins == null)
                        {
                            _bins = RestoreBins();
                        }
                    }
                }
#endif
                return _bins;
            }
            set
            {
#if !NO_STORE
                // only save the bins structure the first time we set it to null
                if (value == null &&
                    _bins != null &&
                    _binsOffset == -1 &&
                    BinsCache != null &&
                    BinsCache.Initialized)
                {
                    SaveBins(_bins);
                }
#endif
                _bins = value;
            }
        }
    }

    public sealed class TsvFeature : Feature
    {
        private readonly uint[] _valueMap;
        private string _name;

        /// <summary>
        /// Initializes a new instance of the <see cref="Feature"/> class.
        /// </summary>
        /// <param name="bins">The bins.</param>
        /// <param name="valueMap"></param>
        /// <param name="name">The name.</param>
        public TsvFeature(IntArray bins, uint[] valueMap, string name)
            : base(bins)
        {
            _valueMap = valueMap;
            _name = name;
        }

        /// <summary>
        /// Constructs an empty (all zero) feature
        /// </summary>
        /// <param name="name"></param>
        /// <param name="length"></param>
        public TsvFeature(string name, int length)
            : base(DenseIntArray.New(length, IntArrayType.Dense, 0, Enumerable.Repeat(0, length)))
        {
            _valueMap = new uint[1];
            _name = name;
        }

        public TsvFeature(byte[] buffer, ref int position)
            : base(buffer, ref position)
        {
            _valueMap = buffer.ToUIntArray(ref position);
            _name = buffer.ToString(ref position);
        }

        public override string LookupName
        {
            get { return _name; }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return base.SizeInBytes() + _valueMap.SizeInBytes() + _name.SizeInBytes();
        }

        public override FeatureType Type
        {
            get
            {
                return FeatureType.Raw;
            }
        }

        public void SetName(string name)
        {
            _name = name;
        }

        /// <summary>
        /// Writes a binary representation of this class to a byte buffer, at a given position.
        /// The position is incremented to the end of the representation
        /// </summary>
        /// <param name="buffer">a byte array where the binary represenaion is written</param>
        /// <param name="position">the position in the byte array</param>
        public override void ToByteArray(byte[] buffer, ref int position)
        {
            base.ToByteArray(buffer, ref position);
            _valueMap.ToByteArray(buffer, ref position);
            _name.ToByteArray(buffer, ref position);
        }

        /// <summary>
        /// Gets the value that represents each bin
        /// </summary>
        public uint[] ValueMap
        {
            get { return _valueMap; }
        }

        public TsvFeature[] Split(int[][] assignment)
        {
            return Bins.Split(assignment)
                .Select(bins => new TsvFeature(bins, _valueMap, _name)).ToArray();
        }

        /// <summary>
        /// Clone a TSVFeature containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned TSVFeature  </param>
        /// <returns> The cloned TSVFeature </returns>
        public TsvFeature Clone(int[] itemIndices)
        {
            return new TsvFeature(Bins.Clone(itemIndices), _valueMap, _name);
        }

        /// <summary>
        /// Concatenates an array of features into one long feature
        /// </summary>
        /// <param name="parts">An array of features</param>
        /// <returns>A concatenated feature</returns>
        public static TsvFeature Concat(TsvFeature[] parts)
        {
            IntArrayBits bitsPerItem = IntArrayBits.Bits0;
            if (parts.Length == 1)
            {
                bitsPerItem = IntArray.NumBitsNeeded(parts[0].ValueMap.Length);
                if (bitsPerItem == parts[0].Bins.BitsPerItem)
                    return parts[0];
                IntArray b = parts[0].Bins;
                IntArray newBins = IntArray.New(b.Length, b.Type, bitsPerItem, b);
                return new TsvFeature(newBins, parts[0].ValueMap, parts[0]._name);
            }

            uint[] concatValueMap = Algorithms.MergeSortedUniqued(parts.Select(x => x.ValueMap).ToArray());
            bitsPerItem = IntArray.NumBitsNeeded(concatValueMap.Length);
            IntArray concatBins = ConcatBins(parts, concatValueMap);

            return new TsvFeature(concatBins, concatValueMap, parts[0]._name);
        }

        private static int[] MakeBinMap(uint[] oldValueMap, uint[] newValueMap)
        {
            int[] binMap = new int[oldValueMap.Length];
            int mappedBin = 0;
            for (int j = 0; j < oldValueMap.Length; ++j)
            {
                while (newValueMap[mappedBin] < oldValueMap[j])
                    ++mappedBin;
                binMap[j] = mappedBin;
            }

            return binMap;
        }

        private static IntArray ConcatBins(TsvFeature[] parts, uint[] concatValueMap)
        {
            using (Timer.Time(TimerEvent.ConcatBins))
            {
                int length = parts.Sum(x => x.Length);

                IntArrayBits bitsPerItem = IntArray.NumBitsNeeded(concatValueMap.Length);
                DenseIntArray concatBins = (DenseIntArray)IntArray.New(length, IntArrayType.Dense, bitsPerItem);

                int pos = 0;

                for (int partIndex = 0; partIndex < parts.Length; ++partIndex)
                {
                    IntArray bins = parts[partIndex].Bins;

                    if (concatValueMap.Length == parts[partIndex].ValueMap.Length)
                    {
                        foreach (int bin in bins)
                        {
                            concatBins[pos++] = bin;
                        }
                    }
                    else
                    {
                        int[] binMap = MakeBinMap(parts[partIndex]._valueMap, concatValueMap);

                        foreach (int bin in bins)
                        {
                            concatBins[pos++] = binMap[bin];
                        }
                    }
                }

                if (bitsPerItem != IntArrayBits.Bits0 && parts.All(x => x.Bins is DeltaSparseIntArray))
                {
                    return new DeltaSparseIntArray(length, bitsPerItem, concatBins);
                }
                else
                {
                    return concatBins;
                }
            }
        }

        private static int[] MakeBinMap(uint[] oldValueMap, double[] newBinUpperBounds)
        {
            int[] binMap = new int[oldValueMap.Length];
            int mappedBin = 0;
            for (int j = 0; j < oldValueMap.Length; ++j)
            {
                while (newBinUpperBounds[mappedBin] < oldValueMap[j])
                    ++mappedBin;
                binMap[j] = mappedBin;
            }

            return binMap;
        }
    }
}
