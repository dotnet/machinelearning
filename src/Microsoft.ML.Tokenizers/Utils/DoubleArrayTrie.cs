// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The implementation of the DoubleArrayBuilder class is based on the following C# port of the C++ implementation of the Double-Array Trie (DART) data structure.
// The original C++ implementation is available at https://github.com/s-yata/darts-clone/blob/master/include/darts.h and used under BSD 2-clause license.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    //
    // Succinct bit vector.
    //
    public class BitVector
    {
        private const int UnitSize = sizeof(uint) * 8;
        private readonly List<uint> _units = new();
        private uint[]? _ranks;
        private uint _numOnes;
        private uint _size;

        public BitVector() { }

        public bool this[int id]
        {
            get => (_units[id / UnitSize] >> (id % UnitSize) & 1) == 1;
        }

        private static uint PopCount(uint unit)
        {
            unit = ((unit & 0xAAAAAAAA) >> 1) + (unit & 0x55555555);
            unit = ((unit & 0xCCCCCCCC) >> 2) + (unit & 0x33333333);
            unit = ((unit >> 4) + unit) & 0x0F0F0F0F;
            unit += unit >> 8;
            unit += unit >> 16;
            return unit & 0xFF;
        }

        public uint Rank(uint id)
        {
            uint unitId = id / UnitSize;
            return (uint)(_ranks![(int)unitId] + PopCount((uint)(_units[(int)unitId] & (~0U >> (int)(UnitSize - (id % UnitSize) - 1)))));
        }

        public void Set(uint id, bool bit)
        {
            if (bit)
            {
                _units[(int)(id / UnitSize)] |= 1U << (int)(id % UnitSize);
            }
            else
            {
                _units[(int)(id / UnitSize)] &= ~(1U << (int)(id % UnitSize));
            }
        }

        public bool IsEmpty => _units.Count == 0;

        public uint NumOnes => _numOnes;

        public uint Size => _size;

        public void Append()
        {
            if ((_size % UnitSize) == 0)
            {
                _units.Add(0);
            }
            ++_size;
        }

        public void Build()
        {
            _ranks = new uint[_units.Count];

            _numOnes = 0;
            for (int i = 0; i < _units.Count; ++i)
            {
                _ranks[i] = _numOnes;
                _numOnes += PopCount(_units[i]);
            }
        }
    }

    internal class AutoPool<T>
    {
        private T[] _buf = Array.Empty<T>();
        private int _size;
        private int _capacity;

        public AutoPool() { }

        public ref T this[int id]
        {
            get => ref _buf[id];
        }

        public T[] Buffer => _buf;

        public bool Empty => _size == 0;

        public int Size => _size;

        public void Clear()
        {
            _buf = Array.Empty<T>();
            _size = 0;
            _capacity = 0;
        }

        public void ResizeBuf(int size)
        {
            if (size <= _capacity)
            {
                return;
            }

            int capacity;
            if (size >= _capacity * 2)
            {
                capacity = size;
            }
            else
            {
                capacity = 1;
                while (capacity < size)
                {
                    capacity <<= 1;
                }
            }

            T[] buf = new T[capacity];

            if (_buf is not null)
            {
                Array.Copy(_buf, buf, _size);
            }

            _buf = buf;
            _capacity = capacity;
        }

        public void Append(T value)
        {
            if (_size == _capacity)
            {
                ResizeBuf(_size + 1);
            }

            _buf[_size++] = value!;
        }

        public void Append()
        {
            if (_size == _capacity)
            {
                ResizeBuf(_size + 1);
            }
            _buf[_size++] = default!;
        }

        public void PushBack(T value) => Append(value);

        public void PopBack()
        {
            if (Empty)
            {
                return;
            }

            _buf[--_size] = default!;
        }

        public void Resize(int size)
        {
            while (_size > size)
            {
                _buf[--_size] = default!;
            }

            if (size > _capacity)
            {
                ResizeBuf(size);
            }

            while (_size < size)
            {
                _buf[_size++] = default!;
            }
        }

        public void Resize(int size, T value)
        {
            while (_size > size)
            {
                _buf[--_size] = default!;
            }

            if (size > _capacity)
            {
                ResizeBuf(size);
            }

            while (_size < size)
            {
                _buf[_size++] = value;
            }
        }

        public void Reserve(int size)
        {
            if (size > _capacity)
            {
                ResizeBuf(size);
            }
        }
    }

    //
    // Fixed unit of Directed Acyclic Word Graph (DAWG).
    //

    //
    // Node of Directed Acyclic Word Graph (DAWG).
    //
    internal struct DawgNode
    {
        public DawgNode() { }

        public uint Child { get; set; }
        public uint Sibling { get; set; }
        public bool IsState { get; set; }
        public bool HasSibling { get; set; }
        public byte Label { get; set; }
        public int Value { get => (int)Child; set => Child = (uint)value; }

        public uint Unit
        {
            get
            {
                if (Label == 0)
                {
                    return (Child << 1) | (uint)(HasSibling ? 1 : 0);
                }
                return (Child << 2) | (uint)(IsState ? 2 : 0) | (uint)(HasSibling ? 1 : 0);
            }
        }
    }

    internal struct DawgUnit
    {
        private readonly uint _unit;
        public DawgUnit(uint unit = 0) => _unit = unit;
        public DawgUnit(DawgUnit unit) => _unit = unit._unit;

        public static implicit operator DawgUnit(uint unit) => new DawgUnit(unit);

        public uint Unit => _unit;

        public uint Child => _unit >> 2;

        public bool HasSibling => (_unit & 1) == 1;

        public int Value => (int)(_unit >> 1);

        public bool IsState => (_unit & 2) == 2;
    }

    //
    // Directed Acyclic Word Graph (DAWG) builder.
    //

    public class DawgBuilder
    {
        private const int InitialTableSize = 1 << 10;

        private readonly AutoPool<DawgNode> _nodes = new();
        private readonly AutoPool<DawgUnit> _units = new();
        private readonly AutoPool<byte> _labels = new();
        private readonly AutoPool<uint> _table = new();
        private readonly BitVector _isIntersections = new();
        private readonly Stack<uint> _nodeStack = new();
        private readonly Stack<uint> _recycleBin = new();
        private int _numStates;

        public DawgBuilder() { }

        public uint Root => 0;

        public uint Child(uint id) => _units[(int)id].Child;

        public uint Sibling(uint id) => _units[(int)id].HasSibling ? id + 1 : 0;

        public int Value(uint id) => _units[(int)id].Value;
        public byte Label(uint id) => _labels[(int)id];

        public bool IsLeaf(uint id) => Label(id) == 0;

        public bool IsIntersection(uint id) => _isIntersections[(int)id];

        public uint IntersectionId(uint id) => _isIntersections.Rank(id) - 1;

        public int NumIntersections => (int)_isIntersections.NumOnes;

        public int Size => _units.Size;

        private static uint Hash(uint key)
        {
            key = ~key + (key << 15);  // key = (key << 15) - key - 1;
            key = key ^ (key >> 12);
            key = key + (key << 2);
            key = key ^ (key >> 4);
            key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
            key = key ^ (key >> 16);
            return key;
        }

        private void FreeNode(uint id) => _recycleBin.Push(id);

        public void Finish()
        {
            Flush(0);

            _units[0] = _nodes[0].Unit;
            _labels[0] = _nodes[0].Label;
            _isIntersections.Build();
        }

        public void Insert(ReadOnlySpan<byte> key, int length, int value)
        {
            if (value < 0)
            {
                throw new ArgumentException("failed to insert key: negative value");
            }
            else if (length == 0)
            {
                throw new ArgumentException("failed to insert key: zero-length key");
            }

            uint id = 0;
            int keyPos = 0;

            for (; keyPos <= length; ++keyPos)
            {
                uint childId = _nodes[(int)id].Child;
                if (childId == 0)
                {
                    break;
                }

                byte keyLabel = key[keyPos];
                if (keyPos < length && keyLabel == 0)
                {
                    throw new InvalidOperationException("failed to insert key: invalid null character");
                }

                byte unitLabel = _nodes[(int)childId].Label;
                if (keyLabel < unitLabel)
                {
                    throw new InvalidOperationException("failed to insert key: wrong key order");
                }
                else if (keyLabel > unitLabel)
                {
                    _nodes[(int)childId].HasSibling = true;
                    Flush(childId);
                    break;
                }

                id = childId;
            }

            if (keyPos > length)
            {
                return;
            }

            for (; keyPos <= length; ++keyPos)
            {
                byte keyLabel = keyPos < length ? key[keyPos] : (byte)0;
                uint childId = AppendNode();

                if (_nodes[(int)id].Child == 0)
                {
                    _nodes[(int)childId].IsState = true;
                }

                _nodes[(int)childId].Sibling = _nodes[(int)id].Child;
                _nodes[(int)childId].Label = keyLabel;
                _nodes[(int)id].Child = childId;
                _nodeStack.Push(childId);
                id = childId;
            }
            _nodes[(int)id].Value = value;
        }

        private uint AppendNode()
        {
            uint id;
            if (_recycleBin.Count == 0)
            {
                id = (uint)_nodes.Size;
                _nodes.Append();
            }
            else
            {
                id = _recycleBin.Pop();
                _nodes[(int)id] = new DawgNode();
            }
            return id;
        }

        private uint AppendUnit()
        {
            _isIntersections.Append();
            _units.Append();
            _labels.Append();
            return _isIntersections.Size - 1;
        }

        public void Init()
        {
            _table.Resize(InitialTableSize, 0);

            AppendNode();
            AppendUnit();

            _numStates = 1;

            _nodes[0].Label = 0xFF;
            _nodeStack.Push(0);
        }

        private void ExpandTable()
        {
            int tableSize = _table.Size << 1;
            _table.Clear();
            _table.Resize(tableSize, 0);

            for (int i = 1; i < _units.Size; ++i)
            {
                uint id = (uint)i;
                if (_labels[i] == 0 || _units[i].IsState)
                {
                    FindUnit(id, out uint hashId);
                    _table[(int)hashId] = id;
                }
            }
        }

        private uint HashNode(uint id)
        {
            uint hashValue = 0;
            for (; id != 0; id = _nodes[(int)id].Sibling)
            {
                uint unit = _nodes[(int)id].Unit;
                byte label = _nodes[(int)id].Label;
                hashValue ^= Hash((uint)((label << 24) ^ unit));
            }

            return hashValue;
        }

        private bool AreEqual(uint nodeId, uint unitId)
        {
            for (uint i = _nodes[(int)nodeId].Sibling; i != 0; i = _nodes[(int)i].Sibling)
            {
                if (!_units[(int)unitId].HasSibling)
                {
                    return false;
                }

                ++unitId;
            }

            if (_units[(int)unitId].HasSibling)
            {
                return false;
            }

            for (uint i = nodeId; i != 0; i = _nodes[(int)i].Sibling, --unitId)
            {
                if (_nodes[(int)i].Unit != _units[(int)unitId].Unit || _nodes[(int)i].Label != _labels[(int)unitId])
                {
                    return false;
                }
            }

            return true;
        }

        private uint FindNode(uint nodeId, out uint hashId)
        {
            hashId = (uint)(HashNode(nodeId) % _table.Size);
            for (; ; hashId = (uint)((hashId + 1) % _table.Size))
            {
                uint unitId = _table[(int)hashId];
                if (unitId == 0)
                {
                    break;
                }

                if (AreEqual(nodeId, unitId))
                {
                    return unitId;
                }
            }

            return 0;
        }

        private uint HashUnit(uint id)
        {
            uint hashValue = 0;
            for (; id != 0; ++id)
            {
                uint unit = _units[(int)id].Unit;
                byte label = _labels[(int)id];
                hashValue ^= Hash((uint)((label << 24) ^ unit));

                if (!_units[(int)id].HasSibling)
                {
                    break;
                }
            }
            return hashValue;
        }
        private uint FindUnit(uint id, out uint hashId)
        {
            hashId = (uint)(HashUnit(id) % _table.Size);
            for (; ; hashId = (uint)((hashId + 1) % _table.Size))
            {
                uint unitId = _table[(int)hashId];
                if (unitId == 0)
                {
                    break;
                }

                // There must not be the same unit.
            }
            return 0;
        }

        private void Flush(uint id)
        {
            while (_nodeStack.Peek() != id)
            {
                uint nodeId = _nodeStack.Pop();

                if (_numStates >= _table.Size - (_table.Size >> 2))
                {
                    ExpandTable();
                }

                uint numSiblings = 0;
                for (uint i = nodeId; i != 0; i = _nodes[(int)i].Sibling)
                {
                    ++numSiblings;
                }

                uint matchId = FindNode(nodeId, out uint hashId);
                if (matchId != 0)
                {
                    _isIntersections.Set(matchId, true);
                }
                else
                {
                    uint unitId = 0;
                    for (uint i = 0; i < numSiblings; ++i)
                    {
                        unitId = AppendUnit();
                    }

                    for (uint i = nodeId; i != 0; i = _nodes[(int)i].Sibling)
                    {
                        _units[(int)unitId] = _nodes[(int)i].Unit;
                        _labels[(int)unitId] = _nodes[(int)i].Label;
                        --unitId;
                    }

                    matchId = unitId + 1;
                    _table[(int)hashId] = matchId;
                    ++_numStates;
                }

                for (uint i = nodeId, next; i != 0; i = next)
                {
                    next = _nodes[(int)i].Sibling;
                    FreeNode(i);
                }

                _nodes[(int)_nodeStack!.Peek()].Child = matchId;
            }
            _nodeStack.Pop();
        }
    }

    internal struct DoubleArrayUnit
    {
        private uint _unit;
        public DoubleArrayUnit() { }

        // returns whether a leaf unit is immediately derived from the unit (true) or not (false).
        public bool HasLeaf
        {
            get => ((_unit >> 8) & 1) == 1;
            set
            {
                if (value)
                {
                    _unit |= 1U << 8;
                }
                else
                {
                    _unit &= ~(1U << 8);
                }
            }
        }

        // value() returns the value stored in the unit, and thus value() is
        // available when and only when the unit is a leaf unit.
        public uint Value
        {
            get => _unit & ((1U << 31) - 1);
            set => _unit = value | (1U << 31);
        }

        // returns the label associated with the unit. Note that a leaf unit always returns an invalid label.
        // For this feature, leaf unit's label returns an id that has the MSB of 1.
        public uint Label
        {
            get => _unit & ((1U << 31) | 0xFF);
            set
            {
                _unit = (_unit & ~0xFFU) | value;
            }
        }

        // offset() returns the offset from the unit to its derived units.
        public uint Offset
        {
            get => (_unit >> 10) << (int)((_unit & (1U << 9)) >> 6);
            set
            {
                if (value >= 1U << 29)
                {
                    throw new InvalidOperationException("failed to modify unit: too large offset");
                }

                _unit &= (1U << 31) | (1U << 8) | 0xFF;

                if (value < 1U << 21)
                {
                    _unit |= value << 10;
                }
                else
                {
                    _unit |= (value << 2) | (1U << 9);
                }
            }
        }

    }

    //
    // Extra unit of double-array builder.
    //

    internal struct DoubleArrayBuilderExtraUnit
    {

        private uint _prev;
        private uint _next;
        private bool _isFixed;
        private bool _isUsed;

        public DoubleArrayBuilderExtraUnit() { }

        public uint Prev
        {
            get => _prev;
            set => _prev = value;
        }

        public uint Next
        {
            get => _next;
            set => _next = value;
        }

        public bool IsFixed
        {
            get => _isFixed;
            set => _isFixed = value;
        }

        public bool IsUsed
        {
            get => _isUsed;
            set => _isUsed = value;
        }
    }

    internal class DoubleArrayBuilder
    {
        private const int BlockSize = 256;
        private const int NumExtraBlock = 16;
        private const int NumExtras = BlockSize * NumExtraBlock;
        private const int UpperMask = 0xFF << 21;
        private const int LowerMask = 0xFF;

        private readonly AutoPool<DoubleArrayUnit> _units = new();
        private readonly DoubleArrayBuilderExtraUnit[] _extras = new DoubleArrayBuilderExtraUnit[NumExtras];
        private readonly AutoPool<byte> _labels = new();
        private uint[]? _table;
        private uint _extrasHead;

        private int NumBlocks() => _units.Size / BlockSize;

        public DoubleArrayUnit[] Units => _units.Buffer;
        public int UnitsSize => _units.Size;

        private ref DoubleArrayBuilderExtraUnit this[uint id]
        {
            get => ref _extras[id % NumExtras];
        }

        internal unsafe void BuildDawg(SortedDictionary<string, int> dictionary, DawgBuilder dawgBuilder)
        {
            dawgBuilder.Init();

            Span<byte> bytes = stackalloc byte[512];
            byte[]? array = null;

            foreach (KeyValuePair<string, int> pair in dictionary)
            {
                int encodingLength = Encoding.UTF8.GetByteCount(pair.Key);
                if (encodingLength > bytes.Length)
                {
                    if (array is not null)
                    {
                        ArrayPool<byte>.Shared.Return(array);
                    }

                    array = ArrayPool<byte>.Shared.Rent(encodingLength * 2);
                    bytes = array;
                }

                encodingLength = Helpers.EncodeToUtf8(pair.Key.AsSpan(), bytes);

                dawgBuilder.Insert(bytes, encodingLength, pair.Value);
            }

            if (array is not null)
            {
                ArrayPool<byte>.Shared.Return(array);
            }

            dawgBuilder.Finish();
        }

        internal void FixBlock(uint blockId)
        {
            uint begin = blockId * BlockSize;
            uint end = begin + BlockSize;

            uint unusedOffset = 0;
            for (uint offset = begin; offset != end; ++offset)
            {
                if (!this[offset].IsUsed)
                {
                    unusedOffset = offset;
                    break;
                }
            }

            for (uint id = begin; id != end; ++id)
            {
                if (!this[id].IsFixed)
                {
                    ReserveId(id);
                    _units[(int)id].Label = (byte)(id ^ unusedOffset);
                }
            }
        }

        internal void ExpandUnits()
        {
            uint srcNumUnits = (uint)_units.Size;
            uint srcNumBlocks = (uint)NumBlocks();

            uint destNumUnits = srcNumUnits + BlockSize;
            uint destNumBlocks = srcNumBlocks + 1;

            if (destNumBlocks > NumExtraBlock)
            {
                FixBlock(srcNumBlocks - NumExtraBlock);
            }

            _units.Resize((int)destNumUnits);

            if (destNumBlocks > NumExtraBlock)
            {
                for (uint id = srcNumUnits; id < destNumUnits; ++id)
                {
                    this[id].IsUsed = false;
                    this[id].IsFixed = false;
                }
            }

            for (uint i = srcNumUnits + 1; i < destNumUnits; ++i)
            {
                this[i - 1].Next = i;
                this[i].Prev = i - 1;
            }

            this[srcNumUnits].Prev = destNumUnits - 1;
            this[destNumUnits - 1].Next = srcNumUnits;
            this[srcNumUnits].Prev = this[_extrasHead].Prev;
            this[destNumUnits - 1].Next = _extrasHead;
            this[this[_extrasHead].Prev].Next = srcNumUnits;
            this[_extrasHead].Prev = destNumUnits - 1;
        }

        internal void ReserveId(uint id)
        {
            if (id >= _units.Size)
            {
                ExpandUnits();
            }

            if (id == _extrasHead)
            {
                _extrasHead = this[id].Next;
                if (_extrasHead == id)
                {
                    _extrasHead = (uint)_units.Size;
                }
            }

            this[this[id].Prev].Next = this[id].Next;
            this[this[id].Next].Prev = this[id].Prev;
            this[id].IsFixed = true;
        }

        internal bool IsValidOffset(uint id, uint offset)
        {
            if (this[offset].IsUsed)
            {
                return false;
            }

            uint relOffset = id ^ offset;
            if ((relOffset & LowerMask) != 0 && (relOffset & UpperMask) != 0)
            {
                return false;
            }

            for (int i = 1; i < _labels.Size; ++i)
            {
                if (this[offset ^ _labels[i]].IsFixed)
                {
                    return false;
                }
            }

            return true;
        }

        internal uint FindValidOffset(uint id)
        {
            if (_extrasHead >= _units.Size)
            {
                return (uint)_units.Size | (id & LowerMask);
            }

            uint unfixedId = _extrasHead;
            do
            {
                uint offset = unfixedId ^ _labels[0];
                if (IsValidOffset(id, offset))
                {
                    return offset;
                }

                unfixedId = this[unfixedId].Next;
            } while (unfixedId != _extrasHead);

            return (uint)_units.Size | (id & LowerMask);
        }

        internal uint ArrangeFromDawg(DawgBuilder dawg, uint dawgId, uint dicId)
        {
            _labels.Resize(0);

            uint dawgChildId = dawg.Child(dawgId);
            while (dawgChildId != 0)
            {
                _labels.Append(dawg.Label(dawgChildId));
                dawgChildId = dawg.Sibling(dawgChildId);
            }

            uint offset = FindValidOffset(dicId);
            _units[(int)dicId].Offset = dicId ^ offset;

            dawgChildId = dawg.Child(dawgId);
            for (int i = 0; i < _labels.Size; ++i)
            {
                uint dicChildId = offset ^ _labels[i];
                ReserveId(dicChildId);

                if (dawg.IsLeaf(dawgChildId))
                {
                    _units[(int)dicId].HasLeaf = true;
                    _units[(int)dicChildId].Value = (uint)dawg.Value(dawgChildId);
                }
                else
                {
                    _units[(int)dicChildId].Label = _labels[i];
                }

                dawgChildId = dawg.Sibling(dawgChildId);
            }

            this[offset].IsUsed = true;

            return offset;
        }

        internal void BuildFromDawg(DawgBuilder dawg, uint dawgId, uint dicId)
        {
            uint dawgChildId = dawg.Child(dawgId);
            uint offset;
            if (dawg.IsIntersection(dawgChildId))
            {
                uint intersectionId = dawg.IntersectionId(dawgChildId);
                offset = _table![intersectionId];
                if (offset != 0)
                {
                    offset ^= dicId;
                    if ((offset & UpperMask) == 0 || (offset & LowerMask) == 0)
                    {
                        if (dawg.IsLeaf(dawgChildId))
                        {
                            _units[(int)dicId].HasLeaf = true;
                        }
                        _units[(int)dicId].Offset = offset;
                        return;
                    }
                }
            }

            offset = ArrangeFromDawg(dawg, dawgId, dicId);
            if (dawg.IsIntersection(dawgChildId))
            {
                _table![dawg.IntersectionId(dawgChildId)] = offset;
            }

            do
            {
                byte childLabel = dawg.Label(dawgChildId);
                uint dicChildId = offset ^ childLabel;
                if (childLabel != 0)
                {
                    BuildFromDawg(dawg, dawgChildId, dicChildId);
                }

                dawgChildId = dawg.Sibling(dawgChildId);
            } while (dawgChildId != 0);
        }

        internal void FixAllBlocks()
        {
            uint begin = 0;
            if (NumBlocks() > NumExtraBlock)
            {
                begin = (uint)NumBlocks() - NumExtraBlock;
            }

            uint end = (uint)NumBlocks();

            for (uint blockId = begin; blockId != end; ++blockId)
            {
                FixBlock(blockId);
            }
        }

        internal void BuildFromDawg(DawgBuilder dawg)
        {
            int numUnits = 1;
            while (numUnits < dawg.Size)
            {
                numUnits <<= 1;
            }

            _units.Reserve(numUnits);
            _table = new uint[dawg.NumIntersections];

            ReserveId(0);

            this[0].IsUsed = true;
            _units[0].Offset = 1;
            _units[0].Label = 0;

            if (dawg.Child(dawg.Root) != 0)
            {
                BuildFromDawg(dawg, dawg.Root, 0);
            }

            FixAllBlocks();
        }

        public void Build(SortedDictionary<string, int> dictionary)
        {
            DawgBuilder dawgBuilder = new();
            BuildDawg(dictionary, dawgBuilder);
            BuildFromDawg(dawgBuilder);
        }
    }

    internal struct DoubleArrayResultPair
    {
        public int Value { get; set; }
        public int Length { get; set; }
    };

    internal class DoubleArrayTrie
    {
        private readonly int _size;
        private readonly DoubleArrayUnit[] _array;

        internal DoubleArrayUnit[] ArrayUnits => _array;
        internal int Size => _size;

        // Sorted Dictionary to store the key value pairs
        public DoubleArrayTrie(SortedDictionary<string, int> dictionary)
        {
            DoubleArrayBuilder builder = new DoubleArrayBuilder();
            builder.Build(dictionary);

            _size = builder.UnitsSize;
            _array = builder.Units;
        }

        public DoubleArrayTrie(DoubleArrayUnit[] preCompiledData)
        {
            if (preCompiledData is null)
            {
                throw new ArgumentNullException(nameof(preCompiledData));
            }

            _size = preCompiledData.Length;
            _array = preCompiledData;
        }

        public int CommonPrefixSearch(ReadOnlySpan<byte> key, Span<DoubleArrayResultPair> results, int nodePos = 0)
        {
            int numResults = 0;

            DoubleArrayUnit unit = _array[nodePos];
            nodePos ^= (int)unit.Offset;

            for (int i = 0; i < key.Length; ++i)
            {
                nodePos ^= key[i];
                unit = _array[nodePos];

                if (unit.Label != key[i])
                {
                    return numResults;
                }

                nodePos ^= (int)unit.Offset;

                if (unit.HasLeaf)
                {
                    if (numResults < results.Length)
                    {
                        results[numResults].Value = (int)_array[nodePos].Value;
                        results[numResults].Length = i + 1;
                    }

                    ++numResults;
                }
            }

            return numResults;
        }

        public int Traverse(ReadOnlySpan<byte> key, ref int nodePos, ref int keyPos, int length)
        {
            uint id = (uint)nodePos;
            DoubleArrayUnit unit = _array[id];

            if (length != 0)
            {
                for (; keyPos < length; ++keyPos)
                {
                    id ^= unit.Offset ^ key[keyPos];
                    unit = _array[id];
                    if (unit.Label != key[keyPos])
                    {
                        return -2;
                    }

                    nodePos = (int)id;
                }
            }
            else
            {
                for (; key[keyPos] != 0; ++keyPos)
                {
                    id ^= unit.Offset ^ key[keyPos];
                    unit = _array[id];
                    if (unit.Label != key[keyPos])
                    {
                        return -2;
                    }

                    nodePos = (int)id;
                }
            }

            if (!unit.HasLeaf)
            {
                return -1;
            }

            unit = _array[id ^ unit.Offset];
            return (int)unit.Value;
        }
    }
}

