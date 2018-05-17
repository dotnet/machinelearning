# IDV File Format

This document describes ML.NET's Binary dataview file format, version 1.1.1.5
written by the `BinarySaver` and `BinaryLoader` classes, commonly known as the
`.idv` format.

## Goal of the Format

A dataview is a collection of columns, over some number of rows. (Do not
confuse column with features. Columns can be and often are vector valued, and
it is expected though not required that commonly all features will be together
in one vector valued column.)

The actual values are stored in blocks. A block holds values for a single
column across multiple rows. Block format is dictated by a codec. There is a
table-of-contents and lookup table to facilitate quasi-random access to
particular blocks. (Quasi in the sense that you can only seek to a block, not
to a particular within a block.)

## General Data Format

Before we discuss the format itself we will establish some conventions on how
individual scalar values, strings, and other data is serialized. All basic
pieces of data (e.g., a single number, or a single string) are encoded in ways
reflecting the semantics of the .NET `BinaryWriter` class, those semantics
being:

* All numbers are stored as little-endian, using their natural fix-length
  binary encoding.

* Strings are stored using an unsigned LEB128 number describing the number of
  bytes, followed by that many bytes containing the UTF-8 encoded string.

A note about this: [LEB128](https://en.wikipedia.org/wiki/LEB128) is a simple
encoding to encode arbitrarily large integers. Each byte of 8-bits follows
this convention. The most significant bit is 0 if and only if this is the end
of the LEB128 encoding. The remaining 7 bits are a part of the number being
encoded. The bytes are stored little-endian, that is, the first byte holds the
7 least significant bits, the second byte (if applicable) holds the next 7
least significant bits, etc., and the last byte holds the 7 most significant
bits. LEB128 is used one or two places in this format. (I might tend to prefer
use of LEB128 in places where we are writing values that, on balance, we
expect to be relatively small, and only in cases where there is no potential
for benefit for random access to the associated stream, since LEB128 is
incompatible with random access. However, this is not formulated into anything
approaching a definite policy.)

## Header

Every binary instances stream has a header composed of 256 bytes, at the start
of the stream. Not all bytes are used. Those bytes that are not explicitly
used have undefined content, and can have anything in them. We strongly
encourage writers of this format to insert obscene messages in this dead
space. The content is defined as follows (the offsets being the start of that
column).

Offsets | Type  | Name and Description
--------|-------|---------------------
0       | ulong | **Signature**: The magic number of this file.
8       | ulong | **Version**: Indicates the version of the data file.
16      | ulong | **CompatibleVersion**: Indicates the minimum reader version that can interpret this file, possibly with some data loss.
24      | long  | **TableOfContentsOffset**: The offset to the column table of contents structure.
32      | long  | **TailOffset**: The eight-byte tail signature starts at this offset. So, the entire dataset stream should be considered to have byte length of eight plus this value.
40      | long  | **RowCount**: The number of rows in this data file.
48      | int   | **ColumnCount**: The number of columns in this data file.

Notes on these:

* The signature of this file is `0x00425644004C4D43`, which is, when written
  little-endian to a file, `CML DVB ` with null characters in the place of
  spaces. These letters are intended  to suggest "CloudML DataView Binary."

* The tail signature is the byte-reversed version of this, that is,
  `0x434D4C0044564200`.

* Versions are encoded as four 16-bit unsigned numbers passed into a single
  ulong, with higher order bits being a more major version. The first
  supported version of the is 1.1.1.4, that is, `0x0001000100010004`.
  (Versions prior to 1.1.1.4 did exist, but were not released, so we do not
  support them, though we do describe them in this document for the sake of
  completeness.)
 
## Table of Contents Format

The table of contents are packed entries, with there being as many entries as
there are columns. The version field here indicates the versions where that
entry is written. ≥ indicates the field occurred in versions after and
including that version, = indicates the field occurs only in that version.

Description | Entry Type | Version
------------|------------|--------
Column name | string     | ≥1.1.1.1
Codec loadname | string  | ≥1.1.1.1
Codec parameterization length | LEB128 integer | ≥1.1.1.1
Codec parameterization, which must have precisely the length indicated above | arbitrary, but with specified length | ≥1.1.1.1
Compression kind | CompressionKind (byte) | ≥1.1.1.1
Rows per block in this column | LEB128 integer | ≥1.1.1.1
Lookup table offset | long | ≥1.1.1.1
Slot names offset, or 0 if this column has no slot names, if 1.1.1.2 behave as if there are no slot names, with this having value 0) | long | =1.1.1.3
Slot names byte size (present only if slot names offset is greater than 0) | long | =1.1.1.3
Slot names count (present only if slot names offset is greater than 0) | int | =1.1.1.3
Metadata table of contents offset, or 0 if there is no metadata (1.1.1.4) | long | ≥1.1.1.4

For those working in the ML.NET codebase: The three `Codec` fields are handled
by the `CodecFactory.WriteCodec/TryReadCodec` methods, with the definition
stream being at the start of the codec loadname, and being at the end of the
codec parameterization, both in the case of success or failure.

CompressionCodec enums are described below, and describe the compression
algorithm used to compress blocks.

### Compression Kind

The enum for compression kind is one byte, and follows this scheme:

Compression Kind                                               | Code
---------------------------------------------------------------|-----
None                                                           | 0
DEFLATE (i.e., [RFC1951](http://www.ietf.org/rfc/rfc1951.txt)) | 1
zlib (i.e., [RFC1950](http://www.ietf.org/rfc/rfc1950.txt))    | 2

None means no compression. DEFLATE is the default scheme. There is a tendency
to conflate zlib and DEFLATE, so to be clear: zlib can be (somewhat inexactly)
considered a wrapped version of DEFLATE, but it is still a distinct (but
closely related) format. However, both are implemented by the zlib library,
which is probably the source of the confusion.

## Metadata Table of Contents Format

The metadata table of contents begins with a LEB128 integer describing the
number of entries. (Should be a positive value, since if a column has no
metadata the expectation is that the offset for the metadata TOC will be
stored as 0.) What follows that are that many packed entries. Each entry is
somewhat akin to the column table of contents entry, with some simplifications
considering that there will be exactly one "block" with one item.

Description                                            | Entry Type
-------------------------------------------------------|------------
Metadata kind                                          | string
Codec loadname                                         | string
Codec parameterization length                          | LEB128 integer
Codec parameterization, which must have precisely the length indicated above | arbitrary, but with specified length
Compression kind                                       | CompressionKind(byte)
Offset of the block where the metadata item is written | long
Byte length of the block                               | LEB128 integer

The "block" written is written in exactly same format as the main content
blocks. This will be very slightly inefficient as that scheme is sometimes
written to accommodate many entries, but I don't expect that to be much of a
burden.

## Lookup Table Format

Each table of contents entry is associated with a lookup table starting at the
indicated lookup table offset. It is written as packed binary, with each
lookup entry consisting of 16 bytes. So in all, the lookup table takes 16
bytes, times the total number of blocks for this column.

Description                                               | Entry Type
----------------------------------------------------------|-----------
Block offset, position in the file where the block starts | long
Block length, its size in bytes in the file               | int
Uncompressed block length, its size in bytes if the block bytes were decompressed according to the column's compression codec | int

## Slot Names

If slot names are stored, they are stored as pairs of integer index/string
pairs. As many pairs are stored as count of slot names were present in the
table of contents entry. Note that this only appeared in version 1.1.1.3. With
1.1.1.4 and later, slot names were just considered yet another piece of
metadata.

Description       | Entry Type
------------------|-----------
Index of the slot | int
The slot name     | string

## Block Format

Columns are ordered into blocks, with each block holding the binary encoded
values for one particular columns across a range of rows. So for example, if
the column's table of contents describes it as having 1000 rows per block, the
first block will contain the values for the column for rows 0 through 999,
second block 1000 through 1999, etc., with all blocks containing the same
number of blocks, except the last block which will contain fewer items (unless
the number of rows just so happens to be a multiple of the block size).

Each column is a possibly compressed sequence of bytes, compressed according
to the compression type field in the table of contents.  It begins and ends at
the offsets indicated in the metadata entry stored in the directory. The
uncompressed bytes will be stored in the format as described by the codec.
