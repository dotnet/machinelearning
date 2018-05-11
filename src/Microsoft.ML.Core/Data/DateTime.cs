// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;
    using SysDateTime = System.DateTime;
    using SysDateTimeOffset = System.DateTimeOffset;
    using SysTimeSpan = System.TimeSpan;

    /// <summary>
    /// A struct to represent a DateTime column type
    /// </summary>
    public struct DvDateTime : IEquatable<DvDateTime>, IComparable<DvDateTime>
    {
        public const long MaxTicks = 3155378975999999999;
        private readonly DvInt8 _ticks;

        /// <summary>
        /// This ctor initializes _ticks to the value of sdt.Ticks, and ignores its DateTimeKind value.
        /// </summary>
        public DvDateTime(SysDateTime sdt)
        {
            _ticks = sdt.Ticks;
            AssertValid();
        }

        /// <summary>
        /// This ctor accepts any value for ticks, but produces an NA if ticks is out of the legal range.
        /// </summary>
        public DvDateTime(DvInt8 ticks)
        {
            if ((ulong)ticks.RawValue > MaxTicks)
                _ticks = DvInt8.NA;
            else
                _ticks = ticks;
            AssertValid();
        }

        [Conditional("DEBUG")]
        internal void AssertValid()
        {
            Contracts.Assert((ulong)_ticks.RawValue <= MaxTicks || _ticks.IsNA);
        }

        public DvInt8 Ticks
        {
            get
            {
                AssertValid();
                return _ticks;
            }
        }

        // REVIEW: Add more System.DateTime members returning their corresponding 'Dv' types (task 4255).
        /// <summary>
        /// Gets the date component of this object.
        /// </summary>
        public DvDateTime Date
        {
            get
            {
                AssertValid();
                if (IsNA)
                    return NA;
                return new DvDateTime(GetSysDateTime().Date);
            }
        }

        /// <summary>
        /// Gets a DvDateTime object representing the current UTC date and time.
        /// </summary>
        public static DvDateTime UtcNow { get { return new DvDateTime(SysDateTimeOffset.Now.UtcDateTime); } }

        public bool IsNA
        {
            get
            {
                AssertValid();
                return (ulong)_ticks.RawValue > MaxTicks;
            }
        }

        public static DvDateTime NA
        {
            get { return new DvDateTime(DvInt8.NA); }
        }

        public static explicit operator SysDateTime?(DvDateTime dvDt)
        {
            if (dvDt.IsNA)
                return null;
            return dvDt.GetSysDateTime();
        }

        /// <summary>
        /// Creates a new DvDateTime with the same number of ticks as in sdt, ignoring its DateTimeKind value.
        /// </summary>
        public static implicit operator DvDateTime(SysDateTime sdt)
        {
            return new DvDateTime(sdt);
        }

        public static implicit operator DvDateTime(SysDateTime? sdt)
        {
            if (sdt == null)
                return DvDateTime.NA;
            return new DvDateTime(sdt.Value);
        }

        public override string ToString()
        {
            AssertValid();
            if (IsNA)
                return "";
            return GetSysDateTime().ToString("o");
        }

        internal SysDateTime GetSysDateTime()
        {
            AssertValid();
            Contracts.Assert(!IsNA);
            return new SysDateTime(_ticks.RawValue);
        }

        public bool Equals(DvDateTime other)
        {
            return _ticks.RawValue == other._ticks.RawValue;
        }

        public override bool Equals(object obj)
        {
            return obj is DvDateTime && Equals((DvDateTime)obj);
        }

        public int CompareTo(DvDateTime other)
        {
            if (_ticks.RawValue == other._ticks.RawValue)
                return 0;
            return _ticks.RawValue < other._ticks.RawValue ? -1 : 1;
        }

        public override int GetHashCode()
        {
            return _ticks.GetHashCode();
        }
    }

    /// <summary>
    /// A struct to represent a DateTimeZone column type.
    /// </summary>
    public struct DvDateTimeZone : IEquatable<DvDateTimeZone>, IComparable<DvDateTimeZone>
    {
        public const long TicksPerMinute = 600000000;
        public const long MaxMinutesOffset = 840;
        public const long MinMinutesOffset = -840;

        // Stores the UTC date-time (convert to clock time by adding the offset).
        private readonly DvDateTime _dateTime;
        // Store the offset in minutes.
        private readonly DvInt2 _offset;

        // This assumes (and asserts) that the dt/offset combination is valid.
        // Callers should do the validation.
        private DvDateTimeZone(DvDateTime dt, DvInt2 offset)
        {
            _dateTime = dt;
            _offset = offset;
            AssertValid();
        }

        /// <summary>
        /// Given a number of ticks for the date time portion and a number of minutes for
        /// the time zone offset, this constructs a new DvDateTimeZone. If anything is invalid,
        /// it produces NA.
        /// </summary>
        /// <param name="ticks">The number of clock ticks in the date time portion</param>
        /// <param name="offset">The time zone offset in minutes</param>
        public DvDateTimeZone(DvInt8 ticks, DvInt2 offset)
        {
            var dt = new DvDateTime(ticks);
            if (dt.IsNA || offset.IsNA || MinMinutesOffset > offset.RawValue || offset.RawValue > MaxMinutesOffset)
            {
                _dateTime = DvDateTime.NA;
                _offset = DvInt2.NA;
            }
            else
            {
                _offset = offset;
                _dateTime = ValidateDate(dt, ref _offset);
            }
            AssertValid();
        }

        public DvDateTimeZone(SysDateTimeOffset dto)
        {
            // Since it is constructed from a SysDateTimeOffset, all the validations should work.
            var success = TryValidateOffset(dto.Offset.Ticks, out _offset);
            Contracts.Assert(success);
            _dateTime = ValidateDate(new DvDateTime(dto.UtcDateTime), ref _offset);
            Contracts.Assert(!_dateTime.IsNA);
            Contracts.Assert(!_offset.IsNA);
            AssertValid();
        }

        /// <summary>
        /// Constructs a DvDateTimeZone from a clock date-time and a time zone offset from UTC.
        /// </summary>
        /// <param name="dt">The clock time</param>
        /// <param name="offset">The offset</param>
        public DvDateTimeZone(DvDateTime dt, DvTimeSpan offset)
        {
            if (dt.IsNA || offset.IsNA || !TryValidateOffset(offset.Ticks, out _offset))
            {
                _dateTime = DvDateTime.NA;
                _offset = DvInt2.NA;
            }
            else
                _dateTime = ValidateDate(dt, ref _offset);
            AssertValid();
        }

        /// <summary>
        /// This method takes a DvDateTime representing clock time, and a TimeSpan representing an offset,
        /// validates that both the clock time and the UTC time (which is the clock time minus the offset)
        /// are within the valid range, and returns a DvDateTime representing the UTC time (dateTime-offset).
        /// </summary>
        /// <param name="dateTime">The clock time</param>
        /// <param name="offset">The offset. This value is assumed to be validated as a legal offset: 
        /// a value in whole minutes, between -14 and 14 hours.</param>
        /// <returns>The UTC DvDateTime representing the input clock time minus the offset</returns>
        private static DvDateTime ValidateDate(DvDateTime dateTime, ref DvInt2 offset)
        {
            Contracts.Assert(!dateTime.IsNA);
            Contracts.Assert(!offset.IsNA);

            // Validate that both the UTC and clock times are legal.
            Contracts.Assert(MinMinutesOffset <= offset.RawValue && offset.RawValue <= MaxMinutesOffset);
            var offsetTicks = offset.RawValue * TicksPerMinute;
            // This operation cannot overflow because offset should have already been validated to be within
            // 14 hours and the DateTime instance is more than that distance from the boundaries of Int64.
            long utcTicks = dateTime.Ticks.RawValue - offsetTicks;
            var dvdt = new DvDateTime(utcTicks);
            if (dvdt.IsNA)
                offset = DvInt2.NA;
            return dvdt;
        }

        /// <summary>
        /// This method takes a TimeSpan offset, validates that it is a legal offset for DvDateTimeZone (i.e.
        /// in whole minutes, and between -14 and 14 hours), and returns the offset in number of minutes.
        /// </summary>
        /// <param name="offsetTicks"></param>
        /// <param name="offset"></param>
        /// <returns></returns>
        private static bool TryValidateOffset(DvInt8 offsetTicks, out DvInt2 offset)
        {
            if (offsetTicks.IsNA || offsetTicks.RawValue % TicksPerMinute != 0)
            {
                offset = DvInt2.NA;
                return false;
            }

            long mins = offsetTicks.RawValue / TicksPerMinute;
            short res = (short)mins;
            if (res != mins || res > MaxMinutesOffset || res < MinMinutesOffset)
            {
                offset = DvInt2.NA;
                return false;
            }
            offset = res;
            Contracts.Assert(!offset.IsNA);
            return true;
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            _dateTime.AssertValid();
            if (_dateTime.IsNA)
                Contracts.Assert(_offset.IsNA);
            else
            {
                Contracts.Assert(MinMinutesOffset <= _offset.RawValue && _offset.RawValue <= MaxMinutesOffset);
                Contracts.Assert((ulong)(_dateTime.Ticks.RawValue + _offset.RawValue * TicksPerMinute)
                    <= (ulong)DvDateTime.MaxTicks);
            }
        }

        public DvDateTime ClockDateTime
        {
            get
            {
                AssertValid();
                if (_dateTime.IsNA)
                    return DvDateTime.NA;
                var res = new DvDateTime(_dateTime.Ticks.RawValue + _offset.RawValue * TicksPerMinute);
                Contracts.Assert(!res.IsNA);
                return res;
            }
        }

        /// <summary>
        /// Gets the UTC date and time.
        /// </summary>
        public DvDateTime UtcDateTime
        {
            get
            {
                AssertValid();
                if (IsNA)
                    return DvDateTime.NA;
                return _dateTime;
            }
        }

        /// <summary>
        /// Gets the offset as a time span.
        /// </summary>
        public DvTimeSpan Offset
        {
            get
            {
                AssertValid();
                if (_offset.IsNA)
                    return DvTimeSpan.NA;
                return new DvTimeSpan(_offset.RawValue * TicksPerMinute);
            }
        }

        /// <summary>
        /// Gets the offset in minutes.
        /// </summary>
        public DvInt2 OffsetMinutes
        {
            get
            {
                AssertValid();
                return _offset;
            }
        }

        // REVIEW: Add more System.DateTimeOffset members returning their corresponding 'Dv' types (task 4255).

        /// <summary>
        /// Gets the date component of the ClockDateTime.
        /// </summary>
        public DvDateTime ClockDate
        {
            get
            {
                AssertValid();
                if (IsNA)
                    return DvDateTime.NA;
                return ClockDateTime.Date;
            }
        }

        /// <summary>
        /// Gets the date component of the UtcDateTime.
        /// </summary>
        public DvDateTime UtcDate
        {
            get
            {
                AssertValid();
                if (IsNA)
                    return DvDateTime.NA;
                return _dateTime.Date;
            }
        }

        /// <summary>
        /// Gets a DvDateTimeZone object representing the current UTC date and time (with offset=0).
        /// </summary>
        public static DvDateTimeZone UtcNow { get { return new DvDateTimeZone(SysDateTimeOffset.UtcNow); } }

        public bool IsNA
        {
            get
            {
                AssertValid();
                return _dateTime.IsNA;
            }
        }

        // The missing value for DvDateTimeZone is represented by a DvDateTimeZone with _dateTime = DvDateTime.NA
        // and _offset = 0.
        public static DvDateTimeZone NA
        {
            get { return new DvDateTimeZone(DvDateTime.NA, DvInt2.NA); }
        }

        public static explicit operator SysDateTimeOffset?(DvDateTimeZone dvDto)
        {
            if (dvDto.IsNA)
                return null;
            return dvDto.GetSysDateTimeOffset();
        }

        public static implicit operator DvDateTimeZone(SysDateTimeOffset sdto)
        {
            return new DvDateTimeZone(sdto);
        }

        public static implicit operator DvDateTimeZone(SysDateTimeOffset? sdto)
        {
            if (sdto == null)
                return DvDateTimeZone.NA;
            return new DvDateTimeZone(sdto.Value);
        }

        public override string ToString()
        {
            AssertValid();
            if (IsNA)
                return "";

            return GetSysDateTimeOffset().ToString("o");
        }

        private DateTimeOffset GetSysDateTimeOffset()
        {
            AssertValid();
            Contracts.Assert(!IsNA);
            return new SysDateTimeOffset(ClockDateTime.GetSysDateTime(), new TimeSpan(0, _offset.RawValue, 0));
        }

        /// <summary>
        /// Compare two values for equality. Note that this differs from System.DateTimeOffset's
        /// definition of Equals, which only compares the UTC values, not the offsets.
        /// </summary>
        public bool Equals(DvDateTimeZone other)
        {
            return _offset.RawValue == other._offset.RawValue && _dateTime.Equals(other._dateTime);
        }

        public override bool Equals(object obj)
        {
            return obj is DvDateTimeZone && Equals((DvDateTimeZone)obj);
        }

        /// <summary>
        /// Compare two values for ordering. Note that this differs from System.DateTimeOffset's
        /// definition of CompareTo, which only compares the UTC values, not the offsets.
        /// </summary>
        public int CompareTo(DvDateTimeZone other)
        {
            AssertValid();
            other.AssertValid();

            int res = _dateTime.CompareTo(other._dateTime);
            if (res != 0)
                return res;
            if (_offset.RawValue == other._offset.RawValue)
                return 0;
            return _offset.RawValue < other._offset.RawValue ? -1 : 1;
        }

        public override int GetHashCode()
        {
            return Hashing.CombineHash(_dateTime.GetHashCode(), _offset.GetHashCode());
        }
    }

    /// <summary>
    /// A struct to represent a DateTime column type
    /// </summary>
    public struct DvTimeSpan : IEquatable<DvTimeSpan>, IComparable<DvTimeSpan>
    {
        private readonly DvInt8 _ticks;

        public DvInt8 Ticks { get { return _ticks; } }

        public DvTimeSpan(DvInt8 ticks)
        {
            _ticks = ticks;
        }

        public DvTimeSpan(SysTimeSpan sts)
        {
            _ticks = sts.Ticks;
        }

        public DvTimeSpan(SysTimeSpan? sts)
        {
            _ticks = sts != null ? sts.GetValueOrDefault().Ticks : DvInt8.NA;
        }

        public bool IsNA
        {
            get { return _ticks.IsNA; }
        }

        public static DvTimeSpan NA
        {
            get { return new DvTimeSpan(DvInt8.NA); }
        }

        public static explicit operator SysTimeSpan?(DvTimeSpan ts)
        {
            if (ts.IsNA)
                return null;
            return new SysTimeSpan(ts._ticks.RawValue);
        }

        public static implicit operator DvTimeSpan(SysTimeSpan sts)
        {
            return new DvTimeSpan(sts);
        }

        public static implicit operator DvTimeSpan(SysTimeSpan? sts)
        {
            return new DvTimeSpan(sts);
        }

        public override string ToString()
        {
            if (IsNA)
                return "";
            return new SysTimeSpan(_ticks.RawValue).ToString("c");
        }

        public bool Equals(DvTimeSpan other)
        {
            return _ticks.RawValue == other._ticks.RawValue;
        }

        public override bool Equals(object obj)
        {
            return obj is DvTimeSpan && Equals((DvTimeSpan)obj);
        }

        public int CompareTo(DvTimeSpan other)
        {
            if (_ticks.RawValue == other._ticks.RawValue)
                return 0;
            return _ticks.RawValue < other._ticks.RawValue ? -1 : 1;
        }

        public override int GetHashCode()
        {
            return _ticks.GetHashCode();
        }
    }
}
