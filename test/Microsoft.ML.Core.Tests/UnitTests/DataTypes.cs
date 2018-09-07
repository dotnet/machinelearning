using System;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public class DataTypesTest : TestDataViewBase
    {
        public DataTypesTest(ITestOutputHelper helper)
           : base(helper)
        {
        }

        private readonly static Conversions _conv = Conversions.Instance;

        [Fact]
        public void TXToSByte()
        {
            var mapper = GetMapper<DvText, sbyte>();

            Assert.NotNull(mapper);

            //1. sbyte.MinValue in text to sbyte.
            sbyte minValue = sbyte.MinValue;
            sbyte maxValue = sbyte.MaxValue;
            DvText src = new DvText(minValue.ToString());
            sbyte dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, minValue);

            //2. sbyte.MaxValue in text to sbyte.
            src = new DvText(maxValue.ToString());
            dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, maxValue);

            //3. ERROR condition: sbyte.MinValue - 1 in text to sbyte.
            src = new DvText((sbyte.MinValue - 1).ToString());
            dst = 0;
            bool error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch(Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to sbyte.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //4. ERROR condition: sbyte.MaxValue + 1 in text to sbyte.
            src = new DvText((sbyte.MaxValue + 1).ToString());
            dst = 0;
            error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch(Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to sbyte.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //5. Empty string in text to sbyte.
            src = default;
            dst = -1;
            mapper(ref src, ref dst);
            Assert.Equal(default, dst);

            //6. Missing value in text to sbyte.
            src = DvText.NA;
            dst = -1;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Missing text value cannot be converted to integer type.", ex.Message);
                error = true;
            }

            Assert.True(error);
        }

        [Fact]
        public void TXToShort()
        {
            var mapper = GetMapper<DvText, short>();

            Assert.NotNull(mapper);

            //1. short.MinValue in text to short.
            short minValue = short.MinValue;
            short maxValue = short.MaxValue;
            DvText src = new DvText(minValue.ToString());
            short dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, minValue);

            //2. short.MaxValue in text to short.
            src = new DvText(maxValue.ToString());
            dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, maxValue);

            //3. ERROR condition: short.MinValue - 1 in text to short.
            src = new DvText((minValue - 1).ToString());
            dst = 0;
            bool error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch(Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to short.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //4. ERROR condition: short.MaxValue + 1 in text to short.
            src = new DvText((maxValue + 1).ToString());
            dst = 0;
            error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to short.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //5. Empty value in text to short.
            src = default;
            dst = -1;
            mapper(ref src, ref dst);
            Assert.Equal(default, dst);

            //6. Missing string in text to sbyte.
            src = DvText.NA;
            dst = -1;
            error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Missing text value cannot be converted to integer type.", ex.Message);
                error = true;
            }

            Assert.True(error);
        }

        [Fact]
        public void TXToInt()
        {
            var mapper = GetMapper<DvText, int>();

            Assert.NotNull(mapper);

            //1. int.MinValue in text to int.
            int minValue = int.MinValue;
            int maxValue = int.MaxValue;
            DvText src = new DvText(minValue.ToString());
            int dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, minValue);

            //2. int.MaxValue in text to int.
            src = new DvText(maxValue.ToString());
            dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, maxValue);

            //3. ERROR condition: int.MinValue - 1 in text to int.
            src = new DvText(((long)minValue - 1).ToString());
            dst = 0;
            bool error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to int.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //4. ERROR condition: int.MaxValue + 1 in text to int.
            src = new DvText(((long)maxValue + 1).ToString());
            dst = 0;
            error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to int.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //5. Empty value in text to int.
            src = default;
            dst = -1;
            mapper(ref src, ref dst);
            Assert.Equal(default, dst);

            //6. Missing string in text to sbyte.
            src = DvText.NA;
            dst = -1;
            error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Missing text value cannot be converted to integer type.", ex.Message);
                error = true;
            }

            Assert.True(error);
        }

        [Fact]
        public void TXToLong()
        {
            var mapper = GetMapper<DvText, long>();

            Assert.NotNull(mapper);

            //1. long.MinValue in text to long.
            var minValue = long.MinValue;
            var maxValue = long.MaxValue;
            DvText src = new DvText(minValue.ToString());
            var dst = default(long);
            mapper(ref src, ref dst);
            Assert.Equal(dst, minValue);

            //2. long.MaxValue in text to long.
            src = new DvText(maxValue.ToString());
            dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, maxValue);

            //3. long.MinValue - 1 in text to long.
            src = new DvText(((long)minValue - 1).ToString());
            dst = 0;
            mapper(ref src, ref dst);
            Assert.Equal(dst, (long)minValue - 1);

            //4. ERROR condition: long.MaxValue + 1 in text to long.
            src = new DvText(((ulong)maxValue + 1).ToString());
            dst = 0;
            bool error = false;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to long.", ex.Message);
                error = true;
            }

            Assert.True(error);

            //5. Empty value in text to long.
            src = default;
            dst = -1;
            mapper(ref src, ref dst);
            Assert.Equal(default, dst);

            //6. Missing string in text to sbyte.
            error = false;
            src = DvText.NA;
            dst = -1;
            try
            {
                mapper(ref src, ref dst);
            }
            catch (Exception ex)
            {
                Assert.Equal("Missing text value cannot be converted to integer type.", ex.Message);
                error = true;
            }

            Assert.True(error);
        }

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            Assert.True(typeof(TSrc).TryGetDataKind(out DataKind srcDataKind));
            Assert.True(typeof(TDst).TryGetDataKind(out DataKind dstDataKind));
            
            return Conversions.Instance.GetStandardConversion<TSrc, TDst>(
                TextType.Instance, NumberType.FromKind(dstDataKind), out bool identity);
        }
    }
}


