namespace Microsoft.ML.Data
{
    [BestFriend]
    internal delegate void ValueMapperEncrypted<TVal1, TVal2, TVal3>(in TVal1 val1, in TVal2 val2, ref TVal3 val3);

    [BestFriend]
    internal interface IValueMapperEncrypted : IValueMapper
    {
        ValueMapperEncrypted<TSrc, TKey, TDst> GetMapper<TSrc, TKey, TDst>();
    }
}