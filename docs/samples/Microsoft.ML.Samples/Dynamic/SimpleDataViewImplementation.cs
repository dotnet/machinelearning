using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    /// <summary>
    /// The <see cref="IDataView"/> interface is the central concept of "data" in
    /// ML.NET. While many conveniences exist to create pre-baked implementations,
    /// it is also useful to know how to create one completely from scratch. We also
    /// take this opportunity to illustrate and motivate the basic principles of how
    /// the IDataView system is architected, since people interested in
    /// implementing <see cref="IDataView"/> need at least some knowledge of those
    /// principles.
    /// </summary>
    public static class SimpleDataViewImplementation
    {
        public static void Example()
        {
            // First we create an array of these objects, which we "present" as this
            // IDataView implementation so that it can be used in a simple ML.NET
            // pipeline.
            var inputArray = new[]
            {
                new InputObject(false, "Hello my friend."),
                new InputObject(true, "Stay awhile and listen."),
                new InputObject(true, "Masterfully done hero!")
            };
            var dataView = new InputObjectDataView(inputArray);

            // So, this is a very simple pipeline: a transformer that tokenizes
            // Text, does nothing with the Label column at all.
            var mlContext = new MLContext();
            var transformedDataView = mlContext.Transforms.Text.TokenizeIntoWords(
                "TokenizedText", "Text").Fit(dataView).Transform(dataView);

            var textColumn = transformedDataView.Schema["Text"];
            var tokensColumn = transformedDataView.Schema["TokenizedText"];

            using (var cursor = transformedDataView.GetRowCursor(
                new[] { textColumn, tokensColumn }))

            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to facilitate buffer sharing (if applicable),
                // and column-type validation once, rather than many times.
                ReadOnlyMemory<char> textValue = default;
                VBuffer<ReadOnlyMemory<char>> tokensValue = default;

                var textGetter = cursor
                    .GetGetter<ReadOnlyMemory<char>>(textColumn);

                var tokensGetter = cursor
                    .GetGetter<VBuffer<ReadOnlyMemory<char>>>(tokensColumn);

                while (cursor.MoveNext())
                {
                    textGetter(ref textValue);
                    tokensGetter(ref tokensValue);

                    Console.WriteLine(
                        $"{textValue} => " +
                        $"{string.Join(", ", tokensValue.DenseValues())}");

                }

                // The output to console is this:

                // Hello my friend. => Hello, my, friend.
                // Stay awhile and listen. => Stay, awhile, and, listen.
                // Masterfully done hero! => Masterfully, done, hero!

                // Note that it may be interesting to set a breakpoint on the
                // Console.WriteLine, and explore what is going on with the cursor,
                // and the buffers. In particular, on the third iteration, while
                // `tokensValue` is logically presented as a three element array,
                // internally you will see that the arrays internal to that
                // structure have (at least) four items, specifically:
                // `Masterfully`, `done`, `hero!`, `listen.`. In this way we see a
                // simple example of the details of how buffer sharing from one
                // iteration to the next actually works.
            }
        }

        private sealed class InputObject
        {
            public bool Label { get; }
            public string Text { get; }

            public InputObject(bool label, string text)
            {
                Label = label;
                Text = text;
            }
        }

        /// <summary>
        /// This is an implementation of <see cref="IDataView"/> that wraps an
        /// <see cref="IEnumerable{T}"/> of the above <see cref="InputObject"/>.
        /// Note that normally under these circumstances, the first recommendation
        /// would be to use a convenience like 
        /// <see cref="DataOperationsCatalog
        /// .LoadFromEnumerable{TRow}(IEnumerable{TRow}, SchemaDefinition)"/>
        /// or something like that, rather than implementing <see cref="IDataView"/>
        /// outright. However, sometimes when code generation is impossible on some
        /// situations, like Unity or other similar platforms, implementing
        /// something even closely resembling this may become necessary.
        ///
        /// This implementation of <see cref="IDataView"/>, being didactic, is much
        /// simpler than practically anything one would find in the ML.NET codebase.
        /// In this case we have a completely fixed schema (the two fields of
        /// <see cref="InputObject"/>), with fixed types.
        ///
        /// For <see cref="Schema"/>, note that we keep a very simple schema based
        /// off the members of the object. You may in fact note that it is possible
        /// in this specific case, this implementation of <see cref="IDatView"/>
        /// could share the same <see cref="DataViewSchema"/> object across all
        /// instances of this object, but since this is almost never the case, I do
        /// not take advantage of that.
        ///
        /// We have chosen to wrap an <see cref="IEnumerable{T}"/>, so in fact only
        /// a very simple implementation is possible. Specifically: we cannot
        /// meaningfully shuffle (so <see cref="CanShuffle"/> is
        /// <see langword="false"/>, and even if a <see cref="Random"/>
        /// parameter were passed to
        /// <see cref="GetRowCursor(IEnumerable{DataViewSchema.Column}, Random)"/>,
        /// we could not make use of it), we do not know the count of the item right
        /// away without counting (so, it is most correct for
        /// <see cref="GetRowCount"/> to return <see langword="null"/>, even after
        /// we might hypothetically know after the first pass, given the
        /// immutability principle of <see cref="IDatView"/>), and the
        /// <see cref="GetRowCursorSet(
        /// IEnumerable{DataViewSchema.Column}, int, Random)"/> method returns a
        /// single item.
        ///
        /// The <see cref="DataViewRowCursor"/> derived class has more documentation
        /// specific to its behavior.
        ///
        /// Note that this implementation, as well as the nested
        /// <see cref="DataViewRowCursor"/> derived class, does almost no validation
        /// of parameters or guard against misuse than we would like from, say,
        /// implementations of the same classes within the ML.NET codebase.
        /// </summary>
        private sealed class InputObjectDataView : IDataView
        {
            private readonly IEnumerable<InputObject> _data;
            public IEnumerable<InputObject> Data
            {
                get
                {
                    return _data;
                }
            }
            public DataViewSchema Schema { get; }
            public bool CanShuffle => false;

            public InputObjectDataView(IEnumerable<InputObject> data)
            {
                _data = data;

                var builder = new DataViewSchema.Builder();
                builder.AddColumn("Label", BooleanDataViewType.Instance);
                builder.AddColumn("Text", TextDataViewType.Instance);
                Schema = builder.ToSchema();
            }

            public long? GetRowCount() => null;

            public DataViewRowCursor GetRowCursor(
                IEnumerable<DataViewSchema.Column> columnsNeeded,
                Random rand = null)

                => new Cursor(this, columnsNeeded.Any(c => c.Index == 0),
                    columnsNeeded.Any(c => c.Index == 1));

            public DataViewRowCursor[] GetRowCursorSet(
                IEnumerable<DataViewSchema.Column> columnsNeeded, int n,
                Random rand = null)

                => new[] { GetRowCursor(columnsNeeded, rand) };

            /// <summary>
            /// Having this be a private sealed nested class follows the typical
            /// pattern: in most <see cref="IDataView"/> implementations, the cursor
            /// instance is almost always that. The only "common" exceptions to this
            /// tendency are those implementations that are such thin wrappings of
            /// existing <see cref="IDataView"/> without even bothering to change
            /// the schema.
            ///
            /// On the subject of schema, note that there is an expectation that
            /// the <see cref="Schema"/> object is reference equal to the
            /// <see cref="IDataView.Schema"/> object that created this cursor, as
            /// we see here.
            ///
            /// Note that <see cref="Batch"/> returns <c>0</c>. As described in the
            /// documentation of that property, that is meant to facilitate the
            /// reconciliation of the partitioning of the data in the case where
            /// multiple cursors are returned from
            /// <see cref="GetRowCursorSet(
            /// IEnumerable{DataViewSchema.Column}, int, Random)"/>, 
            /// but since only one is ever returned from the implementation, this
            /// behavior is appropriate.
            ///
            /// Similarly, since it is impossible to have a shuffled cursor or a
            /// cursor set, it is sufficient for the <see cref="GetIdGetter"/>
            /// implementation to return a simple ID based on the position. If,
            /// however, this had been something built on, hypothetically, an
            /// <see cref="IList{T}"/> or some other such structure, and shuffling
            /// and partitioning was available, an ID based on the index of whatever
            /// item was being returned would be appropriate.
            ///
            /// Note the usage of the <see langword="ref"/> parameters on the
            /// <see cref="ValueGetter{TValue}"/> implementations. This is most
            /// valuable in the case of buffer sharing for <see cref="VBuffer{T}"/>,
            /// but we still of course have to deal with it here.
            ///
            /// Note also that we spend a considerable amount of effort to not make
            /// the <see cref="GetGetter{TValue}(DataViewSchema.Column)"/> and
            /// <see cref="IsColumnActive(DataViewSchema.Column)"/> methods
            /// correctly reflect what was asked for from the
            /// <see cref="GetRowCursor(
            /// IEnumerable{DataViewSchema.Column}, Random)"/> method that was used
            /// to create this method. In this particular case, the point is
            /// somewhat moot: this mechanism exists to enable lazy evaluation,
            /// but since this cursor is implemented to wrap an
            /// <see cref="IEnumerator{T}"/> which has no concept of lazy
            /// evaluation, there is no real practical benefit to doing this.
            /// However, it is best of course to illustrate the general principle
            /// for the sake of the example.
            ///
            /// Even in this simple form, we see the reason why
            /// <see cref="GetGetter{TValue}(DataViewSchema.Column)"/> is
            /// beneficial: the <see cref="ValueGetter{TValue}"/> implementations
            /// themselves are simple to the point where their operation is dwarfed
            /// by the simple acts of casting and validation checking one sees in
            /// <see cref="GetGetter{TValue}(DataViewSchema.Column)"/>. In this way
            /// we only pay the cost of validation and casting once, not every time
            /// we get a value.
            /// </summary>
            private sealed class Cursor : DataViewRowCursor
            {
                private bool _disposed;
                private long _position;
                private readonly IEnumerator<InputObject> _enumerator;
                private readonly Delegate[] _getters;

                public override long Position => _position;
                public override long Batch => 0;
                public override DataViewSchema Schema { get; }

                public Cursor(InputObjectDataView parent, bool wantsLabel,
                    bool wantsText)

                {
                    Schema = parent.Schema;
                    _position = -1;
                    _enumerator = parent.Data.GetEnumerator();
                    _getters = new Delegate[]
                    {
                        wantsLabel ?
                            (ValueGetter<bool>)LabelGetterImplementation : null,

                        wantsText ?
                            (ValueGetter<ReadOnlyMemory<char>>)
                            TextGetterImplementation : null

                    };
                }

                protected override void Dispose(bool disposing)
                {
                    if (_disposed)
                        return;
                    if (disposing)
                    {
                        _enumerator.Dispose();
                        _position = -1;
                    }
                    _disposed = true;
                    base.Dispose(disposing);
                }

                private void LabelGetterImplementation(ref bool value)
                    => value = _enumerator.Current.Label;

                private void TextGetterImplementation(
                    ref ReadOnlyMemory<char> value)

                    => value = _enumerator.Current.Text.AsMemory();

                private void IdGetterImplementation(ref DataViewRowId id)
                    => id = new DataViewRowId((ulong)_position, 0);

                public override ValueGetter<TValue> GetGetter<TValue>(
                    DataViewSchema.Column column)

                {
                    if (!IsColumnActive(column))
                        throw new ArgumentOutOfRangeException(nameof(column));
                    return (ValueGetter<TValue>)_getters[column.Index];
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                    => IdGetterImplementation;

                public override bool IsColumnActive(DataViewSchema.Column column)
                    => _getters[column.Index] != null;

                public override bool MoveNext()
                {
                    if (_disposed)
                        return false;
                    if (_enumerator.MoveNext())
                    {
                        _position++;
                        return true;
                    }
                    Dispose();
                    return false;
                }
            }
        }
    }
}
