# `TermTransform` Architecture

The term transform takes one or more input columns, and builds a map mapping
observed values into a key type, with various options. This requires first
that we build a map given observed data, and then later have a means of
applying that map to new data. There are four helper classes of objects to
perform this task. We describe them here.

* `Builder` instances can have different behavior depending on the item type
  of the input, and whether we are sorting the input. They have mutable state.
  Crucially they work over only primitive types, and are not aware of whether
  the input data is vector or scalar. As their name implies they are stateful
  objects.

* `Trainer` objects wrap a builder, and have different implementations
  depending on whether their input is vector or scalar. They are also
  responsible for making sure the number of values accumulated does not exceed
  the max terms limit. During the term transform's training, these objects are
  constructed given a row on a particular column, and during training a method
  is called to process that row.

The above two classes of objects will be created and in existence only when
the transform is being trained, that is, in the non-deserializing constructor,
and will not be persisted beyond that point.

* `TermMap` objects are created from builder objects, and are the final term
  map. These are sort of the frozen immutable cousins of builders. Like
  builders they work over primitive types. These objects are the ones
  responsible for serialization and deserialization to the model stream and
  other informational streams, construction of the per-item value mapper
  delegates, and accessors for the term values used in constructing the
  metadata (though they do not handle the actual metadata functions
  themselves). Crucially, these objects can be shared among multiple term
  transforms or multiple columns, and are not associated themselves with a
  particular input dataview or column per se.

* `BoundTermMap` objects are bound to a particular dataview, and a particular
  column. They are responsible for the polymorphism depending on whether the
  column they're mapping is vector or scalar, the creation of the metadata
  accessors, and the creation of the actual getters (though, of course, they
  rely on the term map to do this).
