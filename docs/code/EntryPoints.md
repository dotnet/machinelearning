# Entry Points And Helper Classes 

## Overview

Entry-points are a way to interface with ML.NET components, by specifying an execution graph of connected inputs and outputs of those components.
Both the manifest describing available components and their inputs/outputs, and an "experiment" graph description, are expressed in JSON. 
The recommended way of interacting with ML.NET through other, non-.NET programming languages, is by composing, and exchanging pipeline or experiment graphs.  

Through the documentaiton, we also refer to them as 'entry points nodes', and not just entry points, and that is because they are used as nodes of the experiemnt graphs. 
The graph 'variables', the various values of the experiment graph JSON properties serve to describe the relationship between the entry point nodes. 
The 'variables' are therefore the edges of the DAG. 

All of ML.NET entry points are described by their manifest. The manifest is another JSON object that documents and describes the structure of an entry points. 
Manifests are referenced to understand what an entry point does, and how it should be constructed, in a graph.  

This document briefly describes the structure of the entry points, the structure of an entry point manifest, and mentions the ML.NET classes that help construct an entry point
graph.

## EntryPoint manifest - the definition of an entry point

An example of an entry point manifest object, specifically for the `MissingValueIndicator` transform, is:

```javascript
{
    "Name": "Transforms.ColumnTypeConverter",
    "Desc": "Converts a column to a different type, using standard conversions.",
    "FriendlyName": "Convert Transform",
    "ShortName": "Convert",
    "Inputs": [
        {   "Name": "Column",
            "Type": {
                "Kind": "Array",
                "ItemType": {
                    "Kind": "Struct",
                    "Fields": [
                        {
                            "Name": "ResultType",
                            "Type": {
                                "Kind": "Enum",
                                "Values": [ "I1","I2","U2","I4","U4","I8","U8","R4","Num","R8","TX","Text","TXT","BL","Bool","TimeSpan","TS","DT","DateTime","DZ","DateTimeZone","UG","U16" ]
                            },
                            "Desc": "The result type",
                            "Aliases": [ "type" ],
                            "Required": false,
                            "SortOrder": 150,
                            "IsNullable": true,
                            "Default": null
                        },
                        {   "Name": "Range",
                            "Type": "String",
                            "Desc": "For a key column, this defines the range of values",
                            "Aliases": [ "key" ],
                            "Required": false,
                            "SortOrder": 150,
                            "IsNullable": false,
                            "Default": null
                        },
                        {   "Name": "Name",
                            "Type": "String",
                            "Desc": "Name of the new column",
                            "Aliases": [ "name" ],
                            "Required": false,
                            "SortOrder": 150,
                            "IsNullable": false,
                            "Default": null
                        },
                        {   "Name": "Source",
                            "Type": "String",
                            "Desc": "Name of the source column",
                            "Aliases": [ "src" ],
                            "Required": false,
                            "SortOrder": 150,
                            "IsNullable": false,
                            "Default": null
                        }
                    ]
                }
            },
            "Desc": "New column definition(s) (optional form: name:type:src)",
            "Aliases": [ "col" ],
            "Required": true,
            "SortOrder": 1,
            "IsNullable": false
        },
        {   "Name": "Data",
            "Type": "DataView",
            "Desc": "Input dataset",
            "Required": true,
            "SortOrder": 2,
            "IsNullable": false
        },
        {   "Name": "ResultType",
            "Type": {
                "Kind": "Enum",
                "Values": [ "I1","I2","U2","I4","U4","I8","U8","R4","Num","R8","TX","Text","TXT","BL","Bool","TimeSpan","TS","DT","DateTime","DZ","DateTimeZone","UG","U16" ]
            },
            "Desc": "The result type",
            "Aliases": [ "type" ],
            "Required": false,
            "SortOrder": 2,
            "IsNullable": true,
            "Default": null
        },
        {   "Name": "Range",
            "Type": "String",
            "Desc": "For a key column, this defines the range of values",
            "Aliases": [ "key" ],
            "Required": false,
            "SortOrder": 150,
            "IsNullable": false,
            "Default": null
        }
    ],
    "Outputs": [
	    {
            "Name": "OutputData",
            "Type": "DataView",
            "Desc": "Transformed dataset" 
        },
        {
            "Name": "Model",
            "Type": "TransformModel",
            "Desc": "Transform model"
        }
    ],
    "InputKind": ["ITransformInput" ],
    "OutputKind": [ "ITransformOutput" ]
}
```

The respective entry point, constructed based on this manifest would be:

```javascript
    {
        "Name": "Transforms.ColumnTypeConverter",
        "Inputs": {
            "Column": [{ 
            "Name": "Features",
                    "Source": "Features"
                }],
            "Data": "$data0",
            "ResultType": "R4"
        },
        "Outputs": {
            "OutputData": "$Convert_Output",
            "Model": "$Convert_TransformModel"
        }
    }
```

## `EntryPointGraph`

This class encapsulates the list of nodes (`EntryPointNode`) and edges
(`EntryPointVariable` inside a `RunContext`) of the graph.

## `EntryPointNode`

This class represents a node in the graph, and wraps an entry point call. It
has methods for creating and running entry points. It also has a reference to
the `RunContext` to allow it to get and set values from `EntryPointVariable`s.

To express the inputs that are set through variables, a set of dictionaries
are used. The `InputBindingMap` maps an input parameter name to a list of
`ParameterBinding`s. The `InputMap` maps a `ParameterBinding` to a
`VariableBinding`.  For example, if the JSON looks like this:

```javascript
'foo': '$bar'
```

the `InputBindingMap` will have one entry that maps the string "foo" to a list
that has only one element, a `SimpleParameterBinding` with the name "foo" and
the `InputMap` will map the `SimpleParameterBinding` to a
`SimpleVariableBinding` with the name "bar". For a more complicated example,
let's say we have this JSON:

```javascript
'foo': [ '$bar[3]', '$baz']
```

the `InputBindingMap` will have one entry that maps the string "foo" to a list
that has two elements, an `ArrayIndexParameterBinding` with the name "foo" and
index 0 and another one with index 1. The `InputMap` will map the first
`ArrayIndexParameterBinding` to an `ArrayIndexVariableBinding` with name "bar"
and index 3 and the second `ArrayIndexParameterBinding` to a
`SimpleVariableBinding` with the name "baz".

For outputs, a node assumes that an output is mapped to a variable, so the
`OutputMap` is a simple dictionary from string to string.

## `EntryPointVariable`

This class represents an edge in the entry point graph. It has a name, a type
and a value. Variables can be simple, arrays and/or dictionaries. Currently,
only data views, file handles, predictor models and transform models are
allowed as element types for a variable.

## `RunContext`

This class is just a container for all the variables in a graph.

## `VariableBinding` and Derived Classes

The abstract base class represents a "pointer to a (part of a) variable". It
is used in conjunction with `ParameterBinding`s to specify inputs to an entry
point node. The `SimpleVariableBinding` is a pointer to an entire variable,
the `ArrayIndexVariableBinding` is a pointer to a specific index in an array
variable, and the `DictionaryKeyVariableBinding` is a pointer to a specific
key in a dictionary variable.

## `ParameterBinding` and Derived Classes

The abstract base class represents a "pointer to a (part of a) parameter". It
parallels the `VariableBinding` hierarchy and it is used to specify the inputs
to an entry point node. The `SimpleParameterBinding` is a pointer to a
non-array, non-dictionary parameter, the `ArrayIndexParameterBinding` is a
pointer to a specific index of an array parameter and the
`DictionaryKeyParameterBinding` is a pointer to a specific key of a dictionary
parameter.