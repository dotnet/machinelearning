# Overview

An 'entry point', is a representation of a ML.Net type in json format and it is used to serialize and deserialize an ML.Net type in JSON. 
It is also one of the ways ML.Net uses to deserialize experiments, and the recommended way to interface with other languages. 
In terms defining experiments w.r.t entry points, experiments are entry points DAGs, and respectively, entry points are experiment graph nodes.
That's why through the documentaiton, we also refer to them as 'entry points nodes'.
The graph 'variables', the various values of the experiemnt graph json properties serve to describe the relationship between the entry point nodes. 
The 'variables' are therefore the edges of the DAG. 

All of ML.Net entry points are described by their manifest. The manifest is another json object that documents and describes the structure of an entry points. 
Manifests are referenced to understand what an entry point does, and how it should be constructed, in a graph.  

This document briefly describes the structure of the entry points, the structure of an entry point manifest, and mentions the ML.Net classes that help construct an entry point
graph.

## `EntryPoint manifest - the definition of an entry point`

An example of an entry point manifest object, specifically for the MissingValueIndicator transform, is:

```javascript
    {
      "Name": "Transforms.MissingValueIndicator",
      "Desc": "Create a boolean output column with the same number of slots as the input column, where the output value is true if the value in the input column is missing.",
      "FriendlyName": "NA Indicator Transform",
      "ShortName": "NAInd",
      "Inputs": [
        {
          "Name": "Column",
          "Type": {
            "Kind": "Array",
            "ItemType": {
              "Kind": "Struct",
              "Fields": [
                {
                  "Name": "Name",
                  "Type": "String",
                  "Desc": "Name of the new column",
                  "Aliases": [
                    "name"
                  ],
                  "Required": false,
                  "SortOrder": 150.0,
                  "IsNullable": false,
                  "Default": null
                },
                {
                  "Name": "Source",
                  "Type": "String",
                  "Desc": "Name of the source column",
                  "Aliases": [
                    "src"
                  ],
                  "Required": false,
                  "SortOrder": 150.0,
                  "IsNullable": false,
                  "Default": null
                }
              ]
            }
          },
          "Desc": "New column definition(s) (optional form: name:src)",
          "Aliases": [
            "col"
          ],
          "Required": true,
          "SortOrder": 1.0,
          "IsNullable": false
        },
        {
          "Name": "Data",
          "Type": "DataView",
          "Desc": "Input dataset",
          "Required": true,
          "SortOrder": 1.0,
          "IsNullable": false
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
      "InputKind": [
        "ITransformInput"
      ],
      "OutputKind": [
        "ITransformOutput"
      ]
    }
```

The respective entry point, constructed based on this manifest would be:

```javascript
	{
		"Name": "Transforms.MissingValueIndicator",
		"Inputs": {
			"Column": [
				{
					"Name": "Features",
					"Source": "Features"
				}
			],
			"Data": "$data0"
		},
		"Outputs": {
			"OutputData": "$Output_1528136517433",
			"Model": "$TransformModel_1528136517433"
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

## VariableBinding and Derived Classes

The abstract base class represents a "pointer to a (part of a) variable". It
is used in conjunction with `ParameterBinding`s to specify inputs to an entry
point node. The `SimpleVariableBinding` is a pointer to an entire variable,
the `ArrayIndexVariableBinding` is a pointer to a specific index in an array
variable, and the `DictionaryKeyVariableBinding` is a pointer to a specific
key in a dictionary variable.

## ParameterBinding and Derived Classes

The abstract base class represents a "pointer to a (part of a) parameter". It
parallels the `VariableBinding` hierarchy and it is used to specify the inputs
to an entry point node. The `SimpleParameterBinding` is a pointer to a
non-array, non-dictionary parameter, the `ArrayIndexParameterBinding` is a
pointer to a specific index of an array parameter and the
`DictionaryKeyParameterBinding` is a pointer to a specific key of a dictionary
parameter.