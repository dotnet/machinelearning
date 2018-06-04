# JSON Graph format

The entry point graph in TLC is an array of _nodes_. Each node is an object with the following fields:

- _name_: string. Required. Name of the entry point.
- _inputs_: object. Optional. Specifies non-default inputs to the entry point. 
Note that if the entry point has required inputs (which is very common), the _inputs_ field is requred.
- _outputs_: object. Optional. Specifies the variables that will hold the node's outputs.

## Input and output types
The following types are supported in JSON graphs:

- _string_. Represented as a JSON string, maps to a C# string.
- _float_. Represented as a JSON float, maps to a C# float or double.
- _bool_. Represented as a JSON bool, maps to a C# bool.
- _enum_. Represented as a JSON string, maps to a C# enum. The allowed values are those of the C# enum (they are also listed in the manifest).
- _int_. Currently not implemented. Represented as a JSON integer, maps to a C# int or long.
- _array_ of the above. Represented as a JSON array, maps to a C# array.
- _dictionary_. Currently not implemented. Represented as a JSON object, maps to a C# `Dictionary<string,T>`.
- _component_. Currently not implemented. Represented as a JSON object with 2 fields: _name_:string and _settings_:object.

## Variables
The following input/output types can not be represented as a JSON value:
- _DataView_
- _FileHandle_
- _TransformModel_
- _PredictorModel_

These must be passed as _variables_. The variable is represented as a JSON string that begins with "$". 
Note the following rules:

- A variable can appear in the _outputs_ only once per graph. That is, the variable can be 'assigned' only once. 
- If the variable is present in _inputs_ of one node and in the _outputs_ of another node, this signifies the graph 'edge'. 
The same variable can participate in many edges.
- If the variable is present only in _inputs_, but never in _outputs_, it is a _graph input_. All graph inputs must be provided before
a graph can be run.
- The variable has a type, which is the type of inputs (and, optionally, output) that it appears in. If the type of the variable is 
ambiguous, TLC throws an exception.
- Circular references. The experiment graph is expected to be a DAG. If the circular dependency is detected, TLC throws an exception. 
_Currently, this is done lazily: if we couldn't ever run a node because it's waiting for inputs, we throw._

### Variables for arrays and dictionaries.
It is allowed to define variables for arrays and dictionaries, as long as the item types are valid variable types (the four types listed above).
They are treated the same way as regular 'scalar' variables.

If we want to reference an item of the collection, we can use the `[]` syntax:
- `$var[5]` denotes 5th element of an array variable.
- `$var[foo]` and `$var['foo']` both denote the element with key 'foo' of a dictionary variable.
_This is not yet implemented._

Conversely, if we want to build a collection (array or dictionary) of variables, we can do it using JSON arrays and objects:
- `["$v1", "$v2", "$v3"]` denotes an array containing 3 variables.
- `{"foo": "$v1", "bar": "$v2"}` denotes a collection containing 2 key-value pairs.
_This is also not yet implemented._

## Example of a JSON entry point manifest object, and the respective entry point graph node
Let's consider the following manifest snippet, describing an entry point _'CVSplit.Split'_:
```
    {
      "name": "CVSplit.Split",
      "desc": "Split the dataset into the specified number of cross-validation folds (train and test sets)",
      "inputs": [
        {
          "name": "Data",
          "type": "DataView",
          "desc": "Input dataset",
          "required": true
        },
        {
          "name": "NumFolds",
          "type": "Int",
          "desc": "Number of folds to split into",
          "required": false,
          "default": 2
        },
        {
          "name": "StratificationColumn",
          "type": "String",
          "desc": "Stratification column",
          "aliases": [
            "strat"
          ],
          "required": false,
          "default": null
        }
      ],
      "outputs": [
        {
          "name": "TrainData",
          "type": {
            "kind": "Array",
            "itemType": "DataView"
          },
          "desc": "Training data (one dataset per fold)"
        },
        {
          "name": "TestData",
          "type": {
            "kind": "Array",
            "itemType": "DataView"
          },
          "desc": "Testing data (one dataset per fold)"
        }
      ]
    }
```

As we can see, the entry point has 3 inputs (one of them required), and 2 outputs.
The following is a correct graph containing call to this entry point:
```
{
  "nodes": [
    {
      "name": "CVSplit.Split",
      "inputs": {
        "Data": "$data1"
      },
      "outputs": {
        "TrainData": "$cv"
      }
    }]
}
```