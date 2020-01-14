### The Expression Language

The language for the expression estimator should be comfortable to a broad range of users. It shares many similarities with some popular languages.
It is case sensitive, supports multiple types and has a rich set of operators and functions. It is pure functional, in the sense that there are no
mutable values or mutating operations in the language. It does not have, nor need, any exception mechanism, instead producing NA values when a normal
value is not appropriate. It is statically typed, but all types are inferred by the compiler.

## Syntax

Syntax for the lambda consists of a parameter list followed by either colon (:) or arrow (=>) followed by an expression.
The parameter list can be either a single identifier or a comma-separated list of one or more identifiers surrounded by parentheses.

_lambda:_

-    _parameter-list   **:** expression_
-    _parameter-list   **=>** expression_

_parameter-list:_

- _identifier_
- **(**  _parameter-names_   **)**

_parameter-names:_

- _identifier_
- _identifier   **,**  parameter-names_

The expression can use parameters, literals, operators, with-expressions, and functions.

## Literals

- The boolean literals are true and false.
- Integer literals may be decimal or hexadecimal (e.g., 0x1234ABCD). They can be suffixed with u or U, indicating unsigned,
as well as l or L, indicating long (Int64). The use of u or U is rare and only affects promotion of certain 32 bit hexadecimal values,
determining whether the constant is considered a negative Int32 value or a positive Int64 value.
- Floating point literals use the standard syntax, including exponential notation (123.45e-37). They can be suffixed with
f or F, indicating single precision, or d or D, indicating double precision. Unlike in C#, the default precision of a
floating point literal is single precision. To specify double precision, append d or D.
- Text literals are enclosed in double-quotation marks and support the standard escape mechanisms.

## Operators

The operators of the expression language are listed in the following table, in precendence order. Unless otherwise noted,
binary operators are left associative and propagate NA values (if either operand value is NA, the result is NA). Generally,
overflow of integer values produces NA, while overflow of floating point values produces infinity.

| **Operator**    | **Meaning**     | **Arity**| **Comments** |
| --- | --- | ---| --- |
| ​?   : | ​conditional | ​Ternary | The expression ​condition ? value1 : value2 resolves to value1 if condition is true and to value2 if condition is false. The condition must be boolean, while value1 and value2 must be of compatible type. |
| ​?? | ​coalesce | ​Binary | The expression ​x ?? y resolves to x if x is not NA, and resolves to y otherwise. The operands must be both Singles, or both Doubles. This operator is right associative. |
| \| \| or | ​logical or | ​Binary | ​The operands and result are boolean. If one operand is true, the result is true, otherwise it is false. |
| ​&&   and | ​logical and | ​Binary | ​The operands and result are boolean. If one operand is false, the result is false, otherwise, it is true. |
| ​==,   =<br> !=,   <><br><,   <=<br>>,   >= | equals<br>not equals<br>less than or equal to<br>greater than or equal to | ​Multiple |- The comparison operators are multi-arity, meaning they can be applied to two or more operands. For example, a == b == c results in true if a, b, and c have the same value. The not equal operator requires that all of the operands be distinct, so 1 != 2 != 1 is false. To test whether x is non-negative but less than 10, use 0 <= x < 10. There is no need to write 0 <= x && x < 10, and doing so will not perform as well. Operators listed on the same line can be combined in a single expression, so a > b >= c is legal, but a < b >= c is not.<br>- Equals and not equals apply to any operand type, while the ordered operators require numeric operands. |
| ​+   - | ​addition and subtraction | ​Binary | Numeric addition and subtraction with NA propagation. |
| ​\*   /   % | ​multiplication, division, and modulus | ​Binary | ​​Numeric multiplication, division, and modulus with NA propagation. |
| ​-   !   not | numeric negation and logical not​ | ​Unary | ​These are unary prefix operators, negation (-) requiring a numeric operand, and not (!) requiring a boolean operand. |
| ​^ | ​power | ​Binary | ​This is right associative exponentiation. It requires numeric operands. For integer operands, 0^0 produce 1.|
| ​(   ) | ​parenthetical grouping | ​Unary | Standard meaning. |

## The With Expression

The syntax for the with-expression is:

_with-expression:_

-  **with**   **(**    assignment-list   **;**    expression   **)**

_assignment-list:_

-    assignment
-    assignment   **,**    assignment-list

_assignment:_

-   identifier   **=**    expression

The with-expression introduces one or more named values. For example, the following expression converts a celcius temperature to fahrenheit, then produces a message based on whether the fahrenheit is too low or high.
```
c => with(f = c * 9 / 5 + 32 ; f < 60 ? "Too Cold!" : f > 90 ? "Too Hot!" : "Just Right!")
```
The expression for one assignment may reference the identifiers introduced by previous assignments, as in this example that returns -1, 0, or 1 instead of the messages:
```
c : with(f = c * 9 / 5 + 32, cold = f < 60, hot = f > 90 ; -float(cold) + float(hot))
```
As demonstrated above, the with-expression is useful when an expression value is needed multiple times in a larger expression. It is also useful when dealing with complicated or significant constants:
```
    ticks => with(
        ticksPerSecond = 10000000L,
        ticksPerHour = ticksPerSecond \* 3600,
        ticksPerDay = ticksPerHour \* 24,
        day = ticks / ticksPerDay,
        dayEpoch = 1 ;
        (day + dayEpoch) % 7)
```
This computes the day of the week from the number of ticks (as an Int64) since the standard .Net DateTime epoch (01/01/0001
in the idealized Gregorian calendar). Assignments are used for number of ticks in a second, number of ticks in an hour,
number of ticks in a year, and the day of the week for the epoch. For this example, we want to map Sunday to zero, so,
since the epoch is a Monday, we set dayEpoch to 1. If the epoch were changed or we wanted to map a different day of the week to zero,
we'd simply change dayEpoch. Note that ticksPerSecond is defined as 10000000L, to make it an Int64 value (8 byte integer).
Without the L suffix, ticksPerDay will overflow Int32's range.

## Functions

The expression transform supports many useful functions.

General unary functions that can accept an operand of any type are listed in the following table.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​isna | ​test for na | Returns a boolean value indicating whether the operand is an NA value. |
| ​na | the na value | ​Returns the NA value of the same type as the operand (either float or double). Note that this does not evaluate the operand, it only uses the operand to determine the type of NA to return, and that determination happens at compile time. |
| ​default | ​the default value | ​​Returns the default value of the same type as the operand. For example, to map NA values to default values, use x ?? default(x). Note that this does not evaluate the operand, it only uses the operand to determine the type of default value to return, and that determination happens at compile time. For numeric types, the default is zero. For boolean, the default is false. For text, the default is empty. |

The unary conversion functions are listed in the following table. An NA operand produces an NA, or throws if the type does not support it.
A conversion that doesn't succeed, or overflow also result in NA or an exception. The most common case of this is when converting from text,
which uses the standard conversion parsing. When converting from a floating point value (float or double) to an integer value
(Int32 or Int64), the conversion does a truncate operation (round toward zero).

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​bool | ​convert to Boolean | The operand must be text or boolean. |
| ​int | convert to <xref:System.Int32> | ​The input may be of any type. |
| ​long | ​convert to <xref:System.Int64> | The input may be of any type. |
| ​single, float | ​convert to <xref:System.Single> | ​The input may be of any type. |
| ​double | ​convert to <xref:System.Double> | ​​The input may be of any type. |
| ​text | ​convert to [text](xref:Microsoft.ML.Data.TextDataViewType) | ​​​The input may be of any type. This produces a default text representation. |

The unary functions that require a numeric operand are listed in the following table. The result type is the same as the operand type. An NA operand value produces NA.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​abs | ​absolute value | ​Produces the absolute value of the operand. |
| ​sign | sign (-1, 0, 1) | ​Produces -1, 0, or 1 depending on whether the operand is negative, zero, or positive. |

The binary functions that require numeric operands are listed in the following table. When the operand types aren't the same,
the operands are promoted to an appropriate type. The result type is the same as the promoted operand type. An NA operand value produces NA.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​min | ​minimum | ​Produces the minimum of the operands. |
| ​max | maximum | ​Produces the maximum of the operands. |

The unary functions that require a floating point operand are listed in the following table. The result type is the same as the operand type. Overflow produces infinity. Invalid input values produce NA.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​sqrt | ​square root | ​Negative operands produce NA. |
| trunc, ​truncate | ​truncate to an integer | ​Rounds toward zero to the nearest integer value. |
| ​floor | ​floor | ​Rounds toward negative infinity to the nearest integer value. |
| ​ceil, ceiling | ​ceiling | ​Rounds toward positive infinity to the nearest integer value. |
| ​round | ​unbiased rounding | ​Rounds to the nearest integer value. When the operand is half way between two integer values, this produces the even integer. |
| ​exp | ​exponential | ​Raises e to the operand. |
| ln, ​log | ​logarithm | ​Produces the natural (base e) logarithm. There is also a two operand version of log for using a different base. |
| ​deg, degrees | ​radians to degrees | ​Maps from radians to degrees. |
| ​rad, radians | ​degrees to radians | ​Maps from degrees to radians. |
| ​sin, sind | ​sine | ​Takes the sine of an angle. The sin function assumes the operand is in radians, while the sind function assumes that the operand is in degrees. |
| ​cos, cosd | ​cosine | ​​Takes the cosine of an angle. The cos function assumes the operand is in radians, while the cosd function assumes that the operand is in degrees. |
| ​tan, tand | ​tangent | ​​Takes the tangent of an angle. The tan function assumes the operand is in radians, while the tand function assumes that the operand is in degrees. |
| ​sinh | ​hyperbolic sine | ​Takes the hyperbolic sine of its operand. |
| ​cosh | ​hyperbolic cosine | ​​Takes the hyperbolic cosine of its operand. |
| ​tanh | ​hyperbolic tangent | ​​Takes the hyperbolic tangent of its operand. |
| ​asin | ​inverse sine | ​Takes the inverse sine of its operand. |
| ​acos | ​inverse cosine | ​​Takes the inverse cosine of its operand. |
| ​atan | ​inverse tangent | ​​Takes the inverse tangent of its operand. |

The binary functions that require floating point operands are listed in the following table. When the operand types aren&#39;t the same, the operands are promoted to an appropriate type. The result type is the same as the promoted operand type. An NA operand value produces NA.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​log | ​logarithm with given base | ​The second operand is the base. The first is the value to take the logarithm of. |
| ​atan2, atanyx | determine angle | Determines the angle between -pi and pi from the given y and x values. Note that y is the first operand. |

The text functions are listed in the following table.

| **​Name** | ​ **Meaning** | ​ **Comments** |
| --- | --- | --- |
| ​len(x) | length of text | The operand must be text. The result is an I4 indicating the length of the operand. If the operand is NA, the result is NA. |
| ​lower(x), upper(x) | lower or upper case | Maps the text to lower or upper case. |
| left(x, k), ​right(x, k) | ​substring | ​The first operand must be text and the second operand must be Int32. If the second operand is negative it is treated as an offset from the end of the text. This adjusted index is then clamped to 0 to len(x). The result is the characters to the left or right of the resulting position. |
| ​mid(x, a, b) | ​substring | ​The first operand must be text and the other two operands must be Int32. The indices are transformed in the same way as for the left and right functions: negative values are treated as offsets from the end of the text; these adjusted indices are clamped to 0 to len(x). The second clamped index is also clamped below to the first clamped index. The result is the characters between these two clamped indices. |
| ​concat(x1, x2, ..., xn) | ​concatenation | ​This accepts an arbitrary number of operands (including zero). All operands must be text. The result is the concatenation of all the operands, in order. |