# Noom
> Ergonomic and Typesafe Frontend Language for Lua

*NOTE: This repo is still in very early development. Some documentation is being written mainly to help onboard potential early adopters and contributors.*

Noom is a language meant to compile into lua and interoperate seamlessly with the existing lua ecosystem.

##### Features (in progress)
- Helpful error-tolerant compiler. 
  - *The compiler tries to recover from errors to keep giving useful diagnostics about the rest of the file*
- Expressive syntax minimizing visual clutter.
- Expression-based semantics à-la-Rust.
- Produces readable clean lua code.
- Structural type inference à-la-Typescript. (Not yet)

## Syntax
##### Chunks
Noom follows the concept of chunks from lua. The gist of it is that the contents of a file should be seen as the body of a function that can be called by: `require('path.to.file')`.

##### Statments
Statments are terminated with a `;`. Bare expressions are also acceptable in the place of a statement.

###### Let
Let binds a value to an identifier until the end of the current scope.
```typescript
let variable_name = value_expression;
let variable_name; // declaration of the variable.
```
###### Assignment
Assign values to assignable expressions:
```typescript
some.thing[0] = value;
```

###### For Loop
The for loop follows the syntax `for (var_name in iterator_expr) body_expr`.
```typescript
for (i in ipairs(foo)) .{
    print(i);
};
```

There is a special `@range(start, stop[, step])` builtin directive as well.
```typescript
for (i in @range(1, #foo)) .{
    print(i);
};
```

##### Expressions
Numbers and string litterals have the same syntax as in lua. This also includes long strings (eg: `[[ hello, world]]`).
###### Blocks
Blocks are a sequence of statements optionally finishing in a return value that will give the block expression its value.
If there is no return expression, the value is `nil`.
```typescript
let funny_number = .{
    let a = 58;
    let b = 11;
    a + b // The lack of terminating semicolon indicates this is the return value of the block
};

// The following line would fail as those variables only exist inside the block.
// print(a, b);
```
###### Functions
```typescript
let callback = fn(){
    // block body
    foo();
    return_value
};

// If the body consists of a single expression, the shorter `:` syntax can be used
// The following three lines are equivalent.
let callback = fn(): 123;
let callback = fn(): .{ 123 };
let callback = fn() { 123 };

// The following two lines are equivalent.
let add_one(counter) = counter + 1;
let add_one = fn (counter): counter + 1;
```

###### Tables
Keys-value pairs are separated by `:` and there is some nice sugar for inline functions:
```typescript
let opts = {
    array_element,
    first_opt: value,
    
    // The following two lines are equivalent.
    fn callback_opt(arg): arg + 1,
    callback_opt: fn (arg): arg + 1,
};
```
