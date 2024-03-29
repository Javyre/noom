#[cfg(test)]
fn parse_snap<'s>(prog: &'s str) -> crate::par::Block<'s> {
    use std::io::Write;

    use crate::err;
    use crate::par;

    let (chunk, errs) = par::parse_chunk(prog);

    let mut stderr = std::io::stderr();
    err::write_file_errs(&mut stderr, "-", &prog, &errs).unwrap();
    stderr.flush().unwrap();

    assert_eq!(errs.len(), 0);

    chunk
}

macro_rules! test_par {
    ($name:ident, $prog:expr) => {
        #[test]
        fn $name() {
            insta::assert_yaml_snapshot!(parse_snap($prog));
        }
    };
}

test_par!(
    test_table,
    r"
let foo = {
    { abc: 'something', asd, 123 },
    { [2]: 'something', 1, 123 },
    { [asd]: 'other', 123, 544 }
};
"
);

test_par!(
    test_func_assign,
    r"
f = fn (a: number) -> number { a * 2 };
"
);

test_par!(
    test_func_let,
    r"
let f(a: number) -> number = a * 2;
"
);

test_par!(
    test_func_let_2,
    r"
let f = fn (a: number) -> number { a * 2 };
"
);

test_par!(
    test_func_let_3,
    r"
let f: (number) -> number = fn (a){ a * 2 };
"
);

test_par!(
    test_func_let_4,
    r"
let f = fn (a: number) -> number: .{ a * 2 };
"
);

test_par!(
    test_if,
    r"
if (1 + a > foo) .{
    then_()
} else .{
    else_()
};
"
);

test_par!(
    test_if_2,
    r"
if (1 + a > foo) .{
    then_()
};
"
);

test_par!(
    test_if_3,
    r"
if (cond_a) foo()
else if (cond_b) bar()
else if (cond_c) baz()
else baz();
"
);

test_par!(
    test_for,
    r"
for (i in ipairs(list)) .{
    break;
};
"
);

test_par!(
    test_pipe,
    r"
a
|> print()
|> pass();
"
);
