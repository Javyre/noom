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
    { abc = 'something', 123, 123 },
    { [2] = 'something', 1, 123 },
    { [asd] = 'other', 123, 544 }
};
"
);

test_par!(
    test_func_assign,
    r"
f(a: number): number = a * 2;
"
);

test_par!(
    test_func_let,
    r"
let f(a: number): number = a * 2;
"
);

test_par!(
    test_func_let_2,
    r"
let f = .(a: number): number { a * 2 };
"
);

test_par!(
    test_func_let_3,
    r"
let f: (number): number = .(a){ a * 2 };
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
