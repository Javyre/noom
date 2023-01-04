use std::cell::RefCell;

// mod lex;
mod err;
mod par;

// use logos::Logos;

fn main() {
    let prog = r#"
// test
let do_something(thing) = .{
    {hello = world, [(123   = 234};;;
    ^ (1123 *;
    456;
    let a = b;
};
    
let a = b;
"#;

    let state = RefCell::new(par::State::new());
    let (chunk, errs) = par::parse_chunk(par::Span::new_extra(prog, &state));
    println!("chunk: {chunk:#?}");
    err::write_file_errs(&mut std::io::stderr(), "-", prog, &errs).unwrap();
}
