use std::{cell::RefCell, io::Write, process::exit};

use err::Level;

// mod lex;
mod emit;
mod err;
mod luish;
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

    let mut stderr = std::io::stderr();
    err::write_file_errs(&mut stderr, "-", prog, &errs).unwrap();
    stderr.flush().unwrap();

    if !errs
        .iter()
        .any(|e| if let Level::Error = e.0 { true } else { false })
    {
        emit::emit_chunk(&mut std::io::stdout(), chunk).unwrap();
    } else {
        exit(1);
    }
}
