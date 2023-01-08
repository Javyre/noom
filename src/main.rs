use std::{
    cell::RefCell,
    default,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::exit,
};

mod emit;
mod err;
mod luish;
mod par;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a file or directory recursively
    Build {
        /// Input file or directory to build
        input: PathBuf,

        #[arg(short, long = "out", default_value = "./lua/")]
        /// Output fire or directory
        output: PathBuf,
    },
}

fn compile_file(input: &Path, output: &Path) {
    let mut prog = String::new();
    File::open(input)
        .unwrap()
        .read_to_string(&mut prog)
        .unwrap();

    let state = RefCell::new(par::State::new());
    let (chunk, errs) = par::parse_chunk(par::Span::new_extra(&prog, &state));

    let mut stderr = std::io::stderr();
    err::write_file_errs(&mut stderr, input.to_str().unwrap(), &prog, &errs).unwrap();
    stderr.flush().unwrap();

    if !errs.iter().any(|e| {
        if let err::Level::Error = e.0 {
            true
        } else {
            false
        }
    }) {
        let mut output = File::create(output).unwrap();
        emit::emit_chunk(&mut output, chunk).unwrap();
    } else {
        exit(1);
    }
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Command::Build { input, mut output } => {
            if input.is_file() {
                if output.is_dir() {
                    output.push(input.file_name().unwrap());
                    output.set_extension("lua");
                }
                compile_file(&input, &output)
            } else {
                unimplemented!("non-file input")
            }
        }
    }
}
