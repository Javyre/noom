use std::{
    ffi::OsStr,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::exit,
};

mod emit;
mod err;
mod luish;
mod par;
mod par_test;

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

        #[arg(short, long = "out")]
        /// Output fire or directory. Defaults to ./lua/ and . for directory and file inputs
        /// respectively.
        output: Option<PathBuf>,
    },
}

fn compile_file(input: &Path, output: &Path) {
    let mut prog = String::new();
    File::open(input)
        .unwrap()
        .read_to_string(&mut prog)
        .unwrap();

    let (chunk, errs) = par::parse_chunk(&prog);

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

fn compile_dir(base_path: &Path, input: &Path, output_base: &Path) {
    for entry in input.read_dir().expect("failed to read directory") {
        let entry = entry.expect("failed to discover directory entry");

        let path = entry.path();
        if path.is_dir() {
            compile_dir(base_path, &path, output_base)
        } else if path.extension() == Some(OsStr::new("nm")) {
            let mut output = output_base.to_owned();
            output.push(path.as_path().strip_prefix(base_path).unwrap());
            output.set_extension("lua");

            std::fs::create_dir_all(output.as_path().parent().unwrap())
                .expect("failed to create output dir");

            compile_file(&path, &output);
        }
    }
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Command::Build { input, output } => {
            if input.is_file() {
                let mut output = output.unwrap_or(".".into());
                if output.is_dir() {
                    output.push(input.file_name().unwrap());
                    output.set_extension("lua");
                }
                compile_file(&input, &output)
            } else {
                let output = output.unwrap_or("./lua/".into());
                compile_dir(&input, &input, &output);
            }
        }
    }
}
