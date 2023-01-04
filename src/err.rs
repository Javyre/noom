use std::{
    any::{Any, TypeId},
    fmt::Display,
    io::Write,
};

use owo_colors::{OwoColorize, Style};

fn get_stream_type<S: Any>() -> Option<owo_colors::Stream> {
    let tid = TypeId::of::<S>();
    if tid == TypeId::of::<std::io::Stderr>() {
        Some(owo_colors::Stream::Stderr)
    } else if tid == TypeId::of::<std::io::Stdout>() {
        Some(owo_colors::Stream::Stdout)
    } else {
        None
    }
}

enum MaybeStyled<'a, T: OwoColorize> {
    Styled(&'a T, owo_colors::Stream, owo_colors::Style),
    Unstyled(&'a T),
}

impl<'a, T: Display> Display for MaybeStyled<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Styled(s, stream, style) => {
                write!(f, "{}", s.if_supports_color(*stream, |s| s.style(*style)))
            }
            Self::Unstyled(s) => write!(f, "{}", s),
        }
    }
}

trait MaybeStyle: OwoColorize {
    fn maybe_style<'a, S: Any + Write>(&'a self, stream: &S, style: Style)
        -> MaybeStyled<'a, Self>;
}

impl<T: OwoColorize> MaybeStyle for T {
    fn maybe_style<'a, S: Any + Write>(
        &'a self,
        _stream: &S,
        style: Style,
    ) -> MaybeStyled<'a, Self> {
        match get_stream_type::<S>() {
            Some(stream) => MaybeStyled::Styled(self, stream, style),
            None => MaybeStyled::Unstyled(self),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Location {
    line: u32,
    col: u32,
    offset: u32,
    len: u32,
}

impl<'s> From<crate::par::Span<'s>> for Location {
    fn from(span: crate::par::Span<'s>) -> Self {
        Self {
            line: span.location_line(),
            col: span.get_utf8_column() as u32,
            offset: span.location_offset() as u32,
            len: span.len() as u32,
        }
    }
}

#[derive(Debug)]
pub struct Error(pub Location, pub &'static str);

impl Error {
    pub fn print(
        &self,
        out: &mut (impl Write + Any),
        fname: &str,
        file: &str,
    ) -> std::io::Result<()> {
        let Location { line, col, .. } = self.0;
        let msg = self.1;

        let err_style = Style::new().red().bold();

        write!(
            out,
            "{fname}:{line}:{col}:{}{msg}\n",
            "error: ".maybe_style(out, err_style)
        )?;
        underline(out, file, self.0)?;
        Ok(())
    }
}

fn underline(
    out: &mut (impl Write + Any),
    file: &str,
    loc: Location,
) -> std::io::Result<()> {
    // TODO: implement multi-line locations?

    // Find beggining and end of line.
    let line_begin = file[0..loc.offset as usize]
        .char_indices()
        .into_iter()
        .rev()
        .find(|&(_, c)| c == '\n' || c == '\r')
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let line_end = (loc.offset + loc.len) as usize
        + file[(loc.offset + loc.len) as usize..]
            .char_indices()
            .into_iter()
            .find(|&(_, c)| c == '\n' || c == '\r')
            .map(|(i, _)| i)
            .unwrap_or(file[(loc.offset + loc.len) as usize..].len());

    write!(out, "| {}\n", &file[line_begin..line_end])?;

    let underline_style = Style::new().green().bold();
    let underline = format!("{0:^<1$}", "", (loc.len as usize).max(1));

    write!(
        out,
        "| {0: <1$}{2}\n",
        "",
        loc.col as usize - 1,
        underline.maybe_style(out, underline_style)
    )?;

    Ok(())
}

pub fn write_file_errs(
    out: &mut (impl Write + Any),
    fname: &str,
    file: &str,
    errs: &Vec<Error>,
) -> std::io::Result<()> {
    for e in errs {
        e.print(out, fname, file)?;
    }
    Ok(())
}
