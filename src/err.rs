use std::{
    any::{Any, TypeId},
    fmt::Display,
    io::Write,
    ops::Range,
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

impl<'s, 't> From<crate::par::ISpan<'s, 't>> for Location {
    fn from(span: crate::par::ISpan<'s, 't>) -> Self {
        Self {
            line: span.location_line(),
            col: span.get_utf8_column() as u32,
            offset: span.location_offset() as u32,
            len: span.len() as u32,
        }
    }
}

#[derive(Debug)]
pub enum Level {
    Error,
    Warning,
}

#[derive(Debug)]
pub struct Error(pub Level, pub Location, pub &'static str);

impl Error {
    pub fn write(
        &self,
        out: &mut (impl Write + Any),
        fname: &str,
        file: &str,
    ) -> std::io::Result<()> {
        let Location { line, col, .. } = self.1;
        let msg = self.2;

        write!(
            out,
            "{fname}:{line}:{col}:{}{msg}\n",
            match self.0 {
                Level::Error => "error: ".maybe_style(out, Style::new().red().bold()),
                Level::Warning => "warning: ".maybe_style(out, Style::new().yellow().bold()),
            }
        )?;
        underline(out, file, self.1)?;
        Ok(())
    }
}

fn find_line_range(file: &str, offset: usize) -> Range<usize> {
    let last_byte = file.len() - 1;
    let offset = std::cmp::min(offset, last_byte);
    let mut beg = 0;
    let mut end = file.len();

    for i in (0..=std::cmp::max(offset as isize - 1, 0) as usize).rev() {
        match file.as_bytes()[i] {
            b'\n' | b'\r' => {
                beg = i + 1;
                break;
            }
            _ => {}
        }
    }

    for i in offset..=last_byte {
        match file.as_bytes()[i] {
            b'\n' | b'\r' => {
                end = i;
                break;
            }
            _ => {}
        }
    }

    beg..end
}

fn underline(out: &mut (impl Write + Any), file: &str, mut loc: Location) -> std::io::Result<()> {
    // TODO: implement multi-line locations?

    let snippet = &file[find_line_range(file, loc.offset as usize)];

    let underline_style = Style::new().green().bold();
    let underline = format!("{0:^<1$}", "", (loc.len as usize).min(1).max(1));
    let underline = underline.maybe_style(out, underline_style);

    // sometimes EOF is reported as one char after the actual last_char.
    if loc.offset == file.len() as u32 {
        loc.col = snippet.len() as u32 + 1;
    }

    write!(out, "| {}\n", snippet)?;
    write!(out, "| {0: <1$}{2}\n", "", loc.col as usize - 1, underline)?;

    Ok(())
}

pub fn write_file_errs(
    out: &mut (impl Write + Any),
    fname: &str,
    file: &str,
    errs: &Vec<Error>,
) -> std::io::Result<()> {
    for e in errs {
        e.write(out, fname, file)?;
    }
    Ok(())
}
