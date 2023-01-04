use ::owo_colors::{OwoColorize, Stream::Stderr, Style};

#[derive(Debug, Clone, Copy)]
pub struct Location {
    line: u32,
    col: u32,
    offset: u32,
}

impl<'s> From<crate::par::Span<'s>> for Location {
    fn from(span: crate::par::Span<'s>) -> Self {
        Self {
            line: span.location_line(),
            col: span.get_utf8_column() as u32,
            offset: span.location_offset() as u32,
        }
    }
}

#[derive(Debug)]
pub struct Error(pub Location, pub &'static str);

impl Error {
    pub fn print(&self, fname: &str, file: &str) {
        let Location { line, col, .. } = self.0;
        let msg = self.1;

        let err_style = Style::new().red().bold();

        eprint!(
            "{fname}:{line}:{col}:{}{msg}\n",
            "error: ".if_supports_color(Stderr, |t| t.style(err_style))
        );
        underline(file, self.0, self.0);
    }
}

fn underline(file: &str, beg: Location, end: Location) {
    // TODO: implement multi-line ranges
    assert_eq!(beg.line, end.line);

    // Find beggining and end of line.
    let line_begin = file[0..beg.offset as usize]
        .char_indices()
        .into_iter()
        .rev()
        .find(|&(_, c)| c == '\n' || c == '\r')
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let line_end = end.offset as usize
        + file[end.offset as usize..]
            .char_indices()
            .into_iter()
            .find(|&(_, c)| c == '\n' || c == '\r')
            .map(|(i, _)| i)
            .unwrap_or(file[end.offset as usize..].len());

    eprint!("| {}\n", &file[line_begin..line_end]);

    let underline_style = Style::new().green().bold();
    let underline = format!("{0:^<1$}", "", ((end.col - beg.col) as usize).max(1));

    eprint!(
        "| {0: <1$}{2}\n",
        "",
        beg.col as usize - 1,
        underline.if_supports_color(Stderr, |t| t.style(underline_style))
    );
}

pub fn print_file_errs(fname: &str, file: &str, errs: &Vec<Error>) {
    for e in errs {
        e.print(fname, file);
    }
}
