use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take},
    character::complete::{alpha1, alphanumeric1, digit0, digit1, multispace1},
    combinator::{all_consuming, consumed, eof, map, opt, peek, recognize, success, value},
    multi::{many0_count, many1, separated_list0},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    ExtendInto, Slice,
};
use nom_locate::LocatedSpan;
use std::cell::RefCell;

use crate::err::{Error, Level};

type NomError<'s> = nom::error::Error<Span<'s>>;
pub type Span<'s> = LocatedSpan<&'s str, &'s RefCell<State>>;
type IResult<'s, T, E = NomError<'s>> = nom::IResult<Span<'s>, T, E>;
trait Parser<'s, O>: nom::Parser<Span<'s>, O, NomError<'s>> {}

#[derive(Debug)]
pub struct State {
    errs: Vec<Error>,
    // TODO??: lookup table for user-defined operators and their precedence.
}

impl State {
    pub fn new() -> Self {
        Self { errs: Vec::new() }
    }
}

/*
#[derive(Debug, Clone, PartialEq)]
pub struct Span<'s> {
    src: *const u8,
    range: Range<'s>,
}

impl<'s> From<&'s str> for Span<'s> {
    fn from(str: &'s str) -> Self {
        Self {
            src: str.as_ptr(),
            range: Range {
                range: 0..str.len() as u32,
                lifetime: PhantomData::<&'s str>::default(),
            },
        }
    }
}

impl<'s> Offset for Span<'s> {
    fn offset(&self, second: &Self) -> usize {
        debug_assert_eq!(self.src, second.src);
        second.range.range.start as usize - self.range.range.start as usize
    }
}

impl<'s> nom::Slice<RangeTo<usize>> for Span<'s> {
    fn slice(&self, range: RangeTo<usize>) -> Self {
        let mut new_range = self.range.clone();
        new_range.range.end = range.end as u32;

        Self {
            src: self.src,
            range: new_range,
        }
    }
}

impl<'s> nom::InputTake for Span<'s> {
    fn take(&self, count: usize) -> Self {
        let mut new_range = self.range.clone();
        new_range.range.end = self.range.range.start + count as u32;

        Self {
            src: self.src,
            range: new_range,
        }
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        let mut range_a = self.range.clone();
        let mut range_b = self.range.clone();
        range_a.range.end = count as u32;
        range_b.range.start = count as u32;
        (
            Self {
                src: self.src,
                range: range_a,
            },
            Self {
                src: self.src,
                range: range_b,
            },
        )
    }
}
*/

// #[derive(Debug, Clone, PartialEq)]
// pub struct Range<'s> {
//     range: core::ops::Range<u32>,
//     lifetime: PhantomData<&'s str>,
// }

// impl<'s> Range<'s> {
//     pub fn all_of<'a>(s: &'a str) -> Range<'a> {
//         Self {
//             range: 0..s.len() as u32,
//             lifetime: PhantomData::<&'a str>::default(),
//         }
//     }
// }

#[derive(Debug, PartialEq)]
pub struct Block<'s> {
    pub stmts: Vec<Stmt<'s>>,
    pub ret: Option<Box<Expr<'s>>>,
}

#[derive(Debug, PartialEq)]
pub enum Stmt<'s> {
    Error,
    Let(Ident<'s>, Expr<'s>),
    // TODO: implement paths 'some.thing.foo'
    Assign(Ident<'s>, Expr<'s>),
    Expr(Expr<'s>),
}

#[derive(Debug, PartialEq)]
pub enum Expr<'s> {
    Error,
    Ident(Ident<'s>),
    Number(Number<'s>),
    Table(Table<'s>),
    BinaryOp(Box<Expr<'s>>, Span<'s>, Box<Expr<'s>>),
    UnaryOp(Span<'s>, Box<Expr<'s>>),
    Call(Box<Expr<'s>>, Vec<Expr<'s>>),
    Func(Vec<Ident<'s>>, Box<Expr<'s>>),
    Block(Block<'s>),
}

#[derive(Debug, PartialEq)]
pub struct Ident<'s> {
    pub span: Span<'s>,
}

#[derive(Debug, PartialEq)]
pub struct Number<'s> {
    pub span: Span<'s>,
}

#[derive(Debug, PartialEq)]
pub enum TableKey<'s> {
    Expr(Expr<'s>),
    Ident(Ident<'s>),
}

#[derive(Debug, PartialEq)]
pub struct Table<'s> {
    pub span: Span<'s>,
    pub entries: Vec<(TableKey<'s>, Expr<'s>)>,
}

/// For expected success in parasers past a backtrack boundary.
///
///             we can expect() to see '=' here with 100% confidence this isn't just a need for
///             backtracking.
///             v
/// e.g.: let a = b;
///       ^ 'let a' is a boundary into parsing the rest of the let statement with 100% confidence
///         we won't have to backtrack past the 'let'
///
/// e.g.: a + b;
///         ^ A rhs operand should come after '+'
///
fn expect<'s, O>(
    mut f: impl FnMut(Span<'s>) -> IResult<'s, O>,
    err_msg: &'static str,
) -> impl FnMut(Span<'s>) -> IResult<'s, Option<O>> {
    move |i| match f(i) {
        Ok((i, o)) => Ok((i, Some(o))),
        Err(nom::Err::Error(nom::error::Error { input: i, .. }))
        | Err(nom::Err::Failure(nom::error::Error { input: i, .. })) => {
            i.extra
                .borrow_mut()
                .errs
                .push(Error(Level::Error, i.slice(0..1).into(), err_msg));
            Ok((i, None))
        }
        Err(e) => Err(e),
    }
}

fn skip_err_until<'s, O1, O2>(
    mut until: impl FnMut(Span<'s>) -> IResult<'s, O1>,
    mut f: impl FnMut(Span<'s>) -> IResult<'s, O2>,
) -> impl FnMut(Span<'s>) -> IResult<'s, Option<O2>> {
    move |i| match f(i) {
        Ok((i, o)) => Ok((i, Some(o))),

        Err(nom::Err::Error(nom::error::Error { input: mut i, .. }))
        | Err(nom::Err::Failure(nom::error::Error { input: mut i, .. })) => {
            i.extra.borrow_mut().errs.push(Error(
                Level::Error,
                i.slice(0..1).into(),
                "invalid character",
            ));

            // Recover to first position where f(i) succeeds or until(i) reached.
            loop {
                match until(i) {
                    // Act like a peek and don't advance i here.
                    Ok((_, _)) => return Ok((i, None)),
                    _ => {}
                }
                // take the invalid character
                i = take::<usize, Span<'s>, NomError<'s>>(1usize)(i).unwrap().0;
                match f(i) {
                    Ok((i, o)) => return Ok((i, Some(o))),
                    Err(nom::Err::Error(nom::error::Error { input, .. }))
                    | Err(nom::Err::Failure(nom::error::Error { input, .. })) => {
                        i = input;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Err(e) => Err(e),
    }
}

fn parse_comment<'s>(i: Span<'s>) -> IResult<'s, Span<'s>> {
    recognize(pair(tag("//"), opt(is_not("\n\r"))))(i)
}

fn ws<'s>(i: Span<'s>) -> IResult<'s, Span<'s>> {
    recognize(many0_count(alt((multispace1, parse_comment))))(i)
}

fn tok<'s, O>(f: impl FnMut(Span<'s>) -> IResult<'s, O>) -> impl FnMut(Span<'s>) -> IResult<'s, O> {
    preceded(ws, f)
}

fn tok_tag<'s>(pat: &'static str) -> impl FnMut(Span<'s>) -> IResult<'s, Span<'s>> + Copy {
    move |i| match tok(tag(pat))(i) {
        Err(mut err @ nom::Err::Error(_)) | Err(mut err @ nom::Err::Failure(_)) => {
            match &mut err {
                nom::Err::Error(nom::error::Error { ref mut input, .. })
                | nom::Err::Failure(nom::error::Error { ref mut input, .. }) => {
                    *input = i;
                }
                _ => {}
            }
            Err(err)
        }
        other => other,
    }
}

macro_rules! expect_tok_tag {
    ($pat:expr) => {
        expect(tok_tag($pat), concat!("missing `", $pat, "`"))
    };
}

fn parse_ident<'s>(i: Span<'s>) -> IResult<'s, Ident<'s>> {
    tok(map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0_count(alt((alphanumeric1, tag("_")))),
        )),
        |span| Ident { span },
    ))(i)
}

#[test]
fn test_parse_ident() {
    let state = RefCell::new(State::new());
    let input = Span::new_extra("_foo_123 a", &state);
    assert_eq!(
        parse_ident(input).unwrap().1,
        Ident {
            span: input.slice(0..8)
        }
    );
}

fn parse_number<'s>(i: Span<'s>) -> IResult<'s, Number<'s>> {
    tok(map(
        alt((
            recognize(pair(alt((value((), tag("-")), success(()))), digit1)),
            recognize(tuple((tag("-"), digit1, tag("."), digit1))),
            recognize(tuple((digit0, tag("."), digit1))),
        )),
        |span| Number { span },
    ))(i)
}

fn parse_table<'s>(i: Span<'s>) -> IResult<'s, Table<'s>> {
    map(
        consumed(delimited(
            tok_tag("{"),
            separated_list0(
                tok_tag(","),
                separated_pair(
                    alt((
                        map(
                            delimited(tok_tag("["), parse_expr, expect_tok_tag!("]")),
                            |e| TableKey::Expr(e),
                        ),
                        map(parse_ident, |i| TableKey::Ident(i)),
                    )),
                    expect_tok_tag!("="),
                    parse_expr,
                ),
            ),
            expect_tok_tag!("}"),
        )),
        |(span, entries)| Table { span, entries },
    )(i)
}

fn parse_defn_args<'s>(i: Span<'s>) -> IResult<'s, Vec<Ident<'s>>> {
    separated_list0(tok_tag(","), parse_ident)(i)
}
fn parse_func<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
    let (i, args) = delimited(tok_tag(".("), parse_defn_args, expect_tok_tag!(")"))(i)?;
    let (i, body) = delimited(
        tok_tag("{"),
        parse_block_body(tok_tag("}")),
        expect_tok_tag!("}"),
    )(i)?;
    Ok((i, Expr::Func(args, Box::new(Expr::Block(body)))))
}

fn parse_expr_primary<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
    alt((
        map(parse_number, |n| Expr::Number(n)),
        map(parse_ident, |i| Expr::Ident(i)),
        map(parse_table, |t| Expr::Table(t)),
        delimited(
            tok_tag("("),
            map(expect(parse_expr, "expected expression"), |e| {
                e.unwrap_or(Expr::Error)
            }),
            expect_tok_tag!(")"),
        ),
        map(
            delimited(
                tok_tag(".{"),
                parse_block_body(tok_tag("}")),
                expect_tok_tag!("}"),
            ),
            |b| Expr::Block(b),
        ),
        parse_func,
    ))(i)
}

fn parse_expr_call<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
    let (i, e) = parse_expr_primary(i)?;
    let (i, args) = opt(delimited(
        tok_tag("("),
        separated_list0(tok_tag(","), parse_expr),
        expect_tok_tag!(")"),
    ))(i)?;

    if let Some(args) = args {
        return Ok((i, Expr::Call(Box::new(e), args)));
    }

    Ok((i, e))
}

fn parse_expr_unary<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
    alt((
        map(
            pair(tok(alt((tag("-"), tag("!")))), parse_expr_unary),
            |(o, e)| Expr::UnaryOp(o, Box::new(e)),
        ),
        parse_expr_call,
    ))(i)
}

macro_rules! defn_parse_lassoc {
    ($name:ident, term: $term_parser:expr, [$($ops:expr),+]) => {
        fn $name<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
            let (i, a) = $term_parser(i)?;
            let (i, op) = opt(tok(alt(($(tag($ops)),+))))(i)?;

            if let Some(op) = op {
                let (i, b) = expect($name, "missing right hand side expression")(i)?;
                let b = match b {
                    Some(b) => b,
                    None => Expr::Error,
                };
                return Ok((i, Expr::BinaryOp(Box::new(a), op, Box::new(b))));
            }

            Ok((i, a))
        }
    };
}

defn_parse_lassoc!(parse_expr_factor, term: parse_expr_unary, ["*", "/"]);
defn_parse_lassoc!(parse_expr_term, term: parse_expr_factor, ["+", "-"]);
defn_parse_lassoc!(
    parse_expr_comparison,
    term: parse_expr_term,
    [">", ">=", "<", "<="]
);
defn_parse_lassoc!(
    parse_expr_equality,
    term: parse_expr_comparison,
    ["==", "!="]
);

fn parse_expr<'s>(i: Span<'s>) -> IResult<'s, Expr<'s>> {
    parse_expr_equality(i)
}

fn parse_let<'s>(i: Span<'s>) -> IResult<'s, Stmt<'s>> {
    let (i, _) = tok_tag("let")(i)?;
    let (i, ident) = parse_ident(i)?;
    let (i, eq) = opt(tok_tag("="))(i)?;

    match eq {
        Some(_) => {
            let (i, val) = parse_expr(i)?;
            Ok((i, Stmt::Let(ident, val)))
        }
        None => {
            let (i, args) = delimited(tok_tag("("), parse_defn_args, expect_tok_tag!(")"))(i)?;
            let (i, _) = tok_tag("=")(i)?;
            let (i, body) = parse_expr(i)?;
            Ok((i, Stmt::Let(ident, Expr::Func(args, Box::new(body)))))
        }
    }
}

fn parse_assign<'s>(i: Span<'s>) -> IResult<'s, Stmt<'s>> {
    let (i, ident) = parse_ident(i)?;
    let (i, eq) = opt(tok_tag("="))(i)?;

    match eq {
        Some(_) => {
            let (i, val) = parse_expr(i)?;
            Ok((i, Stmt::Assign(ident, val)))
        }
        None => {
            let (i, args) = delimited(tok_tag("("), parse_defn_args, expect_tok_tag!(")"))(i)?;
            let (i, _) = tok_tag("=")(i)?;
            let (i, body) = parse_expr(i)?;
            Ok((i, Stmt::Assign(ident, Expr::Func(args, Box::new(body)))))
        }
    }
}

fn parse_stmt<'s>(i: Span<'s>) -> IResult<'s, Stmt<'s>> {
    alt((parse_let, parse_assign, map(parse_expr, |e| Stmt::Expr(e))))(i)
}

// Infallible (should be called once block has definitely begun via beginning of chunk or `.(){`, `.{`)
fn parse_block_body<'s, O>(
    block_end: impl Copy + FnMut(Span<'s>) -> IResult<'s, O>,
) -> impl FnMut(Span<'s>) -> IResult<'s, Block<'s>> {
    move |mut outer_i| {
        // COMBAK: skip to before next semicolon on "invalid char" error
        let mut stmts = Vec::<Stmt<'s>>::with_capacity(5);
        let mut ret: Option<Box<Expr<'s>>> = None;
        loop {
            let i = outer_i;

            let (i, _) = opt(ws)(i).unwrap();
            let (i, stmt) = skip_err_until(
                tok_tag(";"),
                alt((map(peek(block_end), |_| None), map(parse_stmt, |s| Some(s)))),
            )(i)?;
            let stmt = match stmt {
                Some(Some(stmt)) => stmt,
                Some(None) => {
                    outer_i = i;
                    break;
                }
                None => Stmt::Error,
            };

            let (i, semi) = opt(tok_tag(";"))(i)?;
            if semi.is_none() {
                let (i, end) = peek(opt(block_end))(i)?;
                match end {
                    None => {
                        i.extra.borrow_mut().errs.push(Error(
                            Level::Error,
                            i.slice(0..1).into(),
                            "Expected semicolon here.",
                        ));
                        stmts.push(stmt);
                    }
                    Some(_) => {
                        match stmt {
                            Stmt::Expr(e) => {
                                ret = Some(Box::new(e));
                            }
                            _ => {
                                i.extra.borrow_mut().errs.push(Error(
                                    Level::Error,
                                    i.slice(0..1).into(),
                                    "Expected expr as block return. Add a semicolon here.",
                                ));
                                stmts.push(stmt);
                                ret = Some(Box::new(Expr::Error));
                            }
                        }
                        outer_i = i;
                        break;
                    }
                }
            } else {
                stmts.push(stmt);
            }

            let (i, extra_semi) = opt(recognize(many1(tok_tag(";"))))(i)?;
            if let Some(span) = extra_semi {
                i.extra.borrow_mut().errs.push(Error(
                    Level::Warning,
                    span.into(),
                    "remove these extra semicolons",
                ))
            }

            outer_i = i;
        }
        let i = outer_i;

        Ok((i, Block { stmts, ret }))
    }
}

pub fn parse_chunk<'s>(chunk: Span<'s>) -> (Block<'s>, Vec<Error>) {
    (
        all_consuming(terminated(
            parse_block_body(|i| {
                let (i, _) = ws(i)?;
                eof(i)
            }),
            terminated(ws, expect(eof, "expected EOF")),
        ))(chunk)
        .expect("parser cannot fail")
        .1,
        chunk.extra.replace(State::new()).errs,
    )
}
