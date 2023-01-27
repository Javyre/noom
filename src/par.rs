use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take},
    character::complete::{alpha1, alphanumeric1, digit0, digit1, multispace1},
    combinator::{all_consuming, cond, consumed, eof, map, opt, peek, recognize, success, value},
    multi::{many0_count, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Slice,
};
use nom_locate::{position, LocatedSpan};
use serde::ser::{SerializeStruct, Serializer};
use serde::Serialize;
use std::{cell::RefCell, marker::PhantomData};

use crate::err::{Error, Level};

// ISpan used as cursor during parsing but converted to Span when stored.
type NomError<'s, 't> = nom::error::Error<ISpan<'s, 't>>;
pub type ISpan<'s, 't> = LocatedSpan<&'s str, &'t RefCell<State<'s>>>;
type IResult<'s, 't, T, E = NomError<'s, 't>> = nom::IResult<ISpan<'s, 't>, T, E>;
trait Parser<'s: 't, 't, O>: nom::Parser<ISpan<'s, 't>, O, NomError<'s, 't>> {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span<'s> {
    offset: usize,
    line: u32,
    fragment: &'s str,
}

impl<'s> Serialize for Span<'s> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Span", 3)?;
        state.serialize_field("line", &self.line)?;
        state.serialize_field("col", &self.get_utf8_column())?;
        state.serialize_field("fragment", &self.fragment)?;
        state.end()
    }
}

// match LocatedSpan's api for simplicity
impl<'s> Span<'s> {
    pub fn get_utf8_column(&self) -> usize {
        Into::<LocatedSpan<&'s str, ()>>::into(*self).get_utf8_column()
    }

    pub fn location_offset(&self) -> usize {
        self.offset
    }

    pub fn location_line(&self) -> u32 {
        self.line
    }

    pub fn fragment(&self) -> &&'s str {
        &self.fragment
    }

    pub fn len(&self) -> usize {
        self.fragment.len()
    }
}

impl<'s> Into<LocatedSpan<&'s str, ()>> for Span<'s> {
    fn into(self) -> LocatedSpan<&'s str, ()> {
        unsafe { LocatedSpan::new_from_raw_offset(self.offset, self.line, self.fragment, ()) }
    }
}

impl<'s: 't, 't> From<ISpan<'s, 't>> for Span<'s> {
    fn from(value: ISpan<'s, 't>) -> Self {
        Self {
            offset: value.location_offset(),
            line: value.location_line(),
            fragment: value.fragment(),
        }
    }
}

#[derive(Debug)]
pub struct State<'s> {
    errs: Vec<Error>,
    _phantom: PhantomData<&'s ()>,
    // TODO??: lookup table for user-defined operators and their precedence.
}

impl<'s> State<'s> {
    pub fn new() -> Self {
        Self {
            errs: Vec::new(),
            _phantom: PhantomData,
        }
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

#[derive(Debug, PartialEq, Serialize)]
pub enum Type<'s> {
    Error,
    TypeIdent(Span<'s>, Vec<Type<'s>>),
    Table {
        entry_types: Vec<(TableTypeKey<'s>, Type<'s>)>,
        // TODO: index types i.e. [i: number]: T
    },
    Func(Vec<Type<'s>>, Box<Type<'s>>),
    Union(Vec<Type<'s>>),
    Intersect(Vec<Type<'s>>),
}

#[derive(Debug, PartialEq, Serialize)]
pub enum TableTypeKey<'s> {
    Ident(Span<'s>),
    Index(u32),
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Block<'s> {
    pub stmts: Vec<Stmt<'s>>,
    pub ret: Option<Box<Expr<'s>>>,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum ForIterator<'s> {
    Expr(Expr<'s>),
    Range(Expr<'s>, Expr<'s>, Option<Expr<'s>>),
}

#[derive(Debug, PartialEq, Serialize)]
pub enum Stmt<'s> {
    Error,
    Let(Ident<'s>, Option<Type<'s>>, Expr<'s>),
    // TODO: implement paths 'some.thing.foo'
    Assign(Expr<'s>, Expr<'s>),
    For {
        it_var: Ident<'s>,
        it_type: Option<Type<'s>>,
        it: ForIterator<'s>,
        body: Expr<'s>,
    },
    Expr(Expr<'s>),
    Return(Option<Expr<'s>>),
    Break,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum QuoteType {
    Double,
    Single,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum Expr<'s> {
    Error,
    Ident(Ident<'s>),
    Path(Box<Expr<'s>>, Vec<Ident<'s>>),
    Index(Box<Expr<'s>>, Box<Expr<'s>>),
    Method(Box<Expr<'s>>, Ident<'s>),
    Number(Number<'s>),
    String(Span<'s>, QuoteType),
    Tag(Span<'s>),
    Table(Table<'s>),
    BinaryOp(Box<Expr<'s>>, Span<'s>, Box<Expr<'s>>),
    UnaryOp(Span<'s>, Box<Expr<'s>>),
    Call(Box<Expr<'s>>, Vec<Expr<'s>>),
    Func(
        Vec<(Ident<'s>, Option<Type<'s>>)>,
        Box<Expr<'s>>,
        Option<Type<'s>>,
    ),
    Block(Block<'s>),
    If {
        cases: Vec<(Expr<'s>, Expr<'s>)>,
        else_body: Option<Box<Expr<'s>>>,
    },
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Ident<'s> {
    pub span: Span<'s>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Number<'s> {
    pub span: Span<'s>,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum TableKey<'s> {
    Expr(Expr<'s>),
    Ident(Ident<'s>),
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Table<'s> {
    pub span: Span<'s>,
    pub entries: Vec<(Option<TableKey<'s>>, Expr<'s>)>,
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
fn expect<'s: 't, 't, O>(
    mut f: impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O>,
    err_msg: &'static str,
) -> impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, Option<O>> {
    move |i| match f(i) {
        Ok((i, o)) => Ok((i, Some(o))),
        Err(nom::Err::Error(nom::error::Error { input: i, .. }))
        | Err(nom::Err::Failure(nom::error::Error { input: i, .. })) => {
            i.extra
                .borrow_mut()
                .errs
                .push(Error(Level::Error, i.slice(0..0).into(), err_msg));
            Ok((i, None))
        }
        Err(e) => Err(e),
    }
}

fn skip_err_until<'s: 't, 't, O1, O2>(
    mut until: impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O1>,
    mut f: impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O2>,
) -> impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, Option<O2>> {
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
                i = take::<usize, ISpan<'s, 't>, NomError<'s, 't>>(1usize)(i)
                    .unwrap()
                    .0;
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

fn parse_comment<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, ISpan<'s, 't>> {
    recognize(pair(tag("//"), opt(is_not("\n\r"))))(i)
}

fn ws<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, ISpan<'s, 't>> {
    recognize(many0_count(alt((multispace1, parse_comment))))(i)
}

fn tok<'s: 't, 't, O>(
    f: impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O>,
) -> impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O> {
    preceded(ws, f)
}

fn tok_tag<'s: 't, 't>(
    pat: &'static str,
) -> impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, ISpan<'s, 't>> + Copy {
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
        if cfg!(debug_assertions) {
            expect(
                tok_tag($pat),
                concat!("missing `", $pat, "`", " ", file!(), ":", line!()),
            )
        } else {
            expect(tok_tag($pat), concat!("missing `", $pat, "`"))
        }
    };
}

macro_rules! separated_list0_trail {
    ($sep:expr, $elt:expr) => {
        terminated(separated_list0($sep, $elt), opt($sep))
    };
}
macro_rules! separated_list1_trail {
    ($sep:expr, $elt:expr) => {
        terminated(separated_list1($sep, $elt), opt($sep))
    };
}

fn parse_type_primary<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Type<'s>> {
    alt((
        map(
            pair(
                parse_ident,
                map(
                    opt(delimited(
                        tok_tag("<"),
                        separated_list1_trail!(tok_tag(","), parse_type),
                        expect_tok_tag!(">"),
                    )),
                    |args| match args {
                        Some(args) => args,
                        None => vec![],
                    },
                ),
            ),
            |(id, params)| Type::TypeIdent(id.span, params),
        ),
        map(
            pair(
                delimited(
                    tok_tag("("),
                    separated_list0_trail!(tok_tag(","), parse_type),
                    expect_tok_tag!(")"),
                ),
                preceded(expect_tok_tag!(":"), parse_type),
            ),
            |(args, ret)| Type::Func(args, Box::new(ret)),
        ),
        map(
            delimited(
                tok_tag("{"),
                separated_list0_trail!(
                    tok_tag(","),
                    alt((
                        map(parse_type, |t| (None, t)),
                        separated_pair(map(parse_ident, |i| Some(i)), tok_tag(":"), parse_type),
                    ))
                ),
                expect_tok_tag!("}"),
            ),
            |entry_types| Type::Table {
                entry_types: entry_types
                    .into_iter()
                    .enumerate()
                    .map(|(i, (id, t))| match id {
                        Some(id) => (TableTypeKey::Ident(id.span), t),
                        None => (TableTypeKey::Index(i as u32), t),
                    })
                    .collect(),
            },
        ),
    ))(i)
}

// TODO: make this efficient. associativity is irrelevant within a chain of the same operator and
//       we should collect the terms into a single vec as opposed to a binary tree of pairs.
fn parse_type_combination<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Type<'s>> {
    let (i, a) = parse_type_primary(i)?;
    let (i, op) = opt(tok(alt((tok_tag("|"), tok_tag("&")))))(i)?;

    if let Some(op) = op {
        let (i, b) = expect(parse_type_combination, "missing right hand side type")(i)?;
        let b = match b {
            Some(b) => b,
            None => Type::Error,
        };
        return match *op.fragment() {
            "|" => Ok((i, Type::Union(vec![a, b]))),
            "&" => Ok((i, Type::Intersect(vec![a, b]))),
            _ => unreachable!("invalid operator"),
        };
    }

    Ok((i, a))
}

fn parse_type<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Type<'s>> {
    parse_type_combination(i)
}

fn parse_ident_untok<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Ident<'s>> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0_count(alt((alphanumeric1, tag("_")))),
        )),
        |span: ISpan| Ident { span: span.into() },
    )(i)
}

fn parse_ident<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Ident<'s>> {
    tok(parse_ident_untok)(i)
}

#[test]
fn test_parse_ident() {
    let state = RefCell::new(State::new());
    let input = ISpan::new_extra("_foo_123 a", &state);
    assert_eq!(
        parse_ident(input).unwrap().1,
        Ident {
            span: input.slice(0..8).into()
        }
    );
}

fn parse_number<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Number<'s>> {
    tok(map(
        alt((
            recognize(pair(alt((value((), tag("-")), success(()))), digit1)),
            recognize(tuple((tag("-"), digit1, tag("."), digit1))),
            recognize(tuple((digit0, tag("."), digit1))),
        )),
        |span: ISpan| Number { span: span.into() },
    ))(i)
}

fn parse_table<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Table<'s>> {
    map(
        consumed(delimited(
            tok_tag("{"),
            separated_list0_trail!(
                tok_tag(","),
                map(
                    pair(
                        opt(alt((
                            terminated(
                                map(
                                    delimited(tok_tag(".["), parse_expr, expect_tok_tag!("]")),
                                    |e| (TableKey::Expr(e), None),
                                ),
                                expect_tok_tag!(":"),
                            ),
                            terminated(
                                map(
                                    pair(
                                        preceded(pair(ws, tag(".")), parse_ident_untok),
                                        opt(delimited(
                                            tok_tag("("),
                                            parse_defn_args,
                                            expect_tok_tag!(")")
                                        ))
                                    ),
                                    |(i, args)| (TableKey::Ident(i), args)
                                ),
                                tok_tag(":")
                            ),
                        ))),
                        parse_expr,
                    ),
                    |(key, val)| match key {
                        // FIXME: no way of declaring the return type of the function.
                        Some((key, Some(args))) =>
                            (Some(key), Expr::Func(args, Box::new(val), None)),
                        Some((key, None)) => (Some(key), val),
                        None => (None, val),
                    }
                )
            ),
            expect_tok_tag!("}"),
        )),
        |(span, entries)| Table {
            span: span.into(),
            entries,
        },
    )(i)
}

// this parser is infallible once a type declaration is found. So it is safe to directly register
// the types into the context as this will not be backtracked from.
fn parse_defn_args<'s, 't>(
    i: ISpan<'s, 't>,
) -> IResult<'s, 't, Vec<(Ident<'s>, Option<Type<'s>>)>> {
    separated_list0_trail!(
        tok_tag(","),
        pair(
            parse_ident,
            opt(preceded(
                tok_tag(":"),
                map(expect(parse_type, "expected type"), |t| {
                    t.unwrap_or(Type::Error)
                }),
            )),
        )
    )(i)
}
fn parse_func<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    let (i, args) = delimited(tok_tag(".("), parse_defn_args, expect_tok_tag!(")"))(i)?;
    let (i, ret_ty) = opt(preceded(
        tok_tag("->"),
        map(expect(parse_type, "expected type"), |t| {
            t.unwrap_or(Type::Error)
        }),
    ))(i)?;
    let (i, body) = alt((
        preceded(tok_tag(":"), parse_expr),
        map(
            delimited(
                tok_tag("{"),
                parse_block_body(tok_tag("}")),
                expect_tok_tag!("}"),
            ),
            |body| Expr::Block(body),
        ),
    ))(i)?;
    Ok((i, Expr::Func(args, Box::new(body), ret_ty)))
}

fn parse_if<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    let (i, _) = tok_tag("if")(i)?;
    let (i, cond) = delimited(expect_tok_tag!("("), parse_expr, expect_tok_tag!(")"))(i)?;
    let (i, body) = parse_expr(i)?;

    let mut cases = vec![(cond, body)];
    let mut outer_i = i;
    loop {
        let i = outer_i;
        match opt(pair(tok_tag("else"), tok_tag("if")))(i)? {
            (i, Some(_)) => {
                let (i, cond) =
                    delimited(expect_tok_tag!("("), parse_expr, expect_tok_tag!(")"))(i)?;
                let (i, body) = parse_expr(i)?;
                cases.push((cond, body));
                outer_i = i;
            }
            (i, None) => {
                outer_i = i;
                break;
            }
        }
    }
    let i = outer_i;

    let (i, else_body) = opt(preceded(tok_tag("else"), parse_expr))(i)?;

    Ok((
        i,
        Expr::If {
            cases,
            else_body: else_body.map(|b| Box::new(b)),
        },
    ))
}

fn parse_string<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, (ISpan<'s, 't>, QuoteType)> {
    tok(alt((
        map(
            delimited(
                tag("\""),
                recognize(many0_count(alt((
                    is_not("\"\\"),
                    recognize(pair(tag("\\"), take(1usize))),
                )))),
                expect(tag("\""), "expected closing string double-quote"),
            ),
            |s| (s, QuoteType::Double),
        ),
        map(
            delimited(
                tag("'"),
                recognize(many0_count(alt((
                    is_not("'\\"),
                    recognize(pair(tag("\\"), take(1usize))),
                )))),
                expect(tag("'"), "expected closing string quote"),
            ),
            |s| (s, QuoteType::Single),
        ),
    )))(i)
}

fn parse_expr_primary<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    alt((
        map(parse_string, |(s, q)| Expr::String(s.into(), q)),
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
        map(
            preceded(
                pair(ws, tag(".")),
                expect(
                    recognize(pair(
                        alt((alpha1, tag("_"))),
                        many0_count(alt((alphanumeric1, tag("_")))),
                    )),
                    "tag must be a valid identifier",
                ),
            ),
            |s| s.map(|s| Expr::Tag(s.into())).unwrap_or(Expr::Error),
        ),
        parse_if,
        map(parse_number, |n| Expr::Number(n)),
        map(parse_ident, |i| Expr::Ident(i)),
    ))(i)
}

fn is_expr_callable(expr: &Expr) -> bool {
    match expr {
        Expr::Func(..) | Expr::Index(..) | Expr::Path(..) | Expr::Ident(..) => true,
        _ => false,
    }
}

fn is_expr_indexable(expr: &Expr) -> bool {
    match expr {
        Expr::Number(..) | Expr::Tag(..) | Expr::Error => false,
        _ => true,
    }
}

fn parse_expr_unary_postfix<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    let (i, mut e) = parse_expr_primary(i)?;

    let mut outer_i = i;
    loop {
        let i = outer_i;

        //
        // Fn Call
        //
        let (i, args) = cond(
            is_expr_callable(&e),
            opt(delimited(
                tok_tag("("),
                separated_list0_trail!(tok_tag(","), parse_expr),
                expect_tok_tag!(")"),
            )),
        )(i)?;

        if let Some(Some(args)) = args {
            outer_i = i;
            e = Expr::Call(Box::new(e), args);
            continue;
        }

        //
        // Fn Call (string arg)
        //
        let (i, arg) = cond(
            is_expr_callable(&e),
            opt(map(parse_string, |(s, q)| Expr::String(s.into(), q))),
        )(i)?;

        if let Some(Some(arg)) = arg {
            outer_i = i;
            e = Expr::Call(Box::new(e), vec![arg]);
            continue;
        }

        //
        // Path
        //
        let (i, ids) = cond(
            is_expr_indexable(&e),
            opt(preceded(
                tok_tag("."),
                separated_list1(tok_tag("."), parse_ident),
            )),
        )(i)?;
        if let Some(Some(ids)) = ids {
            outer_i = i;
            e = Expr::Path(Box::new(e), ids);
            continue;
        }

        //
        // Index
        //
        let (i, idx) = cond(
            is_expr_indexable(&e),
            opt(delimited(tok_tag("["), parse_expr, tok_tag("]"))),
        )(i)?;
        if let Some(Some(idx)) = idx {
            outer_i = i;
            e = Expr::Index(Box::new(e), Box::new(idx));
            continue;
        }

        //
        // Method Call
        //
        let (i, id_args) = cond(
            is_expr_indexable(&e),
            opt(preceded(
                tok_tag("->"),
                pair(
                    expect(parse_ident, "exptected identifier"),
                    delimited(
                        expect_tok_tag!("("),
                        separated_list0_trail!(tok_tag(","), parse_expr),
                        expect_tok_tag!(")"),
                    ),
                ),
            )),
        )(i)?;
        if let Some(Some((id, args))) = id_args {
            outer_i = i;
            e = match id {
                Some(id) => Expr::Method(Box::new(e), id),
                None => Expr::Error,
            };
            e = Expr::Call(Box::new(e), args);
            continue;
        }

        break;
    }
    let i = outer_i;

    Ok((i, e))
}

fn parse_expr_unary_prefix<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    alt((
        map(
            pair(tok(alt((tag("-"), tag("!")))), parse_expr_unary_prefix),
            |(o, e)| Expr::UnaryOp(o.into(), Box::new(e)),
        ),
        parse_expr_unary_postfix,
    ))(i)
}

macro_rules! defn_parse_lassoc {
    (@alt, $op:expr) => ( $op );
    (@alt, $($ops:expr),+) => ( alt(($($ops),+)) );

    ($name:ident, term: $term_parser:expr, [$($ops:expr),+]) => {
        fn $name<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
            let (i, a) = $term_parser(i)?;
            let (i, op) = opt(tok(defn_parse_lassoc!(@alt, $(tag($ops)),+)))(i)?;

            if let Some(op) = op {
                let (i, b) = expect($name, "missing right hand side expression")(i)?;
                let b = match b {
                    Some(b) => b,
                    None => Expr::Error,
                };
                return Ok((i, Expr::BinaryOp(Box::new(a), op.into(), Box::new(b))));
            }

            Ok((i, a))
        }
    };
}

// |> is syntax sugar. No actual new node type.
// `a |> b() |> c()` becomes `c(b(a))`
fn parse_expr_pipe<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    let (i, terms) =
        separated_list1(tok_tag("|>"), pair(tok(position), parse_expr_unary_prefix))(i)?;
    Ok((
        i,
        terms
            .into_iter()
            .reduce(|(dummy_pos, acc), next| match next {
                (_, Expr::Call(func, mut args)) => {
                    args.push(acc);
                    (dummy_pos, Expr::Call(func, args))
                }
                (pos, _) => {
                    i.extra.borrow_mut().errs.push(Error(
                        Level::Error,
                        pos.into(),
                        "expected function call as rhs to pipe operator.",
                    ));
                    (dummy_pos, Expr::Error)
                }
            })
            .unwrap()
            .1,
    ))
}

defn_parse_lassoc!(parse_expr_factor, term: parse_expr_pipe, ["*", "/"]);
defn_parse_lassoc!(parse_expr_term, term: parse_expr_factor, ["+", "-"]);
defn_parse_lassoc!(parse_expr_concat, term: parse_expr_term, [".."]);
defn_parse_lassoc!(
    parse_expr_comparison,
    term: parse_expr_concat,
    [">", ">=", "<", "<="]
);
defn_parse_lassoc!(parse_expr_and, term: parse_expr_comparison, ["and"]);
defn_parse_lassoc!(parse_expr_or, term: parse_expr_and, ["or"]);
defn_parse_lassoc!(parse_expr_equality, term: parse_expr_or, ["==", "!="]);

fn parse_expr<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Expr<'s>> {
    parse_expr_equality(i)
}

fn parse_let<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Stmt<'s>> {
    let (i, _) = tok_tag("let")(i)?;
    let (i, ident) = parse_ident(i)?;
    let (i, eq_ty) = opt(pair(
        opt(preceded(
            tok_tag(":"),
            map(expect(parse_type, "expected type"), |t| {
                t.unwrap_or(Type::Error)
            }),
        )),
        tok_tag("="),
    ))(i)?;

    match eq_ty {
        Some((ty, _)) => {
            let (i, val) = parse_expr(i)?;
            Ok((i, Stmt::Let(ident, ty, val)))
        }
        None => {
            let (i, args) = delimited(tok_tag("("), parse_defn_args, expect_tok_tag!(")"))(i)?;
            let (i, ret_ty) = opt(preceded(
                tok_tag(":"),
                map(expect(parse_type, "expected type"), |t| {
                    t.unwrap_or(Type::Error)
                }),
            ))(i)?;
            let (i, _) = tok_tag("=")(i)?;
            let (i, body) = parse_expr(i)?;
            Ok((
                i,
                Stmt::Let(ident, None, Expr::Func(args, Box::new(body), ret_ty)),
            ))
        }
    }
}

fn parse_assign_or_expr<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Stmt<'s>> {
    let (i, target) = parse_expr(i)?;
    let (i, eq) = opt(tok_tag("="))(i)?;

    match eq {
        Some(eq) => {
            // Validate assignable expr
            let target = match target {
                Expr::Path(..) | Expr::Index(..) | Expr::Ident(..) => target,
                _ => {
                    i.extra.borrow_mut().errs.push(Error(
                        Level::Error,
                        eq.into(),
                        "unexpected `=` here. This expression is not assignable.",
                    ));
                    Expr::Error
                }
            };

            let (i, val) = parse_expr(i)?;
            Ok((i, Stmt::Assign(target, val)))
        }
        None => Ok((i, Stmt::Expr(target))),
    }
}

fn parse_for<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Stmt<'s>> {
    let (i, _) = tok_tag("for")(i)?;
    let (i, _) = expect_tok_tag!("(")(i)?;
    let (i, it_var) = parse_ident(i)?;
    let (i, it_type) = opt(preceded(
        tok_tag(":"),
        map(expect(parse_type, "expected type"), |t| {
            t.unwrap_or(Type::Error)
        }),
    ))(i)?;
    let (i, _) = expect_tok_tag!("in")(i)?;
    let (i, it) = alt((
        map(
            preceded(
                tok_tag("@range"),
                delimited(
                    tok_tag("("),
                    terminated(
                        tuple((
                            expect(parse_expr, "exptected range begin"),
                            preceded(tok_tag(","), expect(parse_expr, "exptected range end")),
                            opt(preceded(tok_tag(","), parse_expr)),
                        )),
                        opt(tok_tag(",")),
                    ),
                    tok_tag(")"),
                ),
            ),
            |(beg, end, step)| {
                ForIterator::Range(beg.unwrap_or(Expr::Error), end.unwrap_or(Expr::Error), step)
            },
        ),
        map(parse_expr, |e| ForIterator::Expr(e)),
    ))(i)?;
    let (i, _) = expect_tok_tag!(")")(i)?;
    let (i, body) = parse_expr(i)?;

    Ok((
        i,
        Stmt::For {
            it_var,
            it_type,
            it,
            body,
        },
    ))
}

fn parse_stmt<'s, 't>(i: ISpan<'s, 't>) -> IResult<'s, 't, Stmt<'s>> {
    alt((
        parse_let,
        parse_for,
        map(tok_tag("break"), |_| Stmt::Break),
        map(preceded(tok_tag("return"), opt(parse_expr)), |e| {
            Stmt::Return(e)
        }),
        parse_assign_or_expr,
    ))(i)
}

// Infallible (should be called once block has definitely begun via beginning of chunk or `.(){`, `.{`)
fn parse_block_body<'s: 't, 't, O>(
    block_end: impl Copy + FnMut(ISpan<'s, 't>) -> IResult<'s, 't, O>,
) -> impl FnMut(ISpan<'s, 't>) -> IResult<'s, 't, Block<'s>> {
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

pub fn parse_chunk<'s, 't>(chunk: &'s str) -> (Block<'s>, Vec<Error>) {
    let state = RefCell::new(State::new());
    let chunk_span = ISpan::new_extra(&chunk, &state);

    let block = all_consuming(terminated(
        parse_block_body(|i| {
            let (i, _) = ws(i)?;
            eof(i)
        }),
        terminated(ws, expect(eof, "expected EOF")),
    ))(chunk_span)
    .expect("parser cannot fail")
    .1;

    (block, state.into_inner().errs)
}
