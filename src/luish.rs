use crate::err::Error;
use crate::par;

#[derive(Clone, Copy, PartialEq)]
pub enum Ident<'s> {
    Str(&'s str),
    Id(u32),
}

pub enum Stmt<'s> {
    Call(Expr<'s>, Vec<Expr<'s>>),
    Local(Ident<'s>, Option<Expr<'s>>),
    Assign(Ident<'s>, Expr<'s>),
    Return(Expr<'s>),
    Do(Vec<Stmt<'s>>),
}

pub enum TableKey<'s> {
    Ident(Ident<'s>),
    Expr(Expr<'s>),
}

pub enum Expr<'s> {
    Nil,
    Ident(Ident<'s>),
    Verbatim(&'s str),
    Table(Vec<(TableKey<'s>, Expr<'s>)>),
    Func(Vec<Ident<'s>>, Vec<Stmt<'s>>),
    Call(Box<Expr<'s>>, Vec<Expr<'s>>),
    UnaryOp(&'s str, Box<Expr<'s>>),
    BinaryOp(Box<Expr<'s>>, &'s str, Box<Expr<'s>>),
}

enum Target<'s, 't> {
    Assign(Ident<'s>),
    Value(&'t mut Option<Expr<'s>>),
    Return,
}

type Stmts<'s> = Vec<Stmt<'s>>;

#[derive(Default)]
pub struct State {
    pub errs: Vec<Error>,
    pub id_gen: u32,
}

impl State {
    fn gen_id<'s>(&mut self) -> Ident<'s> {
        let ret = Ident::Id(self.id_gen);
        self.id_gen += 1;
        ret
    }
}

fn is_same_assign(id: Ident, val: &Expr) -> bool {
    if let Expr::Ident(val_id) = val {
        if *val_id == id {
            return true;
        }
    }
    false
}

fn luify_ident<'s>(id: par::Ident<'s>) -> Ident<'s> {
    Ident::Str(id.span.fragment())
}

fn luify_block_body<'s, 't>(
    s: &mut State,
    out: &mut Stmts<'s>,
    stmts: Vec<par::Stmt<'s>>,
    ret: Option<par::Expr<'s>>,
    target: Target<'s, 't>,
) {
    for stmt in stmts.into_iter() {
        luify_stmt(s, out, stmt);
    }
    if let Some(ret) = ret {
        luify_expr(s, out, ret, target);
    }
}

// default target fulfillment
fn fulfill_target<'s, 't>(
    _s: &mut State,
    out: &mut Stmts<'s>,
    expr: Expr<'s>,
    target: Target<'s, 't>,
) {
    match target {
        Target::Return => out.push(Stmt::Return(expr)),
        Target::Assign(target) => out.push(Stmt::Assign(target, expr)),
        Target::Value(val) => *val = Some(expr),
    }
}

fn luify_expr_val<'s>(s: &mut State, out: &mut Stmts<'s>, expr: par::Expr<'s>) -> Expr<'s> {
    let mut expr_out = None;
    luify_expr(s, out, expr, Target::Value(&mut expr_out));
    expr_out.unwrap()
}

fn luify_expr<'s, 't>(
    s: &mut State,
    out: &mut Stmts<'s>,
    expr: par::Expr<'s>,
    target: Target<'s, 't>,
) {
    match expr {
        par::Expr::Error => unreachable!("error node in ast"),
        par::Expr::Func(args, body) => {
            let mut body_out = Vec::new();
            luify_expr(s, &mut body_out, *body, Target::Return);

            fulfill_target(
                s,
                out,
                Expr::Func(
                    args.into_iter().map(|id| luify_ident(id)).collect(),
                    body_out,
                ),
                target,
            );
        }
        par::Expr::Call(fn_expr, args) => {
            let fn_expr = luify_expr_val(s, out, *fn_expr);

            let args = args
                .into_iter()
                .map(|arg| luify_expr_val(s, out, arg))
                .collect();

            fulfill_target(s, out, Expr::Call(Box::new(fn_expr), args), target);
        }
        par::Expr::Ident(id) => fulfill_target(s, out, Expr::Ident(luify_ident(id)), target),
        par::Expr::Table(par::Table { entries, .. }) => {
            let table = Expr::Table(
                entries
                    .into_iter()
                    .map(|(key, val)| {
                        (
                            match key {
                                par::TableKey::Expr(key) => {
                                    TableKey::Expr(luify_expr_val(s, out, key))
                                }
                                par::TableKey::Ident(key) => TableKey::Ident(luify_ident(key)),
                            },
                            luify_expr_val(s, out, val),
                        )
                    })
                    .collect(),
            );
            fulfill_target(s, out, table, target);
        }
        par::Expr::Number(par::Number { span }) => {
            fulfill_target(s, out, Expr::Verbatim(span.fragment()), target)
        }
        par::Expr::UnaryOp(op, expr) => {
            let unop = Expr::UnaryOp(
                match *op.fragment() {
                    "!" => "not",
                    "-" => "-",
                    _ => unimplemented!("unimplemented unary operator translation"),
                },
                Box::new(luify_expr_val(s, out, *expr)),
            );
            fulfill_target(s, out, unop, target)
        }
        par::Expr::BinaryOp(lhs, op, rhs) => {
            let binop = Expr::BinaryOp(
                Box::new(luify_expr_val(s, out, *lhs)),
                match *op.fragment() {
                    "+" => "+",
                    "-" => "-",
                    "*" => "*",
                    "/" => "/",
                    "and" => "and",
                    "or" => "or",
                    _ => unreachable!("unimplemented binary operator translation"),
                },
                Box::new(luify_expr_val(s, out, *rhs)),
            );
            fulfill_target(s, out, binop, target)
        }
        par::Expr::Block(par::Block { stmts, ret }) => {
            let mut body_stmts = Vec::new();

            luify_block_body(s, &mut body_stmts, stmts, ret.map(|r| *r), target);
            out.push(Stmt::Do(body_stmts));
        }
    }
}

fn luify_stmt<'s>(s: &mut State, out: &mut Stmts<'s>, stmt: par::Stmt<'s>) {
    match stmt {
        par::Stmt::Error => unreachable!("error node in ast"),
        par::Stmt::Expr(par::Expr::Number(..)) | par::Stmt::Expr(par::Expr::Ident(..)) => {}
        par::Stmt::Expr(expr) => {
            let id = s.gen_id();
            out.push(Stmt::Local(id, None));
            luify_expr(s, out, expr, Target::Assign(id));
        }
        par::Stmt::Let(id, val) => {
            let id = luify_ident(id);
            out.push(Stmt::Local(id, None));
            luify_expr(s, out, val, Target::Assign(id));
        }
    }
}

pub fn luify_chunk<'s>(
    s: &mut State,
    out: &mut Stmts<'s>,
    par::Block { stmts, ret }: par::Block<'s>,
) {
    luify_block_body(s, out, stmts, ret.map(|r| *r), Target::Return);
}
