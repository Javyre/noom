use std::cell::RefCell;

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
    Assign(Expr<'s>, Expr<'s>),
    Return(Option<Expr<'s>>),
    Do(Vec<Stmt<'s>>),
    If {
        cases: Vec<(Expr<'s>, Vec<Stmt<'s>>)>,
        else_body: Option<Vec<Stmt<'s>>>,
    },
    For {
        it_var: Ident<'s>,
        it: Expr<'s>,
        body: Vec<Stmt<'s>>,
    },
    Break,
}

pub enum TableKey<'s> {
    Ident(Ident<'s>),
    Expr(Expr<'s>),
}

pub enum Expr<'s> {
    Nil,
    String(&'s str, par::QuoteType),
    Ident(Ident<'s>),
    Path(Box<Expr<'s>>, Vec<Ident<'s>>),
    Verbatim(&'s str),
    Table(Vec<(Option<TableKey<'s>>, Expr<'s>)>),
    Func(Vec<Ident<'s>>, Vec<Stmt<'s>>),
    Call(Box<Expr<'s>>, Vec<Expr<'s>>),
    UnaryOp(&'s str, Box<Expr<'s>>),
    BinaryOp(Box<Expr<'s>>, &'s str, Box<Expr<'s>>),
}

#[derive(Clone, Copy)]
enum Target<'s, 't> {
    Assign(Ident<'s>),
    Value(&'t RefCell<Option<Expr<'s>>>),
    Return,
    None,
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
    } else {
        fulfill_target(s, out, Expr::Nil, target);
    }
}

fn luify_expr_stmts<'s, 't>(
    s: &mut State,
    out: &mut Stmts<'s>,
    expr: par::Expr<'s>,
    target: Target<'s, 't>,
) {
    match expr {
        par::Expr::Block(par::Block { stmts, ret }) => {
            luify_block_body(s, out, stmts, ret.map(|e| *e), target)
        }
        expr => luify_expr(s, out, expr, target),
    }
}

// default target fulfillment
fn fulfill_target<'s, 't>(
    s: &mut State,
    out: &mut Stmts<'s>,
    expr: Expr<'s>,
    target: Target<'s, 't>,
) {
    match target {
        Target::Return => match expr {
            Expr::Nil => out.push(Stmt::Return(None)),
            expr => out.push(Stmt::Return(Some(expr))),
        },
        Target::Assign(target) => out.push(Stmt::Assign(Expr::Ident(target), expr)),
        Target::Value(val) => *val.borrow_mut() = Some(expr),
        Target::None => match expr {
            Expr::Nil | Expr::String(..) | Expr::Verbatim(..) | Expr::Ident(..) => {}
            Expr::Call(fn_expr, args) => out.push(Stmt::Call(*fn_expr, args)),
            expr => out.push(Stmt::Local(s.gen_id(), Some(expr))),
        },
    }
}

fn luify_expr_val<'s>(s: &mut State, out: &mut Stmts<'s>, expr: par::Expr<'s>) -> Expr<'s> {
    let expr_out = RefCell::new(None);
    luify_expr(s, out, expr, Target::Value(&expr_out));
    expr_out.into_inner().unwrap()
}

fn luify_expr<'s, 't>(
    s: &mut State,
    out: &mut Stmts<'s>,
    expr: par::Expr<'s>,
    target: Target<'s, 't>,
) {
    match expr {
        par::Expr::Error => unreachable!("error node in ast"),
        par::Expr::Func(args, body, _ret_ty) => {
            let mut body_out = Vec::new();
            luify_expr_stmts(s, &mut body_out, *body, Target::Return);

            fulfill_target(
                s,
                out,
                Expr::Func(
                    args.into_iter().map(|(id, _ty)| luify_ident(id)).collect(),
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
        par::Expr::Path(expr, ids) => {
            let expr = luify_expr_val(s, out, *expr);
            let ids = ids.into_iter().map(|id| luify_ident(id)).collect();

            fulfill_target(s, out, Expr::Path(Box::new(expr), ids), target)
        }
        par::Expr::Table(par::Table { entries, .. }) => {
            let table = Expr::Table(
                entries
                    .into_iter()
                    .map(|(key, val)| {
                        (
                            key.map(|key| match key {
                                par::TableKey::Expr(key) => {
                                    TableKey::Expr(luify_expr_val(s, out, key))
                                }
                                par::TableKey::Ident(key) => TableKey::Ident(luify_ident(key)),
                            }),
                            luify_expr_val(s, out, val),
                        )
                    })
                    .collect(),
            );
            fulfill_target(s, out, table, target);
        }
        par::Expr::String(span, q) => {
            fulfill_target(s, out, Expr::String(span.fragment(), q), target)
        }
        par::Expr::Tag(span) => fulfill_target(
            s,
            out,
            Expr::String(span.fragment(), par::QuoteType::Single),
            target,
        ),
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

            // need to generate a temp variable to avoid binding to a shadowed var
            let target = match (&ret, target) {
                (&Some(_), Target::Value(val)) => {
                    let id = s.gen_id();
                    *val.borrow_mut() = Some(Expr::Ident(id));
                    out.push(Stmt::Local(id, None));
                    Target::Assign(id)
                }
                (_, target) => target,
            };

            luify_block_body(s, &mut body_stmts, stmts, ret.map(|r| *r), target);
            match body_stmts.len() {
                0 => {}
                1 => match body_stmts.pop().unwrap() {
                    Stmt::Local(_id, None) => {}
                    Stmt::Local(Ident::Str(_), val @ Some(_)) => {
                        let id = s.gen_id();
                        out.push(Stmt::Local(id, val));
                    }

                    stmt @ Stmt::Local(Ident::Id(_), Some(_)) | stmt => out.push(stmt),
                },
                _ => out.push(Stmt::Do(body_stmts)),
            }
        }
        par::Expr::If { cases, else_body } => {
            let target = match target {
                Target::Value(val) => {
                    let id = s.gen_id();
                    *val.borrow_mut() = Some(Expr::Ident(id));
                    out.push(Stmt::Local(id, None));
                    Target::Assign(id)
                }
                target => target,
            };

            let mut cases = cases
                .into_iter()
                .map(|(cond, body)| {
                    let cond = luify_expr_val(s, out, cond);

                    let mut body_out = Vec::new();
                    luify_expr_stmts(s, &mut body_out, body, target);
                    (cond, body_out)
                })
                .collect::<Vec<_>>();

            let else_body_out = else_body.map(|else_body| {
                let mut else_body_out = Vec::new();
                luify_expr_stmts(s, &mut else_body_out, *else_body, target);
                else_body_out
            });

            let else_body_out = match else_body_out {
                Some(else_body_out) => {
                    if matches!(else_body_out.as_slice(), [Stmt::If { .. }]) {
                        if let Stmt::If {
                            cases: mut more_cases,
                            else_body,
                        } = else_body_out.into_iter().next().unwrap()
                        {
                            cases.append(&mut more_cases);
                            else_body
                        } else {
                            unreachable!();
                        }
                    } else {
                        Some(else_body_out)
                    }
                }
                None => None,
            };

            let else_body_out = match (else_body_out, target) {
                (None, Target::Return) => Some(vec![Stmt::Return(None)]),
                (None, _) => None,
                (Some(else_body_out), _) => Some(else_body_out),
            };

            out.push(Stmt::If {
                cases,
                else_body: else_body_out,
            })
        }
    }
}

fn luify_stmt<'s>(s: &mut State, out: &mut Stmts<'s>, stmt: par::Stmt<'s>) {
    match stmt {
        par::Stmt::Error => unreachable!("error node in ast"),
        par::Stmt::Expr(expr) => luify_expr(s, out, expr, Target::None),
        par::Stmt::Let(id, _ty, val) => {
            let id = luify_ident(id);
            match val {
                val @ par::Expr::Func(..) => {
                    out.push(Stmt::Local(id, None));
                    luify_expr(s, out, val, Target::Assign(id));
                }
                val => {
                    let val = luify_expr_val(s, out, val);
                    out.push(Stmt::Local(id, Some(val)));
                }
            }
        }
        par::Stmt::Assign(targ_expr, val) => {
            let targ_expr = luify_expr_val(s, out, targ_expr);
            let val = luify_expr_val(s, out, val);
            out.push(Stmt::Assign(targ_expr, val));
        }
        par::Stmt::For {
            it_var, it, body, ..
        } => {
            let it_var = luify_ident(it_var);
            let it = luify_expr_val(s, out, it);

            let mut body_out = Vec::new();
            luify_expr_stmts(s, &mut body_out, body, Target::None);

            out.push(Stmt::For {
                it_var,
                it,
                body: body_out,
            })
        }
        par::Stmt::Return(Some(val)) => luify_expr(s, out, val, Target::Return),
        par::Stmt::Return(None) => out.push(Stmt::Return(None)),
        par::Stmt::Break => out.push(Stmt::Break),
    }
}

pub fn luify_chunk<'s>(
    s: &mut State,
    out: &mut Stmts<'s>,
    par::Block { stmts, ret }: par::Block<'s>,
) {
    luify_block_body(s, out, stmts, ret.map(|r| *r), Target::Return);
}
