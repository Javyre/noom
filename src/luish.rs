use crate::err::Error;
use crate::par;

#[derive(Clone, Copy)]
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

fn luify_ident<'s>(id: par::Ident<'s>) -> Ident<'s> {
    Ident::Str(id.span.fragment())
}

fn luify_block_body<'s>(
    s: &mut State,
    out: &mut Stmts<'s>,
    stmts: Vec<par::Stmt<'s>>,
    ret: Option<par::Expr<'s>>,
) -> Expr<'s> {
    let mut ret_expr = Expr::Nil;

    for stmt in stmts.into_iter() {
        luify_stmt(s, out, stmt);
    }
    if let Some(ret) = ret {
        let ret_id = s.gen_id();
        out.push(Stmt::Local(ret_id, None));

        let ret = luify_expr(s, out, ret);
        out.push(Stmt::Assign(ret_id, ret));
        ret_expr = Expr::Ident(ret_id);
    }
    ret_expr
}

fn luify_expr<'s>(s: &mut State, out: &mut Stmts<'s>, expr: par::Expr<'s>) -> Expr<'s> {
    match expr {
        par::Expr::Error => unreachable!("error node in ast"),
        par::Expr::Func(args, body) => {
            let mut body_out = Vec::new();
            let ret = luify_expr(s, &mut body_out, *body);
            body_out.push(Stmt::Return(ret));

            Expr::Func(
                args.into_iter().map(|id| luify_ident(id)).collect(),
                body_out,
            )
        }
        par::Expr::Call(fn_expr, args) => {
            let fn_expr = luify_expr(s, out, *fn_expr);
            let args = args
                .into_iter()
                .map(|arg| luify_expr(s, out, arg))
                .collect();

            Expr::Call(Box::new(fn_expr), args)
        }
        par::Expr::Ident(id) => Expr::Ident(luify_ident(id)),
        par::Expr::Table(par::Table { entries, .. }) => Expr::Table(
            entries
                .into_iter()
                .map(|(key, val)| {
                    (
                        match key {
                            par::TableKey::Expr(key) => TableKey::Expr(luify_expr(s, out, key)),
                            par::TableKey::Ident(key) => TableKey::Ident(luify_ident(key)),
                        },
                        luify_expr(s, out, val),
                    )
                })
                .collect(),
        ),
        par::Expr::Number(par::Number { span }) => Expr::Verbatim(span.fragment()),
        par::Expr::UnaryOp(op, expr) => Expr::UnaryOp(
            match *op.fragment() {
                "!" => "not",
                "-" => "-",
                _ => unimplemented!("unimplemented unary operator translation"),
            },
            Box::new(luify_expr(s, out, *expr)),
        ),
        par::Expr::BinaryOp(lhs, op, rhs) => Expr::BinaryOp(
            Box::new(luify_expr(s, out, *lhs)),
            match *op.fragment() {
                "+" => "+",
                "-" => "-",
                "*" => "*",
                "/" => "/",
                "and" => "and",
                "or" => "or",
                _ => unreachable!("unimplemented binary operator translation"),
            },
            Box::new(luify_expr(s, out, *rhs)),
        ),
        par::Expr::Block(par::Block { stmts, ret }) => {
            let mut body_stmts = Vec::new();
            let ret_expr = luify_block_body(s, &mut body_stmts, stmts, ret.map(|r| *r));

            out.push(Stmt::Do(body_stmts));

            ret_expr
        }
    }
}

fn luify_stmt<'s>(s: &mut State, out: &mut Stmts<'s>, stmt: par::Stmt<'s>) {
    match stmt {
        par::Stmt::Error => unreachable!("error node in ast"),
        par::Stmt::Expr(par::Expr::Number(..)) | par::Stmt::Expr(par::Expr::Ident(..)) => {}
        par::Stmt::Expr(expr) => {
            let expr = luify_expr(s, out, expr);
            match expr {
                Expr::Nil => {}
                expr => out.push(Stmt::Local(s.gen_id(), Some(expr))),
            }
        }
        par::Stmt::Let(id, val) => {
            let val = luify_expr(s, out, val);
            out.push(Stmt::Local(luify_ident(id), Some(val)));
        }
    }
}

pub fn luify_chunk<'s>(
    s: &mut State,
    out: &mut Stmts<'s>,
    par::Block { stmts, ret }: par::Block<'s>,
) {
    let ret = luify_block_body(s, out, stmts, ret.map(|r| *r));
    match ret {
        Expr::Nil => {}
        ret => out.push(Stmt::Return(ret)),
    }
}
