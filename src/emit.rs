use std::io::Write;

use crate::luish;
use crate::par;

const INDENT_WIDTH: u16 = 4;

fn emit_newline(out: &mut impl Write, indent: u16) -> std::io::Result<()> {
    write!(out, "\n{0: <1$}", "", (indent * INDENT_WIDTH) as usize)
}

fn emit_ident<'s>(out: &mut impl Write, id: luish::Ident) -> std::io::Result<()> {
    match id {
        luish::Ident::Id(id) => write!(out, "_nm_{id}")?,
        luish::Ident::Str(str) => write!(out, "{str}")?,
    }
    Ok(())
}

fn emit_call<'s>(
    out: &mut impl Write,
    indent: u16,
    fn_expr: luish::Expr,
    args: Vec<luish::Expr>,
) -> std::io::Result<()> {
    emit_expr(out, indent, fn_expr)?;
    write!(out, "(")?;

    let args_len = args.len();
    for (i, arg) in args.into_iter().enumerate() {
        emit_expr(out, indent, arg)?;
        if i < args_len - 1 {
            write!(out, ",")?;
        }
    }
    write!(out, ")")?;
    Ok(())
}

fn emit_expr<'s>(out: &mut impl Write, mut indent: u16, expr: luish::Expr) -> std::io::Result<()> {
    match expr {
        luish::Expr::Nil => write!(out, "nil")?,
        luish::Expr::Ident(id) => emit_ident(out, id)?,
        luish::Expr::UnaryOp(op, expr) => {
            write!(out, "{op} ")?;
            emit_expr(out, indent, *expr)?;
        }
        luish::Expr::BinaryOp(lhs, op, rhs) => {
            write!(out, "(")?;
            emit_expr(out, indent, *lhs)?;
            write!(out, " {op} ")?;
            emit_expr(out, indent, *rhs)?;
            write!(out, ")")?;
        }
        luish::Expr::Call(fn_expr, args) => emit_call(out, indent, *fn_expr, args)?,
        luish::Expr::Func(args, body) => {
            write!(out, "function (")?;
            let args_len = args.len();
            for (i, arg) in args.into_iter().enumerate() {
                emit_ident(out, arg)?;
                if i < args_len - 1 {
                    write!(out, ",")?;
                }
            }
            write!(out, ")")?;
            indent += 1;
            emit_newline(out, indent)?;

            let body_len = body.len();
            for (i, stmt) in body.into_iter().enumerate() {
                emit_stmt(out, indent, stmt)?;
                if i < body_len - 1 {
                    emit_newline(out, indent)?;
                }
            }

            indent -= 1;
            emit_newline(out, indent)?;
            write!(out, "end")?;
        }
        luish::Expr::Table(entries) => {
            write!(out, "{{")?;
            indent += 1;
            emit_newline(out, indent)?;

            let entries_len = entries.len();
            for (i, (key, val)) in entries.into_iter().enumerate() {
                if let Some(key) = key {
                    match key {
                        luish::TableKey::Ident(id) => emit_ident(out, id)?,
                        luish::TableKey::Expr(expr) => {
                            write!(out, "[")?;
                            emit_expr(out, indent, expr)?;
                            write!(out, "]")?;
                        }
                    }
                    write!(out, " = ")?;
                }
                emit_expr(out, indent, val)?;
                if i < entries_len - 1 {
                    write!(out, ",")?;
                    emit_newline(out, indent)?;
                }
            }
            indent -= 1;
            emit_newline(out, indent)?;
            write!(out, "}}")?;
        }
        luish::Expr::String(str, par::QuoteType::Double) => write!(out, "\"{str}\"")?,
        luish::Expr::String(str, par::QuoteType::Single) => write!(out, "\'{str}\'")?,
        luish::Expr::Verbatim(str) => write!(out, "{str}")?,
    }
    Ok(())
}

pub fn emit_stmt<'s>(
    out: &mut impl Write,
    mut indent: u16,
    stmt: luish::Stmt,
) -> std::io::Result<()> {
    match stmt {
        luish::Stmt::Do(body) => {
            write!(out, "do")?;
            indent += 1;

            emit_newline(out, indent)?;
            let body_len = body.len();
            for (i, stmt) in body.into_iter().enumerate() {
                emit_stmt(out, indent, stmt)?;
                if i < body_len - 1 {
                    emit_newline(out, indent)?;
                }
            }
            indent -= 1;
            emit_newline(out, indent)?;
            write!(out, "end")?;
        }
        luish::Stmt::If { cases, else_body } => {
            let mut cases_it = cases.into_iter();
            let (cond, body) = cases_it
                .next()
                .expect("if statement must have at least one condition.");

            write!(out, "if ")?;
            emit_expr(out, indent, cond)?;
            write!(out, " then")?;
            indent += 1;

            emit_newline(out, indent)?;
            let body_len = body.len();
            for (i, stmt) in body.into_iter().enumerate() {
                emit_stmt(out, indent, stmt)?;
                if i < body_len - 1 {
                    emit_newline(out, indent)?;
                }
            }

            indent -= 1;
            emit_newline(out, indent)?;

            for (cond, body) in cases_it {
                write!(out, "elseif ")?;
                emit_expr(out, indent, cond)?;
                write!(out, " then")?;
                indent += 1;

                emit_newline(out, indent)?;
                let body_len = body.len();
                for (i, stmt) in body.into_iter().enumerate() {
                    emit_stmt(out, indent, stmt)?;
                    if i < body_len - 1 {
                        emit_newline(out, indent)?;
                    }
                }

                indent -= 1;
                emit_newline(out, indent)?;
            }

            if let Some(else_body) = else_body {
                write!(out, "else")?;
                indent += 1;

                emit_newline(out, indent)?;
                let else_body_len = else_body.len();
                for (i, stmt) in else_body.into_iter().enumerate() {
                    emit_stmt(out, indent, stmt)?;
                    if i < else_body_len - 1 {
                        emit_newline(out, indent)?;
                    }
                }

                indent -= 1;
                emit_newline(out, indent)?;
            }
            write!(out, "end")?;
        }
        luish::Stmt::Call(fn_expr, args) => {
            emit_call(out, indent, fn_expr, args)?;
        }
        luish::Stmt::Local(id, val) => {
            write!(out, "local ")?;
            emit_ident(out, id)?;
            if let Some(val) = val {
                write!(out, " = ")?;
                emit_expr(out, indent, val)?;
            }
        }
        luish::Stmt::Assign(id, val) => {
            emit_ident(out, id)?;
            write!(out, " = ")?;
            emit_expr(out, indent, val)?;
        }
        luish::Stmt::Return(val) => {
            write!(out, "return ")?;
            emit_expr(out, indent, val)?;
        }
    }
    Ok(())
}

pub fn emit_chunk<'s>(out: &mut impl Write, chunk: par::Block<'s>) -> std::io::Result<()> {
    let mut luify_state = luish::State::default();
    let mut stmts = Vec::new();
    luish::luify_chunk(&mut luify_state, &mut stmts, chunk);

    // TODO: check luify_state for errors.

    for stmt in stmts.into_iter() {
        emit_stmt(out, 0, stmt)?;
        emit_newline(out, 0)?;
    }

    Ok(())
}
