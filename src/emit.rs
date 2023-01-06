use std::io::Write;

use crate::luish;
use crate::par;

pub fn emit_ident<'s>(out: &mut impl Write, id: luish::Ident) -> std::io::Result<()> {
    match id {
        luish::Ident::Id(id) => write!(out, "_nm_{id}")?,
        luish::Ident::Str(str) => write!(out, "{str}")?,
    }
    Ok(())
}

pub fn emit_call<'s>(
    out: &mut impl Write,
    fn_expr: luish::Expr,
    args: Vec<luish::Expr>,
) -> std::io::Result<()> {
    emit_expr(out, fn_expr)?;
    write!(out, "(")?;
    let args_len = args.len();
    for (i, arg) in args.into_iter().enumerate() {
        emit_expr(out, arg)?;
        if i < args_len - 1 {
            write!(out, ",")?;
        }
    }
    write!(out, ")")?;
    Ok(())
}

pub fn emit_expr<'s>(out: &mut impl Write, expr: luish::Expr) -> std::io::Result<()> {
    match expr {
        luish::Expr::Nil => write!(out, "nil")?,
        luish::Expr::Ident(id) => emit_ident(out, id)?,
        luish::Expr::UnaryOp(op, expr) => {
            write!(out, "{op} ")?;
            emit_expr(out, *expr)?;
        }
        luish::Expr::BinaryOp(lhs, op, rhs) => {
            write!(out, "(")?;
            emit_expr(out, *lhs)?;
            write!(out, " {op} ")?;
            emit_expr(out, *rhs)?;
            write!(out, ")")?;
        }
        luish::Expr::Call(fn_expr, args) => emit_call(out, *fn_expr, args)?,
        luish::Expr::Func(args, body) => {
            write!(out, "function (")?;
            let args_len = args.len();
            for (i, arg) in args.into_iter().enumerate() {
                emit_ident(out, arg)?;
                if i < args_len - 1 {
                    write!(out, ",")?;
                }
            }
            write!(out, ")\n")?;
            for stmt in body.into_iter() {
                emit_stmt(out, stmt)?;
            }
            write!(out, "end")?;
        }
        luish::Expr::Table(entries) => {
            write!(out, "{{\n")?;
            let entries_len = entries.len();
            for (i, (key, val)) in entries.into_iter().enumerate() {
                match key {
                    luish::TableKey::Ident(id) => emit_ident(out, id)?,
                    luish::TableKey::Expr(expr) => {
                        write!(out, "[")?;
                        emit_expr(out, expr)?;
                        write!(out, "]")?;
                    }
                }
                write!(out, " = ")?;
                emit_expr(out, val)?;
                if i < entries_len - 1 {
                    write!(out, ",\n")?;
                } else {
                    write!(out, "\n")?;
                }
            }
            write!(out, "}}")?;
        }
        luish::Expr::Verbatim(str) => write!(out, "{str}")?,
    }
    Ok(())
}

pub fn emit_stmt<'s>(out: &mut impl Write, stmt: luish::Stmt) -> std::io::Result<()> {
    match stmt {
        luish::Stmt::Do(body) => {
            write!(out, "do\n")?;
            for stmt in body {
                emit_stmt(out, stmt)?;
            }
            write!(out, "end\n")?;
        }
        luish::Stmt::Call(fn_expr, args) => {
            emit_call(out, fn_expr, args)?;
            write!(out, "\n")?;
        }
        luish::Stmt::Local(id, val) => {
            write!(out, "local ")?;
            emit_ident(out, id)?;
            if let Some(val) = val {
                write!(out, " = ")?;
                emit_expr(out, val)?;
            }
            write!(out, "\n")?;
        }
        luish::Stmt::Assign(id, val) => {
            emit_ident(out, id)?;
            write!(out, " = ")?;
            emit_expr(out, val)?;
            write!(out, "\n")?;
        }
        luish::Stmt::Return(val) => {
            write!(out, "return ")?;
            emit_expr(out, val)?;
            write!(out, "\n")?;
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
        emit_stmt(out, stmt)?;
    }

    Ok(())
}
