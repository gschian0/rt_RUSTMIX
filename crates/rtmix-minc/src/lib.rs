//! Minc subset: numbers, variables, arithmetic, `for` / `if`, and the phase-1 calls.
//! The interpreter runs on the control thread and only schedules notes.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use rtmix_core::{Engine, Pfield, SceneNote};
use rtmix_dsp::{
    install_table, make_harmonics, schedule_chant, schedule_delay_prepared, schedule_filter,
    schedule_granular,
    schedule_formant, schedule_fof, schedule_lpc, schedule_mix, schedule_reverb_prepared,
    schedule_speak, schedule_wavetable, ChantNote, DelayNote, FilterKind, FilterNote, FofNote,
    FormantNote, GranularNote, LpcNote, MixNote, ReverbNote, SpeakNote, WavetableNote,
};
use rtmix_spat::{
    schedule_binaural, schedule_decode, schedule_rotate, schedule_spat, BinauralNote, DecodeNote,
    RotateNote, SpatNote, SpeakerLayout,
};

#[derive(Clone, Debug)]
pub struct MincError {
    pub message: String,
    pub line: usize,
}

#[derive(Clone, Debug)]
enum Value {
    Num(f64),
    Str(String),
}

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    Str(String),
    Var(String),
    Unary { op: char, rhs: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Assign { name: String, rhs: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
}

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Clone, Debug)]
enum Stmt {
    Expr(Expr),
    Block(Vec<Stmt>),
    For {
        init: Expr,
        cond: Expr,
        step: Expr,
        body: Box<Stmt>,
    },
    If {
        cond: Expr,
        then_body: Box<Stmt>,
        else_body: Option<Box<Stmt>>,
    },
}

#[derive(Clone, Debug)]
enum Tok {
    Num(f64, usize),
    Str(String, usize),
    Ident(String, usize),
    Sym(char, usize),
    Semi(usize),
    Newline(usize),
    Eof(usize),
}

struct Lexer<'a> {
    src: &'a [u8],
    i: usize,
    line: usize,
    paren: i32,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            i: 0,
            line: 1,
            paren: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        if c == b'\n' {
            self.line += 1;
        }
        Some(c)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if c == b' ' || c == b'\t' || c == b'\r' {
                    self.bump();
                } else {
                    break;
                }
            }
            if self.peek() == Some(b'/') && self.src.get(self.i + 1) == Some(&b'/') {
                while let Some(c) = self.peek() {
                    if c == b'\n' {
                        break;
                    }
                    self.bump();
                }
                continue;
            }
            if self.peek() == Some(b'/') && self.src.get(self.i + 1) == Some(&b'*') {
                self.bump();
                self.bump();
                while self.i < self.src.len() {
                    if self.peek() == Some(b'*') && self.src.get(self.i + 1) == Some(&b'/') {
                        self.bump();
                        self.bump();
                        break;
                    }
                    self.bump();
                }
                continue;
            }
            break;
        }
    }

    fn tokenize(mut self) -> Vec<Tok> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            let Some(c) = self.peek() else {
                out.push(Tok::Eof(self.line));
                break;
            };
            let line = self.line;
            if c == b'\n' {
                self.bump();
                if self.paren == 0 {
                    out.push(Tok::Newline(line));
                }
                continue;
            }
            if c == b'"' {
                self.bump();
                let mut s = String::new();
                while let Some(ch) = self.bump() {
                    if ch == b'"' {
                        break;
                    }
                    if ch == b'\\' {
                        if let Some(n) = self.bump() {
                            s.push(n as char);
                        }
                    } else {
                        s.push(ch as char);
                    }
                }
                out.push(Tok::Str(s, line));
                continue;
            }
            if c.is_ascii_digit() || (c == b'.' && self.src.get(self.i + 1).copied().unwrap_or(0).is_ascii_digit())
            {
                let start = self.i;
                while self.peek().unwrap_or(0).is_ascii_digit() {
                    self.bump();
                }
                if self.peek() == Some(b'.') {
                    self.bump();
                    while self.peek().unwrap_or(0).is_ascii_digit() {
                        self.bump();
                    }
                }
                if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                    self.bump();
                    if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                        self.bump();
                    }
                    while self.peek().unwrap_or(0).is_ascii_digit() {
                        self.bump();
                    }
                }
                let text = core::str::from_utf8(&self.src[start..self.i]).unwrap_or("0");
                let n = text.parse::<f64>().unwrap_or(0.0);
                out.push(Tok::Num(n, line));
                continue;
            }
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = self.i;
                while self.peek().unwrap_or(0).is_ascii_alphanumeric() || self.peek() == Some(b'_') {
                    self.bump();
                }
                let text = core::str::from_utf8(&self.src[start..self.i]).unwrap_or("");
                out.push(Tok::Ident(text.to_string(), line));
                continue;
            }
            self.bump();
            let two = match c {
                b'=' if self.peek() == Some(b'=') => Some('='),
                b'!' if self.peek() == Some(b'=') => Some('!'),
                b'<' if self.peek() == Some(b'=') => Some('<'),
                b'>' if self.peek() == Some(b'=') => Some('>'),
                b'&' if self.peek() == Some(b'&') => Some('&'),
                b'|' if self.peek() == Some(b'|') => Some('|'),
                _ => None,
            };
            if let Some(kind) = two {
                self.bump();
                let sym = match kind {
                    '=' => 'E',
                    '!' => 'N',
                    '<' => 'L',
                    '>' => 'G',
                    '&' => 'A',
                    '|' => 'O',
                    _ => '?',
                };
                out.push(Tok::Sym(sym, line));
                continue;
            }
            if c == b'(' {
                self.paren += 1;
            } else if c == b')' {
                self.paren -= 1;
            }
            if c == b';' {
                out.push(Tok::Semi(line));
            } else {
                out.push(Tok::Sym(c as char, line));
            }
        }
        out
    }
}

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn line(&self) -> usize {
        match self.toks.get(self.i) {
            Some(Tok::Num(_, l))
            | Some(Tok::Str(_, l))
            | Some(Tok::Ident(_, l))
            | Some(Tok::Sym(_, l))
            | Some(Tok::Semi(l))
            | Some(Tok::Newline(l))
            | Some(Tok::Eof(l)) => *l,
            None => 1,
        }
    }

    fn err(&self, message: &str) -> MincError {
        MincError {
            message: message.to_string(),
            line: self.line(),
        }
    }

    fn bump(&mut self) -> Tok {
        let tok = self.toks.get(self.i).cloned().unwrap_or(Tok::Eof(self.line()));
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        tok
    }

    fn current(&self) -> &Tok {
        self.toks.get(self.i).unwrap_or(&Tok::Eof(1))
    }

    fn skip_newlines(&mut self) {
        while matches!(self.current(), Tok::Newline(_)) {
            self.bump();
        }
    }

    fn parse(src: &str) -> Result<Vec<Stmt>, MincError> {
        let toks = Lexer::new(src).tokenize();
        let mut p = Parser { toks, i: 0 };
        let mut stmts = Vec::new();
        loop {
            p.skip_newlines();
            while matches!(p.current(), Tok::Semi(_)) {
                p.bump();
                p.skip_newlines();
            }
            if matches!(p.current(), Tok::Eof(_)) {
                break;
            }
            stmts.push(p.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, MincError> {
        self.skip_newlines();
        match self.current() {
            Tok::Ident(name, _) if name == "for" => self.parse_for(),
            Tok::Ident(name, _) if name == "if" => self.parse_if(),
            Tok::Sym('{', _) => self.parse_block(),
            Tok::Semi(_) => {
                self.bump();
                Ok(Stmt::Block(Vec::new()))
            }
            _ => {
                let expr = self.parse_expr()?;
                if matches!(self.current(), Tok::Semi(_) | Tok::Newline(_)) {
                    self.bump();
                }
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_block(&mut self) -> Result<Stmt, MincError> {
        if !matches!(self.current(), Tok::Sym('{', _)) {
            return Err(self.err("expected '{'"));
        }
        self.bump();
        let mut body = Vec::new();
        loop {
            self.skip_newlines();
            while matches!(self.current(), Tok::Semi(_)) {
                self.bump();
                self.skip_newlines();
            }
            if matches!(self.current(), Tok::Sym('}', _)) {
                self.bump();
                break;
            }
            if matches!(self.current(), Tok::Eof(_)) {
                return Err(self.err("unclosed block"));
            }
            body.push(self.parse_stmt()?);
        }
        Ok(Stmt::Block(body))
    }

    fn parse_for(&mut self) -> Result<Stmt, MincError> {
        self.bump();
        self.skip_newlines();
        if !matches!(self.current(), Tok::Sym('(', _)) {
            return Err(self.err("expected '(' after for"));
        }
        self.bump();
        let init = self.parse_expr()?;
        if !matches!(self.current(), Tok::Semi(_)) {
            return Err(self.err("expected ';' in for"));
        }
        self.bump();
        let cond = self.parse_expr()?;
        if !matches!(self.current(), Tok::Semi(_)) {
            return Err(self.err("expected ';' in for"));
        }
        self.bump();
        let step = self.parse_expr()?;
        if !matches!(self.current(), Tok::Sym(')', _)) {
            return Err(self.err("expected ')' after for"));
        }
        self.bump();
        self.skip_newlines();
        let body = self.parse_stmt()?;
        Ok(Stmt::For {
            init,
            cond,
            step,
            body: Box::new(body),
        })
    }

    fn parse_if(&mut self) -> Result<Stmt, MincError> {
        self.bump();
        self.skip_newlines();
        if !matches!(self.current(), Tok::Sym('(', _)) {
            return Err(self.err("expected '(' after if"));
        }
        self.bump();
        let cond = self.parse_expr()?;
        if !matches!(self.current(), Tok::Sym(')', _)) {
            return Err(self.err("expected ')' after if"));
        }
        self.bump();
        self.skip_newlines();
        let then_body = self.parse_stmt()?;
        self.skip_newlines();
        let else_body = if matches!(self.current(), Tok::Ident(name, _) if name == "else") {
            self.bump();
            self.skip_newlines();
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        Ok(Stmt::If {
            cond,
            then_body: Box::new(then_body),
            else_body,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, MincError> {
        if let Tok::Ident(name, _) = self.current().clone() {
            if matches!(self.toks.get(self.i + 1), Some(Tok::Sym('=', _))) {
                self.bump();
                self.bump();
                let rhs = self.parse_expr()?;
                return Ok(Expr::Assign {
                    name,
                    rhs: Box::new(rhs),
                });
            }
        }
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, MincError> {
        let mut lhs = self.parse_and()?;
        while matches!(self.current(), Tok::Sym('O', _)) {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary {
                op: BinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, MincError> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.current(), Tok::Sym('A', _)) {
            self.bump();
            let rhs = self.parse_cmp()?;
            lhs = Expr::Binary {
                op: BinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr, MincError> {
        let mut lhs = self.parse_sum()?;
        loop {
            let op = match self.current() {
                Tok::Sym('E', _) => BinOp::Eq,
                Tok::Sym('N', _) => BinOp::Ne,
                Tok::Sym('<', _) => BinOp::Lt,
                Tok::Sym('>', _) => BinOp::Gt,
                Tok::Sym('L', _) => BinOp::Le,
                Tok::Sym('G', _) => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_sum()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_sum(&mut self) -> Result<Expr, MincError> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.current() {
                Tok::Sym('+', _) => BinOp::Add,
                Tok::Sym('-', _) => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_term()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, MincError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.current() {
                Tok::Sym('*', _) => BinOp::Mul,
                Tok::Sym('/', _) => BinOp::Div,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, MincError> {
        if matches!(self.current(), Tok::Sym('-', _) | Tok::Sym('!', _) | Tok::Sym('+', _)) {
            let op = match self.bump() {
                Tok::Sym(op, _) => op,
                _ => '+',
            };
            let rhs = self.parse_unary()?;
            if op == '+' {
                return Ok(rhs);
            }
            return Ok(Expr::Unary {
                op,
                rhs: Box::new(rhs),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, MincError> {
        match self.current().clone() {
            Tok::Num(n, _) => {
                self.bump();
                Ok(Expr::Num(n))
            }
            Tok::Str(s, _) => {
                self.bump();
                Ok(Expr::Str(s))
            }
            Tok::Ident(name, _) => {
                self.bump();
                if matches!(self.current(), Tok::Sym('(', _)) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.current(), Tok::Sym(')', _)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.current(), Tok::Sym(',', _)) {
                                self.bump();
                                continue;
                            }
                            break;
                        }
                    }
                    if !matches!(self.current(), Tok::Sym(')', _)) {
                        return Err(self.err("expected ')'"));
                    }
                    self.bump();
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Tok::Sym('(', _) => {
                self.bump();
                let expr = self.parse_expr()?;
                if !matches!(self.current(), Tok::Sym(')', _)) {
                    return Err(self.err("expected ')'"));
                }
                self.bump();
                Ok(expr)
            }
            _ => Err(self.err("expected expression")),
        }
    }
}

struct Interp<'a> {
    engine: &'a mut Engine,
    vars: BTreeMap<String, Value>,
    assets: Vec<String>,
}

pub fn execute(score: &str, engine: &mut Engine) -> Result<Vec<String>, MincError> {
    let program = Parser::parse(score)?;
    let mut interp = Interp {
        engine,
        vars: BTreeMap::new(),
        assets: Vec::new(),
    };
    for stmt in &program {
        interp.exec_stmt(stmt)?;
    }
    Ok(interp.assets)
}

impl Interp<'_> {
    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<(), MincError> {
        match stmt {
            Stmt::Expr(expr) => {
                self.eval(expr)?;
                Ok(())
            }
            Stmt::Block(body) => {
                for stmt in body {
                    self.exec_stmt(stmt)?;
                }
                Ok(())
            }
            Stmt::For { init, cond, step, body } => {
                self.eval(init)?;
                for _ in 0..100_000 {
                    if !truthy(&self.eval(cond)?) {
                        break;
                    }
                    self.exec_stmt(body)?;
                    self.eval(step)?;
                }
                Ok(())
            }
            Stmt::If { cond, then_body, else_body } => {
                if truthy(&self.eval(cond)?) {
                    self.exec_stmt(then_body)
                } else if let Some(else_body) = else_body {
                    self.exec_stmt(else_body)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Value, MincError> {
        match expr {
            Expr::Num(n) => Ok(Value::Num(*n)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Var(name) => self
                .vars
                .get(name)
                .cloned()
                .ok_or_else(|| MincError {
                    message: format!("unknown variable {name}"),
                    line: 0,
                }),
            Expr::Unary { op, rhs } => {
                let value = self.eval(rhs)?;
                let v = self.num(&value)?;
                Ok(Value::Num(match op {
                    '-' => -v,
                    '!' => {
                        if v == 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => v,
                }))
            }
            Expr::Binary { op, lhs, rhs } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let l = truthy(&self.eval(lhs)?);
                    let r = truthy(&self.eval(rhs)?);
                    let v = match op {
                        BinOp::And => l && r,
                        BinOp::Or => l || r,
                        _ => false,
                    };
                    return Ok(Value::Num(if v { 1.0 } else { 0.0 }));
                }
                let left = self.eval(lhs)?;
                let right = self.eval(rhs)?;
                let l = self.num(&left)?;
                let r = self.num(&right)?;
                let v = match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Eq => {
                        if (l - r).abs() < 1.0e-9 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Ne => {
                        if (l - r).abs() >= 1.0e-9 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Lt => {
                        if l < r {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Gt => {
                        if l > r {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Le => {
                        if l <= r {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Ge => {
                        if l >= r {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::And | BinOp::Or => 0.0,
                };
                Ok(Value::Num(v))
            }
            Expr::Assign { name, rhs } => {
                let v = self.eval(rhs)?;
                self.vars.insert(name.clone(), v.clone());
                Ok(v)
            }
            Expr::Call { name, args } => self.call(name, args),
        }
    }

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<Value, MincError> {
        let mut values = Vec::new();
        for arg in args {
            values.push(self.eval(arg)?);
        }
        match name {
            "rtsetparams" => {
                let sr = self.num_at(&values, 0)? as u32;
                let ch = self.num_at(&values, 1)? as usize;
                let block = self.num_at(&values, 2)? as usize;
                self.engine.set_params(sr, ch, block.max(1));
                Ok(Value::Num(0.0))
            }
            "bus_config" => {
                let inst = self.str_at(&values, 0)?;
                let bus = self.str_at(&values, 1)?;
                self.engine.bus_config(&inst, &bus).map_err(|e| MincError {
                    message: e.message.to_string(),
                    line: 0,
                })?;
                Ok(Value::Num(0.0))
            }
            "spat_config" => {
                let kind = self.str_at(&values, 0)?;
                if kind != "ambix" {
                    return Err(MincError {
                        message: format!("unsupported spat format {kind}"),
                        line: 0,
                    });
                }
                let order = self.num_at(&values, 1)? as usize;
                if order == 0 || order > rtmix_spat::max_order() {
                    return Err(MincError {
                        message: format!("ambisonic order {order} is not enabled"),
                        line: 0,
                    });
                }
                self.engine.set_order(order as u8).map_err(|e| MincError {
                    message: e.message.to_string(),
                    line: 0,
                })?;
                Ok(Value::Num(order as f64))
            }
            "maketable" => {
                let (kind, size) = if values.len() >= 3 {
                    (self.str_at(&values, 2)?, self.num_at(&values, 1)? as usize)
                } else {
                    (self.str_at(&values, 0)?, self.num_at(&values, 1)? as usize)
                };
                let id = install_table(self.engine, &kind, size.max(4)).map_err(|e| MincError {
                    message: e.message.to_string(),
                    line: 0,
                })?;
                Ok(Value::Num(id as f64))
            }
            "makeharm" => {
                let size = self.num_at(&values, 0)? as usize;
                let partials = self.num_at(&values, 1)? as usize;
                let rolloff = self.num_at_opt(&values, 2)?.unwrap_or(1.0) as f32;
                let rule = self.str_at(&values, 3).unwrap_or_else(|_| "odd".to_string());
                let data = make_harmonics(size, partials, rolloff, &rule);
                let id = self.engine.add_table(data);
                Ok(Value::Num(id as f64))
            }
            "FORMANT" => {
                let vowel = self
                    .str_at(&values, 5)
                    .unwrap_or_else(|_| "a".to_string());
                let ch = vowel.chars().next().unwrap_or('a');
                let table = self.num_at_opt(&values, 6)?.map(|n| n as usize);
                schedule_formant(
                    self.engine,
                    FormantNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        freq: self.num_at(&values, 3)? as f32,
                        pan: self.num_at(&values, 4)? as f32,
                        vowel: ch,
                        table,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "CHANT" => {
                let vowels = self.str_at(&values, 8).unwrap_or_else(|_| "aeiou".to_string());
                let table = self.num_at_opt(&values, 9)?.map(|n| n as usize);
                schedule_chant(
                    self.engine,
                    ChantNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        root: self.num_at(&values, 3)? as f32,
                        pan: self.num_at(&values, 4)? as f32,
                        rate: self.num_at(&values, 5)? as f32,
                        pulses: self.num_at(&values, 6)? as usize,
                        steps: self.num_at(&values, 7)? as usize,
                        vowels: vowels.chars().collect(),
                        table,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "FOF" => {
                schedule_fof(
                    self.engine,
                    FofNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        fundamental: self.num_at(&values, 3)? as f32,
                        formant: self.num_at(&values, 4)? as f32,
                        bandwidth: self.num_at(&values, 5)? as f32,
                        jitter: self.num_at_opt(&values, 6)?.unwrap_or(0.3) as f32,
                        pan: self.num_at_opt(&values, 7)?.unwrap_or(0.5) as f32,
                        seed: self.num_at_opt(&values, 8)?.unwrap_or(1.0) as u32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "LPC" => {
                schedule_lpc(
                    self.engine,
                    LpcNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        root: self.num_at(&values, 3)? as f32,
                        pan: self.num_at_opt(&values, 4)?.unwrap_or(0.5) as f32,
                        seed: self.num_at_opt(&values, 5)?.unwrap_or(1.0) as u32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "SPEAK" => {
                schedule_speak(
                    self.engine,
                    SpeakNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        pitch: self.num_at(&values, 3)? as f32,
                        text: self.str_at(&values, 4)?,
                        grit: self.num_at_opt(&values, 5)?.unwrap_or(0.5) as f32,
                        pan: self.num_at_opt(&values, 6)?.unwrap_or(0.5) as f32,
                        seed: self.num_at_opt(&values, 7)?.unwrap_or(1.0) as u32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "WAVETABLE" => {
                let start = self.num_at(&values, 0)?;
                let dur = self.num_at(&values, 1)?;
                let amp = self.num_at(&values, 2)? as f32;
                let freq = self.num_at(&values, 3)? as f32;
                let pan = self.num_at_opt(&values, 4)?.unwrap_or(0.5) as f32;
                let table = self.num_at_opt(&values, 5)?.map(|n| n as usize);
                schedule_wavetable(
                    self.engine,
                    WavetableNote { start, dur, amp, freq, pan, table },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "GRANULAR" => {
                schedule_granular(
                    self.engine,
                    GranularNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        grain_dur: self.num_at_opt(&values, 3)?.unwrap_or(0.12) as f32,
                        density: self.num_at_opt(&values, 4)?.unwrap_or(18.0) as f32,
                        pitch: self.num_at_opt(&values, 5)?.unwrap_or(1.0) as f32,
                        jitter: self.num_at_opt(&values, 6)?.unwrap_or(0.15) as f32,
                        pan: self.num_at_opt(&values, 7)?.unwrap_or(0.5) as f32,
                        seed: self.num_at_opt(&values, 8)?.unwrap_or(1.0) as u32,
                        table: self.num_at_opt(&values, 9)?.map(|n| n as usize),
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "MIX" => {
                schedule_mix(
                    self.engine,
                    MixNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "FILTER" => {
                let kind = if let Ok(name) = self.str_at(&values, 5) {
                    match name.as_str() {
                        "highpass" => FilterKind::Highpass,
                        "bandpass" => FilterKind::Bandpass,
                        _ => FilterKind::Lowpass,
                    }
                } else {
                    FilterKind::Lowpass
                };
                schedule_filter(
                    self.engine,
                    FilterNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        cutoff: self.num_at(&values, 3)? as f32,
                        q: self.num_at(&values, 4)? as f32,
                        kind,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "DELAY" => {
                schedule_delay_prepared(
                    self.engine,
                    DelayNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        delay_secs: self.num_at(&values, 3)? as f32,
                        feedback: self.num_at(&values, 4)? as f32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "REVERB" => {
                schedule_reverb_prepared(
                    self.engine,
                    ReverbNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        amp: self.num_at(&values, 2)? as f32,
                        room: self.num_at_opt(&values, 3)?.unwrap_or(0.7) as f32,
                        damp: self.num_at_opt(&values, 4)?.unwrap_or(0.3) as f32,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "SPAT" => {
                let az = self.num_at(&values, 2)? as f32 * core::f32::consts::PI / 180.0;
                let el = self.num_at(&values, 3)? as f32 * core::f32::consts::PI / 180.0;
                let dist = self.num_at(&values, 4)? as f32;
                schedule_spat(
                    self.engine,
                    SpatNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        azimuth: Pfield::constant(az),
                        elevation: Pfield::constant(el),
                        distance: Pfield::constant(dist),
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "ROTATE" => {
                let deg = core::f32::consts::PI / 180.0;
                schedule_rotate(
                    self.engine,
                    RotateNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        yaw: Pfield::constant(self.num_at(&values, 2)? as f32 * deg),
                        pitch: Pfield::constant(self.num_at(&values, 3)? as f32 * deg),
                        roll: Pfield::constant(self.num_at(&values, 4)? as f32 * deg),
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "DECODE" => {
                let layout = SpeakerLayout::from_name(&self.str_at(&values, 2)?).ok_or_else(|| {
                    MincError {
                        message: "layout must be stereo, square, or cube".to_string(),
                        line: 0,
                    }
                })?;
                schedule_decode(
                    self.engine,
                    DecodeNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                        layout,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "BINAURAL" => {
                schedule_binaural(
                    self.engine,
                    BinauralNote {
                        start: self.num_at(&values, 0)?,
                        dur: self.num_at(&values, 1)?,
                    },
                )
                .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            "SPLAT" | "SPLAT4D" => {
                let path = self.str_at(&values, 2)?;
                let asset = self.intern(path);
                let (t0, t1, kind) = if name == "SPLAT4D" {
                    (
                        self.num_at_opt(&values, 3)?.unwrap_or(0.0) as f32,
                        self.num_at_opt(&values, 4)?.unwrap_or(1.0) as f32,
                        2,
                    )
                } else {
                    (0.0, 1.0, 1)
                };
                self.engine
                    .schedule_named(
                        name,
                        self.num_at(&values, 0)?,
                        self.num_at(&values, 1)?,
                        Box::new(SceneNote {
                            kind,
                            asset,
                            t0,
                            t1,
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            yaw: 0.0,
                            pitch: 0.0,
                            roll: 0.0,
                        }),
                    )
                    .map_err(err_core)?;
                Ok(Value::Num(asset as f64))
            }
            "CAMERA" => {
                let x = self.num_at(&values, 2)? as f32;
                let y = self.num_at(&values, 3)? as f32;
                let z = self.num_at(&values, 4)? as f32;
                let yaw = libm::atan2f(-x, z.max(1.0e-4));
                let horiz = libm::sqrtf(x * x + z * z);
                let pitch = libm::atan2f(-(y - 1.0), horiz.max(1.0e-4));
                self.engine
                    .schedule_named(
                        "CAMERA",
                        self.num_at(&values, 0)?,
                        self.num_at(&values, 1)?,
                        Box::new(SceneNote {
                            kind: 3,
                            asset: 0,
                            t0: 0.0,
                            t1: 1.0,
                            x,
                            y,
                            z,
                            yaw,
                            pitch,
                            roll: 0.0,
                        }),
                    )
                    .map_err(err_core)?;
                Ok(Value::Num(0.0))
            }
            other => Err(MincError {
                message: format!("unknown function {other}"),
                line: 0,
            }),
        }
    }

    fn intern(&mut self, path: String) -> u32 {
        if let Some(index) = self.assets.iter().position(|p| p == &path) {
            return index as u32;
        }
        let id = self.assets.len() as u32;
        self.assets.push(path);
        id
    }

    fn num(&self, value: &Value) -> Result<f64, MincError> {
        match value {
            Value::Num(n) => Ok(*n),
            Value::Str(_) => Err(MincError {
                message: "expected a number".to_string(),
                line: 0,
            }),
        }
    }

    fn num_at(&self, values: &[Value], index: usize) -> Result<f64, MincError> {
        let value = values.get(index).ok_or_else(|| MincError {
            message: format!("missing argument {index}"),
            line: 0,
        })?;
        self.num(value)
    }

    fn num_at_opt(&self, values: &[Value], index: usize) -> Result<Option<f64>, MincError> {
        match values.get(index) {
            Some(Value::Num(n)) => Ok(Some(*n)),
            Some(Value::Str(_)) => Err(MincError {
                message: "expected a number".to_string(),
                line: 0,
            }),
            None => Ok(None),
        }
    }

    fn str_at(&self, values: &[Value], index: usize) -> Result<String, MincError> {
        match values.get(index) {
            Some(Value::Str(s)) => Ok(s.clone()),
            _ => Err(MincError {
                message: format!("missing string argument {index}"),
                line: 0,
            }),
        }
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Num(n) => *n != 0.0,
        Value::Str(s) => !s.is_empty(),
    }
}

fn err_core(err: rtmix_core::CoreError) -> MincError {
    MincError {
        message: err.message.to_string(),
        line: 0,
    }
}

pub const EXAMPLE_SCORE: &str = r#"rtsetparams(48000, 2, 256)
// Harmonic Constellation: deterministic vocal timbres orbit one tonal center.
bus_config("FORMANT", "aux")
bus_config("CHANT", "aux")
bus_config("FOF", "out")
bus_config("LPC", "out")
bus_config("MIX", "out")
bus_config("REVERB", "out")
harm = makeharm(2048, 24, 0.85, "odd")
FORMANT(0, 18, 900, 55, 0.5, "u", harm)
CHANT(0, 18, 4300, 110, 0.5, 6.0, 5, 8, "aeiou", harm)
FOF(0, 12, 950, 55, 430, 72, 0.18, 0.12, 17)
FOF(0.35, 11.65, 850, 73.42, 760, 95, 0.16, 0.88, 29)
FOF(0.7, 11.3, 760, 110, 1230, 130, 0.14, 0.38, 41)
LPC(1.5, 8, 3000, 73.42, 0.22, 101)
LPC(5.5, 8, 2700, 110, 0.78, 211)
MIX(0, 18, 0.72)
REVERB(0, 18, 2200, 0.78, 0.28)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minc_score_runs_end_to_end() {
        let score = r#"
            rtsetparams(48000, 2, 128);
            spat_config("ambix", 1);
            bus_config("WAVETABLE", "aux");
            bus_config("SPAT", "spat");
            bus_config("DECODE", "out");
            maketable("sine", 256);
            for (i = 0; i < 2; i = i + 1) {
                if (i == 0) {
                    WAVETABLE(0, 0.03, 8000, 440, 0.5);
                } else {
                    i = i;
                }
            }
            SPAT(0, 0.03, -90, 0, 1);
            DECODE(0, 0.03, "stereo");
        "#;
        let mut engine = Engine::new(48_000, 2, 128);
        execute(score, &mut engine).unwrap();
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        let end = engine.sample_time() + (0.03 * 48_000.0) as u64;
        while engine.sample_time() < end {
            engine.process_block();
            for s in engine.bus_plane(engine.out_bus(), 0) {
                left += s * s;
            }
            for s in engine.bus_plane(engine.out_bus(), 1) {
                right += s * s;
            }
        }
        assert!(left + right > 1.0e-4, "silent score left {left} right {right}");
        assert!(right > left, "expected a right pan, left {left} right {right}");
    }

    #[test]
    fn plan_score_fragment_renders() {
        let score = r#"
rtsetparams(48000, 2, 256)
spat_config("ambix", 3)
bus_config("WAVETABLE", "spat")
bus_config("DECODE", "out")
WAVETABLE(0, 2.0, 20000, 440, 0.5)
DECODE(0, 10, "stereo")
SPLAT(0, 1.0, "scene.spz")
SPLAT4D(0, 1.0, "scene.rt4d", 0.0, 1.0)
CAMERA(0, 1.0, 0, 1.6, 3.0)
"#;
        let mut engine = Engine::new(48_000, 2, 256);
        let assets = execute(score, &mut engine).unwrap();
        assert_eq!(assets, alloc::vec!["scene.spz".to_string(), "scene.rt4d".to_string()]);
        engine.process_seconds(0.02);
        let energy: f32 = engine
            .bus_plane(engine.out_bus(), 0)
            .iter()
            .chain(engine.bus_plane(engine.out_bus(), 1).iter())
            .map(|s| s * s)
            .sum();
        assert!(energy > 0.0);
        let pose = engine.pose();
        assert!((pose.y - 1.6).abs() < 1.0e-3);
    }

    #[test]
    fn example_score_plays_formants() {
        let mut engine = Engine::new(48_000, 2, 256);
        execute(EXAMPLE_SCORE, &mut engine).unwrap();
        let mut prev_l = 0.0f32;
        let mut prev_r = 0.0f32;
        let mut max_jump = 0.0f32;
        let mut peak = 0.0f32;
        let mut started = false;
        let blocks = (48_000 / 256) * 2;
        for _block in 0..blocks {
            engine.process_block();
            let left_bus = engine.out_bus();
            let n = engine.block_size();
            for i in 0..n {
                let l = engine.bus_sample(left_bus, 0, i);
                let r = engine.bus_sample(left_bus, 1, i);
                peak = peak.max(l.abs()).max(r.abs());
                if started {
                    max_jump = max_jump.max((l - prev_l).abs()).max((r - prev_r).abs());
                }
                prev_l = l;
                prev_r = r;
                started = true;
            }
        }
        let energy = prev_l * prev_l + prev_r * prev_r;
        assert!(
            peak > 0.01 && peak < 1.05 && max_jump < 0.35 && energy.is_finite(),
            "peak {peak} max_jump {max_jump}"
        );
    }

    #[test]
    fn granular_score_produces_windowed_audio() {
        let score = r#"
rtsetparams(48000, 2, 128)
maketable("sine", 512)
bus_config("GRANULAR", "out")
GRANULAR(0, 0.4, 12000, 0.08, 32, 1.0, 0.2, 0.5, 77, 1)
"#;
        let mut engine = Engine::new(48_000, 2, 128);
        execute(score, &mut engine).unwrap();
        let mut energy = 0.0;
        for _ in 0..((0.45 * 48_000.0) as usize / 128 + 1) {
            engine.process_block();
            energy += engine
                .bus_plane(engine.out_bus(), 0)
                .iter()
                .chain(engine.bus_plane(engine.out_bus(), 1).iter())
                .map(|sample| sample * sample)
                .sum::<f32>();
        }
        assert!(energy > 1.0e-3, "granular score rendered silence");
    }
}
