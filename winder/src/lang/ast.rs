use super::lexer::Position;
use super::lexer::Token;
//use super::interpreter::Interpreter;
use super::staticcheck::*;
//use super::compiler::*;

use crate::lang::compiler::{CompilableStat, CompilableExpr};

/* Traits */
pub trait Statement : CheckableStat + CompilableStat {
    fn to_json_string(&self) -> String {
        format!("to_json_string not implemented")
    }
}

pub trait Expression : CheckableExpr + CompilableExpr {
    fn to_json_string(&self) -> String {
        format!("to_json_string not implemented")
    }
}

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd)]
pub enum Type {
    UnitStream,
    IntStream
}

pub trait Operation {
    fn enum_string(&self) -> String;
}

impl std::fmt::Debug for dyn Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        write!(f, "{}", self.enum_string())
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self ,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Operation for BinOp {
    fn enum_string(&self) -> String {
        format!("{:?}",self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Minus,
    Plus,
    Div,
    Mult,
    Modulo
}

impl BinOp {
    pub fn get_from_token(t : Token) -> Option<BinOp> {
        use super::lexer::Token::*;
        match t {
            OpMinus => Some(BinOp::Minus),
            OpPlus => Some(BinOp::Plus),
            OpDiv => Some(BinOp::Div),
            OpMul => Some(BinOp::Mult),
            OpModulo => Some(BinOp::Modulo),
            _ => None
        }
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self ,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opstr = match self {
            BinOp::Minus => "-",
            BinOp::Plus  => "+",
            BinOp::Div   => "/",
            BinOp::Modulo=> "%",
            BinOp::Mult  => "*",
        };
        write!(f, "{}", opstr)
    }
}

/* Expression Node Types */

pub struct UnitExpr {
    pub pos: Position
}

impl Expression for UnitExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Unit\", \n\"pos\": {}\n}}",
                self.pos.to_json_string())
    }
}

pub struct DefaultExpr {
    pub value : i32,
    pub pos: Position
}

impl Expression for DefaultExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Default\", \n\"value\": \"{}\", \n\"pos\": {}\n}}",
                self.value, self.pos.to_json_string())
    }
}

pub struct SLiftExpr {
    pub op : BinOp,
    pub left : Vec<u8>,
    pub right : Vec<u8>,
    pub pos: Position
}

impl Expression for SLiftExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"SLift\", \n\"op\": \"{:?}\", \n\"left\": \"{}\",\n \"right\": \"{}\",\n\"pos\": {}\n}}",
                self.op, String::from_utf8_lossy(&self.left),
                String::from_utf8_lossy(&self.right),
                self.pos.to_json_string())
    }
}

pub struct ArithExpr {
    pub op : BinOp,
    pub id : Vec<u8>,
    pub val : i32,
    // indicates whether constant value is left operand
    pub val_left : bool,
    pub pos: Position
}

impl Expression for ArithExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Arith\", \n\"op\": \"{:?}\", \n\"id\": \"{}\",\n \"val\": \"{}\",\n\"val_left\": \"{}\",\n\"pos\": {}\n}}",
                self.op, String::from_utf8_lossy(&self.id),
                self.val, self.val_left,
                self.pos.to_json_string())
    }
}
pub struct DelayExpr {
    pub left : Vec<u8>,
    pub right : Vec<u8>,
    pub pos: Position
}

impl Expression for DelayExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Delay\", \n\"left\": \"{}\",\n\"right\": \"{}\",\n\"pos\": {}\n}}",
                String::from_utf8_lossy(&self.left),
                String::from_utf8_lossy(&self.right),
                self.pos.to_json_string())
    }
}

pub struct LastExpr {
    pub left : Vec<u8>,
    pub right : Vec<u8>,
    pub pos: Position
}

impl Expression for LastExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Last\", \n\"left\": \"{}\",\n\"right\": \"{}\",\n\"pos\": {}\n}}",
                String::from_utf8_lossy(&self.left),
                String::from_utf8_lossy(&self.right),
                self.pos.to_json_string())
    }
}

pub struct TimeExpr {
    pub id : Vec<u8>,
    pub pos: Position
}

impl Expression for TimeExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Time\", \n\"id\": \"{}\",\n \"pos\": {}\n}}",
                String::from_utf8_lossy(&self.id),
                self.pos.to_json_string())
    }
}

pub struct MergeExpr {
    pub left : Vec<u8>,
    pub right : Vec<u8>,
    pub pos: Position
}

impl Expression for MergeExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Merge\", \n\"left\": \"{}\",\n\"right\": \"{}\",\n\"pos\": {}\n}}",
                String::from_utf8_lossy(&self.left),
                String::from_utf8_lossy(&self.right),
                self.pos.to_json_string())
    }
}

pub struct CountExpr {
    pub id : Vec<u8>,
    pub pos: Position
}

impl Expression for CountExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Count\", \n\"id\": \"{}\",\n \"pos\": {}\n}}",
                String::from_utf8_lossy(&self.id),
                self.pos.to_json_string())
    }
}

pub struct IDExpr {
    pub id : Vec<u8>,
    pub pos: Position
}

impl Expression for IDExpr {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"ID\", \n\"id\": \"{}\",\n \"pos\": {}\n}}",
                String::from_utf8_lossy(&self.id),
                self.pos.to_json_string())
    }
}

/* Statement Node Types */

// in x : Events[Int]
pub struct InputStat {
    pub id : Vec<u8>,
    pub t : Type,
    pub pos : Position
}

impl Statement for InputStat {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Input\", \n\"id\": \"{}\", \n\"type\": \"{}\", \n\"pos\": {}\n}}",
                String::from_utf8_lossy(&self.id), self.t, self.pos.to_json_string())
    }
}

// out x
pub struct OutputStat {
    pub id : Vec<u8>,
    pub pos : Position
}

impl Statement for OutputStat {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Output\", \n\"id\": \"{}\",\n\"pos\": {}\n}}",
                String::from_utf8_lossy(&self.id), self.pos.to_json_string())
    }
}

// def x = y
pub struct DefStat {
    pub id : Vec<u8>,
    pub value : Box<dyn Expression>,
    pub pos : Position
}

impl Statement for DefStat {
    fn to_json_string(&self) -> String{
        format!("{{ \n\"node\": \"Define\", \n\"id\": \"{}\", \n\"value\": {}, \n\"pos\": {}\n}}",
        String::from_utf8_lossy(&self.id), self.value.to_json_string(), self.pos.to_json_string())
    }
}

pub fn ast_to_json_string (stat_list: &Vec<Box<dyn Statement>>) -> String {
    let mut stats = String::from("{\n");
    for (i, p) in stat_list.iter().enumerate() {
        stats += &format!("\"{}\": ", i);
        stats += &p.as_ref().to_json_string();
        if i != stat_list.len()-1 {stats += ",\n"}
    } 
    stats + "\n}"
}

// Program Error

#[derive(PartialEq, Clone)]
pub enum ProgramError {
    TypeMismatch(Position, Type, Type), // position, expected, actual
    VarNotFound(Position, Vec<u8>), // positon, var_id
    DivideByZero(Position),
    AlreadyDefined(Position, Vec<u8>) // position, var_id
}

impl ProgramError {
    pub fn err_string(&self) -> String {
        let utf8 = String::from_utf8_lossy;
        match self {
            ProgramError::TypeMismatch(pos, exp_type, act_type) => {
                format!("{}: Type mismatch, expected {}, got {}", pos, exp_type, act_type)
            },
            ProgramError::VarNotFound(pos, id) => {
                format!("{}: Variable \"{}\" is not defined", pos, utf8(id))
            },
            ProgramError::DivideByZero(pos) => {
                format!("{}: Cannot divide by zero", pos)
            },
            ProgramError::AlreadyDefined(pos,id) => {
                format!("{}: Variable {} is already defined", pos, utf8(id))
            },
        }
    }
}
