use super::ast::*;
use super::ast::Type::*;
use super::lexer::Position;
use std::collections::HashMap;

pub trait CheckableStat {
    fn static_check(&self, s: &mut StaticChecker) -> StaticCheckResult;
}

pub trait CheckableExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult;

    fn ret_type(&self, s: &mut StaticChecker) -> Type;
}

impl CheckableExpr for SLiftExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        let r1 = s.check_variable(&self.left, IntStream, self.pos);
        let r2 = s.check_variable(&self.right, IntStream, self.pos);
        return StaticCheckResult::merge(r1,r2);
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// always ok
impl CheckableExpr for UnitExpr {
    fn static_expr_check(&self, _s: &mut StaticChecker) -> StaticCheckResult {
        return StaticCheckResult::Ok;
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return UnitStream
    }
}

// always ok
impl CheckableExpr for DefaultExpr {
    fn static_expr_check(&self, _s: &mut StaticChecker) -> StaticCheckResult {
        return StaticCheckResult::Ok;
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// int operand and no divide by zero
impl CheckableExpr for ArithExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        let r = s.check_variable(&self.id, IntStream, self.pos);
        if !self.val_left && self.val == 0 && self.op == BinOp::Div {
            return StaticCheckResult::merge(r,
                StaticCheckResult::Err(vec![ProgramError::DivideByZero(self.pos)])
            )
        }
        return r
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// arbitrary type
impl CheckableExpr for TimeExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return s.check_definition(&self.id, self.pos)
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// arbitrary type
impl CheckableExpr for CountExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return s.check_definition(&self.id, self.pos)
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// left int right arbitrary
impl CheckableExpr for DelayExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        let r1 = s.check_variable(&self.left, IntStream, self.pos);
        let r2 = s.check_definition(&self.right, self.pos);
        return StaticCheckResult::merge(r1,r2);
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return UnitStream
    }
}

// left int right arbitrary
impl CheckableExpr for LastExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        let r1 = s.check_variable(&self.left, IntStream, self.pos);
        let r2 = s.check_definition(&self.right, self.pos);
        return StaticCheckResult::merge(r1,r2);
    }

    fn ret_type(&self, _s: &mut StaticChecker) -> Type {
        return IntStream
    }
}

// both parameters must have same type
impl CheckableExpr for MergeExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        let r1 : StaticCheckResult;
        let r2 : StaticCheckResult;
        match s.get_variable_type(&self.left) {
            Some(t) => {
                r1 = s.check_variable(&self.left, t, self.pos);
                r2 = s.check_variable(&self.right, t, self.pos);
            },
            None => {
                r1 = s.check_definition(&self.left, self.pos);
                r2 = s.check_definition(&self.right, self.pos);
            }
        }
        return StaticCheckResult::merge(r1,r2);
    }

    fn ret_type(&self, s: &mut StaticChecker) -> Type {
        // unwrap ok since statement checks validity before return type
        return s.get_variable_type(&self.left).unwrap();
    }
}

impl CheckableExpr for IDExpr {
    fn static_expr_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return s.check_definition(&self.id, self.pos);
    }

    fn ret_type(&self, s: &mut StaticChecker) -> Type {
        // unwrap ok since statement checks validity before return type
        return s.get_variable_type(&self.id).unwrap();
    }
}

impl CheckableStat for InputStat {
    fn static_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return if s.var_defined(&self.id) {
            StaticCheckResult::Err(vec!(ProgramError::AlreadyDefined(self.pos, self.id.clone())))
        } else {
            s.define_variable(&self.id, self.t);
            StaticCheckResult::Ok
        }
    }
}

impl CheckableStat for OutputStat {
    fn static_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return s.check_definition(&self.id, self.pos);
    }
}

impl CheckableStat for DefStat {
    fn static_check(&self, s: &mut StaticChecker) -> StaticCheckResult {
        return if s.var_defined(&self.id) {
            StaticCheckResult::Err(vec!(ProgramError::AlreadyDefined(self.pos, self.id.clone())))
        } else {
            let r = self.value.static_expr_check(s);
            if r != StaticCheckResult::Ok {
                return r;
            }

            let t = self.value.ret_type(s);
            s.define_variable(&self.id, t);
            StaticCheckResult::Ok
        }
    }
}

pub struct StaticChecker {
    symtable: HashMap<Vec<u8>, Type>
}

impl StaticChecker {
    pub fn new() -> StaticChecker {
        StaticChecker {
            symtable: HashMap::new()
        }
    } 

    pub fn define_variable(&mut self, id: &Vec<u8>, ty: Type) {
        self.symtable.insert(id.clone(),ty);
    }

    pub fn var_defined(&mut self, id: &Vec<u8>) -> bool {
        return self.symtable.get(id) != None;
    }

    pub fn get_variable_type(&mut self, id: &Vec<u8>) -> Option<Type> {
        let var = self.symtable.get(id);
        match var {
            Some(v) => Some(*v),
            None => None
        }
    }

    pub fn check_variable(&mut self, id: &Vec<u8>, expected : Type, pos: Position) -> StaticCheckResult {
        return match self.symtable.get(id) {
            Some(t) => {
                if t == &expected {
                    StaticCheckResult::Ok
                } else {
                    StaticCheckResult::Err(vec![ProgramError::TypeMismatch(pos,expected,*t)])
                }
            },
            None => StaticCheckResult::Err(vec![ProgramError::VarNotFound(pos,id.clone())])
        }
    }

    pub fn check_definition(&mut self, id: &Vec<u8>, pos: Position) -> StaticCheckResult {
        return match self.symtable.get(id) {
            Some(_) => StaticCheckResult::Ok,
            None => StaticCheckResult::Err(vec![ProgramError::VarNotFound(pos,id.clone())])
        }
    }
}

pub fn static_check(ast: &Vec<Box<dyn Statement>>) -> bool {
    let mut s = StaticChecker::new();
    let mut result = StaticCheckResult::Ok;
    for stat in ast {
        result = StaticCheckResult::merge(result, stat.static_check(&mut s));
    }
    let success : bool;
    match result {
        StaticCheckResult::Ok => success = true,
        StaticCheckResult::Err(errors) => {
            eprintln!("Static Check Error\n");
            for e in errors {
                eprintln!("\t{}", e.err_string())
            }
            success = false;
        }
    }
    success
}

#[derive(Clone, PartialEq)]
pub enum StaticCheckResult {
    Ok,
    Err(Vec<ProgramError>)
}

impl StaticCheckResult {
    pub fn merge(res1: StaticCheckResult, res2: StaticCheckResult) -> StaticCheckResult{
        match (res1, res2) {
            (StaticCheckResult::Ok, StaticCheckResult::Ok) => StaticCheckResult::Ok, 
            (StaticCheckResult::Err(e1), StaticCheckResult::Ok) => StaticCheckResult::Err(e1), 
            (StaticCheckResult::Ok, StaticCheckResult::Err(e2)) => StaticCheckResult::Err(e2), 
            (StaticCheckResult::Err(e1), StaticCheckResult::Err(e2)) => {
                let mut err_vec = vec!();
                err_vec.extend(e1);
                err_vec.extend(e2);
                StaticCheckResult::Err(err_vec)
            }
        }
    }
}
