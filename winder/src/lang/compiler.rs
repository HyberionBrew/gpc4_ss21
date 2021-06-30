use super::ast::*;
use std::collections::HashMap;
use petgraph::{Graph, Directed};
use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;

// when true, 4 bit registers, else 2-bit
static mut LONG_REGS : bool = false;

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Add(u32, u32, u32),
    Mul(u32, u32, u32),
    Sub(u32, u32, u32),
    Div(u32, u32, u32),
    Mod(u32, u32, u32),
    Delay(u32, u32, u32),
    Last(u32, u32, u32),
    Time(u32, u32),
    Merge(u32, u32, u32),
    Count(u32, u32),
    AddI(u32, i32, u32),
    MulI(u32, i32, u32),
    SubI(u32, i32, u32),
    SubII(u32, i32, u32),
    DivI(u32, i32, u32),
    DivII(u32, i32, u32),
    ModI(u32, i32, u32),
    ModII(u32, i32, u32),
    Default(u32, i32),
    Load(u32),
    // currently unused
    // Load4(u32),
    // Load6(u32),
    // Load8(u32),
    Store(u32),
    Free(u32),
    Unit(u32),
    Exit
}

impl Instruction {
    fn get_bytes(&self) -> Vec<u8>{
        macro_rules! r_type {
            ($opcode : expr, $r1: expr, $r2: expr, $rd: expr) => {
                {
                    let mut vec = [$opcode as u8].to_vec();
                    unsafe {
                        if (LONG_REGS) {
                            vec.extend_from_slice(&(*$r1 as u32).to_be_bytes());
                            vec.extend_from_slice(&(*$r2 as u32).to_be_bytes());
                            vec.extend_from_slice(&(*$rd as u32).to_be_bytes());
                        } else {
                            vec.extend_from_slice(&(*$r1 as u16).to_be_bytes());
                            vec.extend_from_slice(&(*$r2 as u16).to_be_bytes());
                            vec.extend_from_slice(&(*$rd as u16).to_be_bytes());
                        }
                    }
                    return vec;
                }
            };
        }
        macro_rules! i_type {
            ($opcode : expr, $r1: expr, $im: expr, $rd: expr) => {
                {
                    let mut vec = [$opcode as u8].to_vec();
                    unsafe {
                        if (LONG_REGS) {
                            vec.extend_from_slice(&(*$r1 as u32).to_be_bytes());
                            vec.extend_from_slice(&(*$im as i32).to_be_bytes());
                            vec.extend_from_slice(&(*$rd as u32).to_be_bytes());
                        } else {
                            vec.extend_from_slice(&(*$r1 as u16).to_be_bytes());
                            vec.extend_from_slice(&(*$im as i32).to_be_bytes());
                            vec.extend_from_slice(&(*$rd as u16).to_be_bytes());
                        }
                    }
                    return vec;
                }
            };
        }
        macro_rules! m_type {
            ($opcode : expr, $r1: expr) => {
                {
                    let mut vec = [$opcode as u8].to_vec();
                    unsafe {
                        if (LONG_REGS) {
                            vec.extend_from_slice(&(*$r1 as u32).to_be_bytes());
                        } else {
                            vec.extend_from_slice(&(*$r1 as u16).to_be_bytes());
                        }
                    }
                    return vec;
                }
            };
        }
        match &self {
            // R-Type
            Instruction::Add(r1,r2,rd) => r_type!(0x08, r1, r2, rd),
            Instruction::Mul(r1,r2,rd) => r_type!(0x09, r1, r2, rd),
            Instruction::Sub(r1,r2,rd) => r_type!(0x0A, r1, r2, rd),
            Instruction::Div(r1,r2,rd) => r_type!(0x0B, r1, r2, rd),
            Instruction::Mod(r1,r2,rd) => r_type!(0x0C, r1, r2, rd),
            Instruction::Delay(r1,r2,rd) => r_type!(0x0D, r1, r2, rd),
            Instruction::Last(r1,r2,rd) => r_type!(0x0E, r1, r2, rd),
            Instruction::Time(r1,rd) => r_type!(0x0F, r1, &0, rd),
            Instruction::Merge(r1,r2,rd) => r_type!(0x10, r1, r2, rd),
            Instruction::Count(r1,rd) => r_type!(0x11, r1, &0, rd),
            // I-Type
            Instruction::AddI(r1,imm,rd) => i_type!(0x48, r1, imm, rd),
            Instruction::MulI(r1,imm,rd) => i_type!(0x49, r1, imm, rd),
            Instruction::SubI(r1,imm,rd) => i_type!(0x4A, r1, imm, rd),
            Instruction::SubII(r1,imm,rd) => i_type!(0x4B, r1, imm, rd),
            Instruction::DivI(r1,imm,rd) => i_type!(0x4C, r1, imm, rd),
            Instruction::DivII(r1,imm,rd) => i_type!(0x4D, r1, imm, rd),
            Instruction::ModI(r1,imm,rd) => i_type!(0x4E, r1, imm, rd),
            Instruction::ModII(r1,imm,rd) => i_type!(0x4F, r1, imm, rd),
            Instruction::Default(imm,rd) => i_type!(0x50, &0, imm, rd),
            // M-Type
            Instruction::Load(r1) => m_type!(0x88,r1),
            // Instruction::Load4(r1) => m_type!(0x89,r1),
            // Instruction::Load6(r1) => m_type!(0x8A,r1),
            // Instruction::Load8(r1) => m_type!(0x8B,r1),
            Instruction::Store(r1) => m_type!(0x8C,r1),
            Instruction::Free(r1) => m_type!(0x8D,r1),
            Instruction::Unit(r1) => m_type!(0x8E,r1),
            Instruction::Exit => [0xFF as u8].to_vec(),
        }
    }

    /*
    fn byte_len(&self) -> usize {
        match self {
            Instruction::Add(_,_,_)
            | Instruction::Mul(_,_,_)
            | Instruction::Sub(_,_,_)
            | Instruction::Div(_,_,_)
            | Instruction::Mod(_,_,_)
            | Instruction::Delay(_,_,_)
            | Instruction::Last(_,_,_)
            | Instruction::Time(_,_)
            | Instruction::Merge(_,_,_)
            | Instruction::Count(_,_) => unsafe { if !LONG_REGS { 7 } else { 13 } }
            Instruction::AddI(_,_,_)
            | Instruction::MulI(_,_,_)
            | Instruction::SubI(_,_,_)
            | Instruction::SubII(_,_,_)
            | Instruction::DivI(_,_,_)
            | Instruction::DivII(_,_,_)
            | Instruction::ModI(_,_,_)
            | Instruction::ModII(_,_,_)
            | Instruction::Default(_,_) => unsafe { if !LONG_REGS { 9 } else { 13 } }
            Instruction::Load(_)
            // | Instruction::Load4(_)
            // | Instruction::Load6(_)
            // | Instruction::Load8(_)
            | Instruction::Store(_)
            | Instruction::Free(_)
            | Instruction::Unit(_) => unsafe { if !LONG_REGS { 3 } else { 5 } }
            Instruction::Exit => 1
        }
    }
     */
}

macro_rules! rd {
    ($compiler: expr, $id: expr) => {
        *$compiler.reg_map.get($id).unwrap();
    }
}

pub trait CompilableStat {
    fn preprocess(&self, c: &mut Compiler);
    fn compute_last_use(&self, c: &mut Compiler);
    fn compile(&self, c: &mut Compiler);
}

pub trait CompilableExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>);
    fn compute_last_use(&self, c: &mut Compiler);
    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>);
}

impl CompilableStat for InputStat {
    fn preprocess(&self, c: &mut Compiler) {
        c.add_node(&self.id);
    }

    fn compute_last_use(&self, _c: &mut Compiler) {    }

    fn compile(&self, c: &mut Compiler) {
        // only compile when rd is needed
        if c.is_used(&self.id) {
            let r = c.alloc_reg(&self.id);
            c.in_regs.push((self.id.clone(), self.t, r))
        }
    }
}

impl CompilableStat for OutputStat {
    fn preprocess(&self, c: &mut Compiler) {
        c.set_needed(&self.id);
        c.out_nodes.push((self.id.clone(), *c.nodes.get(&self.id).unwrap()));
    }

    fn compute_last_use(&self, _c: &mut Compiler) {    }

    fn compile(&self, c: &mut Compiler) {
        // unwrap asserted by static check
        let r = c.get_reg(&self.id);
        c.out_regs.push((self.id.clone(),r));
    }
}

impl CompilableStat for DefStat {
    fn preprocess(&self, c: &mut Compiler) {
        c.add_node(&self.id);
        self.value.preprocess(c, &self.id);
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        if c.is_used(&self.id) {
            self.value.compute_last_use(c);
           c.mark_last_usage(&self.id);
        }
    }

    fn compile(&self, c: &mut Compiler) {
        // only compile when rd is needed
        if c.is_used(&self.id) {
            c.alloc_reg(&self.id);
            self.value.compile(c,&self.id);
            c.free_if_last_usage(&self.id);
        }
    }
}

impl CompilableExpr for UnitExpr {
    fn preprocess(&self, _c: &mut Compiler, _parent: &Vec<u8>) { }

    fn compute_last_use(&self, _c: &mut Compiler) { }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.push_instruction(Instruction::Unit(rd!(c,rd)));
    }
}

impl CompilableExpr for DefaultExpr {
    fn preprocess(&self, _c: &mut Compiler, _parent: &Vec<u8>) { }

    fn compute_last_use(&self, _c: &mut Compiler) { }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.push_instruction(Instruction::Default(rd!(c,rd), self.value));
    }
}

impl CompilableExpr for SLiftExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.left, parent);
        c.add_dependency(&self.right, parent);
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.left);
        c.mark_last_usage(&self.right);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.left);
        c.load_if_first_usage(&self.right);
        // unwraps ok because of static check
        let lr = c.get_reg(&self.left);
        let rr = c.get_reg(&self.right);
        c.push_instruction(match self.op {
            BinOp::Plus => Instruction::Add(lr,rr,rd!(c,rd)),
            BinOp::Mult => Instruction::Mul(lr,rr,rd!(c,rd)),
            BinOp::Minus => Instruction::Sub(lr,rr,rd!(c,rd)),
            BinOp::Div => Instruction::Div(lr,rr,rd!(c,rd)),
            BinOp::Modulo => Instruction::Mod(lr,rr,rd!(c,rd))
        });
        c.free_if_last_usage(&self.left);
        c.free_if_last_usage(&self.right);
    }
}

impl CompilableExpr for ArithExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.id, parent)
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.id);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.id);
        let r = c.get_reg(&self.id);
        c.push_instruction(match self.op {
            BinOp::Plus => Instruction::AddI(r,self.val,rd!(c,rd)),
            BinOp::Mult => Instruction::MulI(r,self.val,rd!(c,rd)),
            o => {
                if self.val_left {
                    match o {
                        BinOp::Minus => Instruction::SubII(r,self.val,rd!(c,rd)),
                        BinOp::Div => Instruction::DivII(r,self.val,rd!(c,rd)),
                        BinOp::Modulo => Instruction::ModII(r,self.val,rd!(c,rd)),
                        _ => unreachable!()
                    }
                } else {
                    match o {
                        BinOp::Minus => Instruction::SubI(r,self.val,rd!(c,rd)),
                        BinOp::Div => Instruction::DivI(r,self.val,rd!(c,rd)),
                        BinOp::Modulo => Instruction::ModI(r,self.val,rd!(c,rd)),
                        _ => unreachable!()
                    }
                }
            }
        });
        c.free_if_last_usage(&self.id)
    }
}

impl CompilableExpr for DelayExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.left, parent);
        c.add_dependency(&self.right, parent);
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.left);
        c.mark_last_usage(&self.right);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.left);
        c.load_if_first_usage(&self.right);
        // unwraps ok because of static check
        let lr = c.get_reg(&self.left);
        let rr = c.get_reg(&self.right);
        c.push_instruction(Instruction::Delay(lr,rr,rd!(c,rd)));
        c.free_if_last_usage(&self.left);
        c.free_if_last_usage(&self.right);
    }
}

impl CompilableExpr for LastExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.left, parent);
        c.add_dependency(&self.right, parent);
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.left);
        c.mark_last_usage(&self.right);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.left);
        c.load_if_first_usage(&self.right);
        let lr = c.get_reg(&self.left);
        let rr = c.get_reg(&self.right);
        c.push_instruction(Instruction::Last(lr,rr,rd!(c,rd)));
        c.free_if_last_usage(&self.left);
        c.free_if_last_usage(&self.right);
    }
}

impl CompilableExpr for TimeExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.id, parent)
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.id);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.id);
        let r = c.get_reg(&self.id);
        c.push_instruction(Instruction::Time(r, rd!(c,rd)));
        c.free_if_last_usage(&self.id)
    }
}

impl CompilableExpr for MergeExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.left, parent);
        c.add_dependency(&self.right, parent);
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.left);
        c.mark_last_usage(&self.right);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.left);
        c.load_if_first_usage(&self.right);
        let lr = c.get_reg(&self.left);
        let rr = c.get_reg(&self.right);
        c.push_instruction(Instruction::Merge(lr, rr, rd!(c,rd)));
        c.free_if_last_usage(&self.left);
        c.free_if_last_usage(&self.right);
    }
}

impl CompilableExpr for CountExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.add_dependency(&self.id, parent)
    }

    fn compute_last_use(&self, c: &mut Compiler) {
        c.mark_last_usage(&self.id);
    }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        c.load_if_first_usage(&self.id);
        let r = c.get_reg(&self.id);
        c.push_instruction(Instruction::Count(r, rd!(c,rd)));
        c.free_if_last_usage(&self.id)
    }
}

impl CompilableExpr for IDExpr {
    fn preprocess(&self, c: &mut Compiler, parent: &Vec<u8>) {
        c.aliases.insert(parent.clone(), self.id.clone());
        c.add_dependency(&self.id, parent)
    }

    fn compute_last_use(&self, _c: &mut Compiler) { }

    fn compile(&self, c: &mut Compiler, rd: &Vec<u8>) {
        // free target register since it is renamed
        c.free_reg(rd);
    }
}

pub struct Compiler {
    // REGISTER MANAGEMENT
    reg_map: HashMap<Vec<u8>, u32>, // Register Map with currently occupied registers
    max_reg: u32, // Maximum register number that has not been occupied
    freed_regs: Vec<u32>,   // Freed registers

    // BYTECODE DATA
    instructions: Vec<Instruction>, // instruction vector for main (out-of-funcdef) bytecode
    in_regs: Vec<(Vec<u8>, Type, u32)>, // Input Registers (reg_name, type, address)
    out_regs: Vec<(Vec<u8>, u32)>, // Output Registers (reg_name, address)

    // COMPILER STATE
    curr_index: u64,

    // PREPROCESSING
    deps: Graph<bool, (), Directed>, // dependency graph
    nodes: HashMap<Vec<u8>, NodeIndex<u32>>, // node indices for deps
    out_nodes: Vec<(Vec<u8>, NodeIndex<u32>)>, // output nodes in graph
    first_use: HashMap<Vec<u8>, u64>, // first use of Vec<u8> input stream at u64
    last_use: HashMap<Vec<u8>, u64>, // last use of Vec<u8> at u64
    aliases: HashMap<Vec<u8>,Vec<u8>> // record aliases of variables
}

impl Compiler {
    fn new() -> Compiler {
        Compiler {
            instructions: vec!(),
            reg_map: HashMap::new(),
            max_reg: 0,
            freed_regs: vec!(),
            in_regs: vec!(),
            out_regs: vec!(),
            curr_index: 0,
            last_use: HashMap::new(),
            first_use: HashMap::new(),
            aliases: HashMap::new(),
            deps: Graph::new(),
            out_nodes: vec!(),
            nodes: HashMap::new(),
        }
    }

    fn alloc_reg(&mut self, id: &Vec<u8>) -> u32 {
        let r : u32;
        if !self.freed_regs.is_empty() {
            r = self.freed_regs.remove(0); // unwrap ok because empty check
        } else {
            r = self.max_reg;
            self.max_reg += 1;
            if self.max_reg > u16::max_value() as u32{
                unsafe {
                    LONG_REGS = true;
                }
            }
        }
        self.reg_map.insert(id.clone(),r);
        r
    }

    fn free_reg(&mut self, id : &Vec<u8>) {
        let v = self.reg_map.remove(id).expect("Tried to free unoccupied register");
        self.freed_regs.push(v);
        self.instructions.push(Instruction::Free(v));
    }

    fn get_reg(&mut self, id: &Vec<u8>) -> u32 {
        let act_id;
        if let Some(i) = self.aliases.get(id) {
            act_id = i;
        } else {
            act_id = id;
        }
        // unwrap ok because of static check
        return *self.reg_map.get(act_id).unwrap()
    }

    fn add_node(&mut self, id : &Vec<u8>) {
        let n = self.deps.add_node(false);
        self.nodes.insert(id.clone(), n);
    }

    fn add_dependency(&mut self, target : &Vec<u8>, source: &Vec<u8>) {
        self.mark_first_usage(target);
        let si = *self.nodes.get(source).expect("Failed to get source node");
        let ti = *self.nodes.get(target).expect("Failed to get target node");
        self.deps.add_edge(si, ti, ());
    }

    fn set_needed(&mut self, id : &Vec<u8>) {
        let i = self.nodes.get(id).expect("Failed to get node");
        self.deps[*i] = true;
    }

    fn act_id<'a>(&'a self, id: &'a Vec<u8>) -> &'a Vec<u8> {
        // get actual id for aliases
        let act_id;
        if let Some(s) = self.aliases.get(id) {
            act_id = s;
        } else {
            act_id = id;
        }
        act_id
    }

    fn mark_first_usage(&mut self, id: &Vec<u8>) {
        let act_id = self.act_id(id);
        // set first use if applicable
        if !self.first_use.contains_key(act_id) {
            let act_id = act_id.clone();
            self.first_use.insert(act_id, self.curr_index);
        }
    }

    fn mark_last_usage(&mut self, id: &Vec<u8>) {
        let act_id = self.act_id(id);
        // set last use if applicable
        if !self.last_use.contains_key(act_id) {
            let act_id = act_id.clone();
            self.last_use.insert(act_id, self.curr_index);
        }
    }

    fn is_used(&self, id: &Vec<u8>) -> bool{
        let i = *self.nodes.get(id).unwrap();
        self.deps[i]
    }

    fn update_last_use(&mut self, ast: &Vec<Box<dyn Statement>>) {
        self.curr_index = ast.len() as u64;
        for s in ast.iter().rev() {
            self.curr_index -= 1;
            s.compute_last_use(self);
        }
    }

    fn push_instruction(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    fn free_if_last_usage(&mut self, id: &Vec<u8>) {
        // unwrap ok because of static and dependency check
        if *self.last_use.get(id).unwrap() == self.curr_index {
            if !self.out_nodes.iter().any(|(x,_)| x == id) { // don't free output regs
                self.free_reg(id);
            } else {
                // unwrap ok because node is in dependency graph
                self.instructions.push(Instruction::Store(*self.reg_map.get(id).unwrap()))
            }
        }
    }

    fn load_if_first_usage(&mut self, id: &Vec<u8>) {
        // unwrap ok because of static and dependency check
        if *self.first_use.get(id).unwrap() == self.curr_index {
            // check whether register is input register
            if self.in_regs.iter().any(|(x,_,_)| x == id ) {
                let r = self.get_reg(&id);
                self.instructions.push(Instruction::Load(r));
            }
        }
    }

    fn check_dependencies(&mut self) {
        for (_,o) in &self.out_nodes {
            let mut dfs = Dfs::new(&self.deps, *o);
            while let Some(nx) = dfs.next(&self.deps) {
                self.deps[nx] = true;
            }
        }
    }

    fn generate_bytecode(&mut self, file_name : String) {
        macro_rules! field_delim {
            ($bytes : expr) => {
                $bytes.extend(&vec![0xF0,0xF0])
            }
        }

        let mut bytecode = Vec::<u8>::new();
        // Push magic number
        bytecode.extend(b"XRAY");

        // Push spec field
        bytecode.extend(b"SPEC");
        // Version 1.0
        bytecode.extend(&vec![0x00,0x01,0x00,0x00]);

        field_delim!(bytecode);

        // Push register length
        bytecode.extend(b"REGL");
        // Version 1.0
        if unsafe {LONG_REGS} {
            bytecode.push(0x01);
        } else {
            bytecode.push(0x00);
        }

        field_delim!(bytecode);

        // Push input streams
        for (id,t,r) in &self.in_regs {
            bytecode.extend(b"INST");
            if unsafe {LONG_REGS} {
                bytecode.extend(&r.to_be_bytes());
            } else {
                bytecode.extend(&(*r as u16).to_be_bytes());
            }
            match &t {
                Type::UnitStream => bytecode.push(0x00),
                Type::IntStream => bytecode.push(0x01)
            }
            bytecode.extend(id);
            bytecode.push(0x0);
            field_delim!(bytecode);
        }


        // Push output streams
        for (id,r) in &self.out_regs {
            bytecode.extend(b"OUST");
            if unsafe {LONG_REGS} {
                bytecode.extend(&r.to_be_bytes());
            } else {
                bytecode.extend(&(*r as u16).to_be_bytes());
            }
            bytecode.extend(id);
            bytecode.push(0x0);
            field_delim!(bytecode);
        }

        // Header end
        bytecode.extend(&vec![0xFF,0xFF]);

        for i in &self.instructions {
            bytecode.extend(i.get_bytes());
        }

        // write binary to file
        use std::io::Write;
        let file_name = file_name + ".coil";
        let file = std::fs::File::create(&file_name);
        match file {
            Result::Ok(mut f) => {
                match f.write(&bytecode) {
                    Ok(b) => {
                        println!("Compiled successfully to {} ({} bytes)", file_name, b)
                    },
                    Err(_) => {
                        eprintln!("Failed to write bytecode to file, exiting");
                        std::process::exit(1)
                    }
                }
            }
            _ => {
                eprintln!("Failed to create file, exiting");
                std::process::exit(1)
            },
        }
    }

    fn compile(&mut self, ast: Vec<Box<dyn Statement>>) {
        for stat in &ast {
            stat.preprocess(self);
            self.curr_index += 1;
        }
        self.check_dependencies();
        self.update_last_use(&ast);
        self.curr_index = 0;
        for stat in &ast {
            stat.compile(self);
            self.curr_index += 1;
        }
        self.push_instruction(Instruction::Exit);
    }
}

fn print_compiler_results(c: &Compiler) {
    print!("{}", get_ir(c));
}

fn get_ir(c : &Compiler) -> String {
    let mut out = String::new();
    // print inputs
    out += "Input Streams:\n";
    for (var,t,r) in &c.in_regs {
        out += &*format!("In \"{}\" ({:?}): {}\n", String::from_utf8_lossy(&var), t, r);
    }
    out += "Output Streams:\n";
    // print outputs
    for (var, r) in &c.out_regs {
        out += &*format!("Out \"{}\": {}\n",String::from_utf8_lossy(var),r);
    }
    out += "=============\n";
    // print assembly
    let mut index = 1;
    for inst in &c.instructions {
        out += &*format!("{}: {:?}\n",index, inst);
        index += 1;
    }
    out
}

pub fn compile (ast: Vec<Box<dyn Statement>>, debug: bool, file_name : String) {
    let mut c = Compiler::new();
    c.compile(ast);
    if debug {
        print_compiler_results(&c);
    } else {
        c.generate_bytecode(file_name);
    }
}