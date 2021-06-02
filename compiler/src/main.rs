mod lang;
use clap::{App, Arg};
use crate::lang::ast::ast_to_json_string;
//use lang::interpreter::*;

fn main() {
    let matches = App::new("winder").version("1.0")
        .author("Klaus Kraßnitzer <klaus.krassnitzer@tuwien.ac.at>")
        .about("Compiler for flat TeSSLa specifications")
        .after_help("winder supports 3 running modes:\
        \n    compile \t compile a flat TeSSLa specification to a .coil file\
        \n    print-ast \t print the abstract syntax tree of a TeSSLa specification\
        \n    print-ir \t print the intermediate representation of a TeSSLa specification")
        .arg(Arg::with_name("MODE")
            .help("operation mode")
            .possible_values(&vec!("compile", "print-ast", "print-ir"))
            .required(true)
            .index(1))
        .arg(Arg::with_name("SPEC")
            .help("flat TeSSLa specification file")
            .required(true)
            .required(true)
            .index(2))
        .get_matches();

    // unwraps ok because required args
    let path = matches.value_of("SPEC").unwrap();
    let mode = Mode::from_str(matches.value_of("MODE").unwrap()).unwrap();

    compile(path,mode);
}

fn compile(input: &str, mode: Mode) {
    use lang::*;
    let source = std::fs::read_to_string(input).expect("Error while reading file");
    
    let tokens = lexer::tokenize(&source[..]);
    if tokens.is_none() {
        eprintln!("\nAborting due to previous lexer errors\n");
        std::process::exit(1);
    }

    let ast = parser::parse(tokens.unwrap());
    if ast.is_none() {
        eprintln!("Aborting due to previous parser errors\n");
        std::process::exit(1);
    }

    let ast = ast.unwrap();

    if mode == Mode::PrintAST {
        std::print!("{}\n",ast_to_json_string(&ast));
        std::process::exit(0);
    }
            
    let static_success = staticcheck::static_check(&ast);
    if !static_success {
        eprintln!("\nAborting due to previous static check errors\n");
        std::process::exit(1);
    }

    let mut outfile = input.clone().to_owned();
    // truncate file path if it has a file ending that is not .coil
    if let Some(p) = outfile.chars().rev().position(|c| c == '.') {
        let pos = outfile.len()-p-1;
        if &outfile[pos..] != ".coil" {
            outfile.truncate(pos);
        } 
    }

    let print_ir = mode == Mode::PrintIR;
    compiler::compile(ast, print_ir, outfile);
}

#[derive(PartialEq)]
enum Mode {
    Compile,
    PrintAST,
    PrintIR
}

impl Mode {
    pub fn from_str(str : &str) -> Option<Mode> {
        match str {
            "compile" => Some(Mode::Compile),
            "print-ast" => Some(Mode::PrintAST),
            "print-ir" => Some(Mode::PrintIR),
            _ => None
        }
    }
}