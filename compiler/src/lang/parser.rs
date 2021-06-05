use super::ast::*;
use super::lexer::Token;
use super::lexer::Position;
use std::collections::HashMap;

struct Parser {
    index : usize,
    tokenlist : Vec<(Token, Position)>,
    errors : HashMap<Position, Vec<Token>>,
    error_queue : (Position, Vec<Token>),
}

macro_rules! consume {
    ($parser:expr, $rewind_point:expr, $token:expr) => {{
        if !Parser::token_matches(&$parser, 0, &$token) {
            $parser.report_error($token);
            $parser.rewind($rewind_point);
            return None; 
        }
        $parser.pop().unwrap()
    }};
}

macro_rules! consume_any {
    ($parser:expr, $rewind_point:expr, $( $token:expr ),+) => {{
        let mut x : Option<Token> = None;
        let index = $parser.index;
        $(
            if Parser::token_matches_at_index(&$parser,index,&$token) {
                x = $parser.pop()
            }
        )*
        if x.is_none() {
            match $parser.curr_pos() {
                Some(p) => {
                    if p == $parser.error_queue.0 {
                        $(
                            $parser.error_queue.1.push($token);
                        )*
                    } else if p > $parser.error_queue.0 {
                        $parser.error_queue.1.clear();
                        $(
                            $parser.error_queue.1.push($token);
                        )*
                        $parser.error_queue.0 = p;
                    }
                },
                None => ()
            } 
            $parser.rewind($rewind_point);
            return None; 
        }
        x.unwrap()
    }};
}

macro_rules! consume_func {
    ($parser:expr, $rewind_point:expr, $func:ident) => {{
        let retvalue = $parser.$func();
        if retvalue.is_none() {
            $parser.rewind($rewind_point);
            return None; 
        }
        retvalue.unwrap()
    }};
}

macro_rules! unwrap_token {
    ($token:expr, Vec::<u8>) => {{
        match $token {
            Token::ID(x) => x.to_vec(),
            _ => Vec::new()
        }
    }};
    ($token:expr, i32) => {{
        match $token {
            Token::IConst(x) => x,
            _ => 0i32
        }
    }};
    ($token:expr, Type) => {{
        match $token {
            Token::Type(x) => x,
            _ => Type::UnitStream
        }
    }};
}

impl Parser {
    // Utility methods
    fn peek(&self) -> Option<Token> {
        if self.index < self.tokenlist.len() {
            Some(self.tokenlist[self.index].0.clone())
        } else {
            None
        }
    }

    fn pop(&mut self) -> Option<Token> {
        if self.index < self.tokenlist.len() {
            self.advance();
            Some(self.tokenlist[self.index-1].0.clone())
        } else {
            None
        }
    }

    fn curr_pos(&self) -> Option<Position> {
        if self.index < self.tokenlist.len() {
            Some(self.tokenlist[self.index].1.clone())
        } else {
            None
        }
    }

    // Error Handling
    fn report_error(&mut self, t:Token) {
        match self.curr_pos() {
            Some(p) => {
                if p == self.error_queue.0 {
                    self.error_queue.1.push(t);
                } else if p > self.error_queue.0 {
                    self.error_queue.1.clear();
                    self.error_queue.1.push(t);
                    self.error_queue.0 = p;
                }},
            None => ()
        }
    }

    fn record_error(&mut self) {
        let (p, mut el) = self.error_queue.clone();
        if p != Position::new(0,0) {
            el.sort_by(|a, b| a.partial_cmp(b).unwrap());
            el.dedup();
            self.errors.insert(p, el);
        }
    }

    fn forward_to_statement(&mut self) {
        // "Panic Button", increment Position to next potential Statement start
        fn is_stat_term_token(t: Token) -> bool {
            use Token::*;
            return t == NewLine;
        }
        
        while self.peek().is_some() {
            if let Some(i1) = self.peek() {
                if is_stat_term_token(i1) {
                    self.advance();
                    break;
                } else {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }
    } 
    
    fn lookahead(&self, positions : usize) -> Option<Token> {
        let index = self.index + positions; 
        if index < self.tokenlist.len() {
            Some(self.tokenlist[index].0.clone())
        } else {
            None
        }
    }

    fn advance(&mut self) {
        self.index += 1;
    }
    
    fn token_matches (&self, lookahead: usize, t: &Token) -> bool {
        use std::mem::discriminant;
        if let Some(u) = self.lookahead(lookahead) {
            discriminant(&u) == discriminant(&t)
        } else {
            false
        }
    }

    fn token_matches_at_index(&self, index: usize, t: &Token) -> bool {
        use std::mem::discriminant;
        if index < self.tokenlist.len() {   
            let (u, _p) = &self.tokenlist[index];
            return discriminant(u) == discriminant(&t);
        }
        false
    }

    fn rewind(&mut self, index: usize) {
        self.index = index;
    } 

    fn grammar_stat_or(&mut self, nts: Vec<fn(&mut Parser) ->Option<Box<dyn Statement>>>)
                       -> Option<Box<dyn Statement>>
    {
        let mut node : Option<Box<dyn Statement>> = None;
        for f in nts {
            node = f(self);
            if node.is_some() {
                break;
            }
        }
        node
    }

    fn grammar_expr_or(&mut self, nts: Vec<fn(&mut Parser)->Option<Box<dyn Expression>>>)
                  -> Option<Box<dyn Expression>>
    {
        let mut node : Option<Box<dyn Expression>> = None;
        for f in nts {
            node = f(self);
            if node.is_some() {
                break;
            }
        }
        node
    }

    // Recursive Descent Methods

    fn id_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        return Some(Box::new(IDExpr {
            id,
            pos: pos.unwrap()
        }));
    }

    fn unit_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        consume!(self, rewind_point, Token::BraceL);
        consume!(self, rewind_point, Token::BraceR);

        return Some(Box::new(UnitExpr {
            pos: pos.unwrap()
        }));
    }

    fn default_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        let value = consume!(self, rewind_point, Token::IConst(0));
        let value = unwrap_token!(value, i32);

        return Some(Box::new(DefaultExpr {
            pos: pos.unwrap(),
            value
        }));
    }

    fn slift_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;

        let left = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let left = unwrap_token!(left, Vec::<u8>);

        let pos = self.curr_pos();
        let op = consume_any!(self, rewind_point, Token::OpPlus, Token::OpMinus,
            Token::OpMul, Token::OpDiv, Token::OpModulo);

        let right = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let right = unwrap_token!(right, Vec::<u8>);

        return Some(Box::new(SLiftExpr{
            op: BinOp::get_from_token(op).unwrap(),
            left,
            right,
            pos: pos.unwrap()
        }))
    }

    // arithmetic operation with value left
    fn arith_left_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;

        let val = consume!(self, rewind_point, Token::IConst(0));
        let val = unwrap_token!(val, i32);

        let pos = self.curr_pos();
        let op = consume_any!(self, rewind_point, Token::OpPlus, Token::OpMinus,
            Token::OpMul, Token::OpDiv, Token::OpModulo);

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        return Some(Box::new(ArithExpr{
            op: BinOp::get_from_token(op).unwrap(),
            val,
            id,
            val_left: true,
            pos: pos.unwrap()
        }));
    }

    // arithmetic operation with value right
    fn arith_right_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        let pos = self.curr_pos();
        let op = consume_any!(self, rewind_point, Token::OpPlus, Token::OpMinus,
            Token::OpMul, Token::OpDiv, Token::OpModulo);

        let val = consume!(self, rewind_point, Token::IConst(0));
        let val = unwrap_token!(val, i32);

        return Some(Box::new(ArithExpr{
            op: BinOp::get_from_token(op).unwrap(),
            val,
            id,
            val_left: false,
            pos: pos.unwrap()
        }));
    }

    fn un_func_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        let op = consume_any!(self, rewind_point, Token::Time, Token::Count);

        consume!(self, rewind_point, Token::BraceL);

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        consume!(self, rewind_point, Token::BraceR);

        let x : Option<Box<dyn Expression>>;
        match &op {
            Token::Time => {
                x = Some(Box::new(TimeExpr {
                    id,
                    pos: pos.unwrap()
                }))
            },
            Token::Count => {
                x = Some(Box::new(CountExpr {
                    id,
                    pos: pos.unwrap()
                }))
            },
            _ => {
                unreachable!();
            }
        }
        return x;
    }

    fn bin_func_expr(&mut self) -> Option<Box<dyn Expression>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        let op = consume_any!(self, rewind_point, Token::Delay, Token::Last, Token::Merge);

        consume!(self, rewind_point, Token::BraceL);

        let left = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let left = unwrap_token!(left, Vec::<u8>);

        consume!(self, rewind_point, Token::Comma);

        let right = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let right = unwrap_token!(right, Vec::<u8>);

        consume!(self, rewind_point, Token::BraceR);

        let x : Option<Box<dyn Expression>>;
        match &op {
            Token::Delay => x = Some(Box::new(DelayExpr {
                left,
                right,
                pos: pos.unwrap()
            })),
            Token::Last => x = Some(Box::new(LastExpr {
                left,
                right,
                pos: pos.unwrap()
            })),
            Token::Merge => x = Some(Box::new(MergeExpr {
                left,
                right,
                pos: pos.unwrap()
            })),
            _ => unreachable!()
        };
        return x;
    }

    fn expr(&mut self) -> Option<Box<dyn Expression>> {
        return self.grammar_expr_or(vec![
            // must be first due to ambiguities:
            Parser::arith_left_expr,
            Parser::arith_right_expr,
            Parser::slift_expr,
            // only from now id and default
            Parser::id_expr,
            Parser::unit_expr,
            Parser::default_expr,
            Parser::un_func_expr,
            Parser::bin_func_expr,
        ]);
    }

    fn def_stat(&mut self) -> Option<Box<dyn Statement>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        consume!(self, rewind_point, Token::Define);

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        consume!(self, rewind_point, Token::Assign);

        let value = consume_func!(self,rewind_point,expr);

        consume!(self, rewind_point, Token::NewLine);
        return Some(Box::new(DefStat{
            id,
            value,
            pos: pos.unwrap()
        }));
    }

    fn in_stat(&mut self) -> Option<Box<dyn Statement>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        consume!(self, rewind_point, Token::In);

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        consume!(self, rewind_point, Token::Colon);

        let t = consume!(self, rewind_point, Token::Type(Type::UnitStream));
        let t = unwrap_token!(t, Type);

        consume!(self, rewind_point, Token::NewLine);
        return Some(Box::new(InputStat{
            id,
            t,
            pos: pos.unwrap()
        }));
    }

    fn out_stat(&mut self) -> Option<Box<dyn Statement>> {
        let rewind_point = self.index;
        let pos = self.curr_pos();

        consume!(self, rewind_point, Token::Out);

        let id = consume!(self, rewind_point, Token::ID(Vec::<u8>::new()));
        let id = unwrap_token!(id, Vec::<u8>);

        consume!(self, rewind_point, Token::NewLine);
        return Some(Box::new(OutputStat{
            id,
            pos: pos.unwrap()
        }));
    }

    fn stat(&mut self) -> Option<Box<dyn Statement>> {
        let stat = self.grammar_stat_or(vec![
            Parser::def_stat,
            Parser::in_stat,
            Parser::out_stat
        ]);
        return stat;
    }

    fn new(tokenlist: Vec<(Token, Position)>) -> Parser {
        Parser {
            index: 0,
            tokenlist,
            errors: HashMap::new(),
            error_queue: (Position::new(0,0),vec!())
        }
    }

    fn get_token(&self, pos: &Position) -> Token {
        self.tokenlist.iter().find(|(_, p)| p == pos)
            .expect("Attempted to get token at unknown position").0.clone()
    }

    pub fn parse (&mut self) -> Vec<Box<dyn Statement>>{
        //consume_stat_func!(self, stat);
        let mut stats : Vec<Box<dyn Statement>> = vec!();
        while self.peek().is_some() {
            let stat = self.stat();
            if stat.is_none() {
                self.record_error();
                self.forward_to_statement();
            } else {
                if stat.is_some() {
                    stats.push(stat.unwrap());
                } else {
                    break;
                }
            }
        }
        stats
    }

}

pub fn parse(token_list: Vec<(Token, Position)>) -> Option<Vec<Box<dyn Statement>>>{
    let mut p = Parser::new(token_list);
    let result = p.parse();
    if p.errors.len() != 0 {
        eprintln!("Parser Error\n");
        let mut errors : Vec<(Position, Vec<Token>)> = p.errors.clone().into_iter().collect();
        errors.sort_by(|(p1, _), (p2, _)| p1.partial_cmp(p2).unwrap());
        for (pos, expected_tokens) in errors { 
            eprint!("\t{}: Unexpected token \"{}\", expected: \n\t\t", pos, p.get_token(&pos));
            for (i, tok) in expected_tokens.iter().enumerate() {
                if i == expected_tokens.len()-1 {
                    eprint!("{}\n\n", tok);
                } else {
                    eprint!("{}, ", tok);
                }
            }
        }
        return None;
    }
    return Some(result);
}

#[cfg(test)]
pub mod test {
    #[test]
    pub fn ast() {
        use super::super::lexer::{Token::*, Position};
        use super::super::ast::{Type::*, ast_to_json_string};
        let p = Position::new(0,0);
        let q = Position::new(3,3);
        let tokenlist = vec!(
            (In,p),
            (ID(b"x".to_vec()),p),
            (Colon,p),
            (Type(IntStream), p),
            (NewLine, p),
            (Define,q),
            (ID(b"y".to_vec()),p),
            (Assign,p),
            (ID(b"x".to_vec()), p),
            (OpModulo, p),
            (IConst(-133), p),
            (NewLine, p),
            (Define,p),
            (ID(b"y".to_vec()),p),
            (Assign,p),
            (IConst(-133), p),
            (OpPlus, p),
            (ID(b"x".to_vec()), p),
            (NewLine, p),
            (Define, p),
            (ID(b"z".to_vec()),q),
            (Assign,p),
            (IConst(420), p),
            (NewLine, p),
            (Define, p),
            (ID(b"z".to_vec()),q),
            (Assign,p),
            (Merge, q),
            (BraceL,p),
            (ID(b"x".to_vec()), q),
            (Comma, p),
            (ID(b"y".to_vec()),p),
            (BraceR,p),
            (NewLine,p),
            (Out, p),
            (ID(b"z".to_vec()),q),
            (NewLine,p)
        );
        let t = super::parse(tokenlist);
        let cmp_val = "{\n\"0\": { \n\"node\": \"Input\", \n\"id\": \"x\", \n\"type\": \
        \"IntStream\", \n\"pos\": { \"line\": 0, \"pos\": 0 }\n},\n\"1\": { \n\"node\": \"Def\
        ine\", \n\"id\": \"y\", \n\"value\": { \n\"node\": \"Arith\", \n\"op\": \"Modulo\", \n\"i\
        d\": \"x\",\n \"val\": \"-133\",\n\"val_left\": \"false\",\n\"pos\": { \"line\": 0, \"p\
        os\": 0 }\n}, \n\"pos\": { \"line\": 3, \"pos\": 3 }\n},\n\"2\": { \n\"node\": \"Def\
        ine\", \n\"id\": \"y\", \n\"value\": { \n\"node\": \"Arith\", \n\"op\": \"Plus\", \n\"i\
        d\": \"x\",\n \"val\": \"-133\",\n\"val_left\": \"true\",\n\"pos\": { \"line\": 0, \"p\
        os\": 0 }\n}, \n\"pos\": { \"line\": 0, \"pos\": 0 }\n},\n\"3\": { \n\"node\": \"Defin\
        e\", \n\"id\": \"z\", \n\"value\": { \n\"node\": \"Default\", \n\"value\": \"420\", \n\"p\
        os\": { \"line\": 0, \"pos\": 0 }\n}, \n\"pos\": { \"line\": 0, \"pos\": 0 }\n},\n\"4\": \
        { \n\"node\": \"Define\", \n\"id\": \"z\", \n\"value\": { \n\"node\": \"Merge\", \n\"lef\
        t\": \"x\",\n\"right\": \"y\",\n\"pos\": { \"line\": 3, \"pos\": 3 }\n}, \n\"pos\": { \"li\
        ne\": 0, \"pos\": 0 }\n},\n\"5\": { \n\"node\": \"Output\", \n\"id\": \"z\",\n\"pos\": \
        { \"line\": 0, \"pos\": 0 }\n}\n}";

        let t = t.unwrap();
        assert_eq!(ast_to_json_string(&t), cmp_val);
    }

    #[test]
    pub fn error_handling() {
        use super::super::lexer::{Token::*, Position};
        use super::super::ast::Type::*;
        use super::Parser;
        use std::collections::HashMap;

        let token_list = vec!(
            (Define, Position::new(0,0)),
            (ID(b"l".to_vec()), Position::new(0,1)),
            (Assign, Position::new(0,2)),
            (IConst(1), Position::new(0,2)),
            (NewLine, Position::new(0,3)),
            (Define, Position::new(1,0)),
            (ID(b"a".to_vec()),Position::new(1,1)),
            (Assign,Position::new(1,2)),
            (ID(b"b".to_vec()),Position::new(1,3)),
            (OpModulo,Position::new(1,4)),
            (IConst(3),Position::new(1,5)),
            (NewLine,Position::new(1,6)),
            //(,Position::new(2,2)),
            //(ID(b"b".to_vec()),Position::new(2,1)),
            (Assign,Position::new(2,2)),
            (Type(IntStream),Position::new(2,3)),
            (NewLine,Position::new(2,4)),
            //(IConst(3),Position::new(2,5)),
            (Define, Position::new(4,1)),
            (ID(b"a".to_vec()),Position::new(5,1)),
            (Assign,Position::new(5,4)),
            //(IConst(3),Position::new(1,5)),
            (NewLine,Position::new(6,6)),
        );
        let mut parser = Parser {
            index: 0,
            tokenlist: token_list,
            errors: HashMap::new(),
            error_queue: (Position::new(0,0),vec!())
        };
        parser.parse();
        let mut m = HashMap::<Position,Vec<super::super::lexer::Token>>::new();
        m.insert(Position::new(2, 2), vec![Define, In, Out]);
        m.insert(Position::new(6, 6), vec![ID(vec!()), IConst(0), Merge, Last, Delay, Time, Count, BraceL]);
        assert_eq!(parser.errors, m);
    }
}