use std::iter::Peekable;
use super::ast::Type;
use std::cmp::Ordering;

#[derive(Debug, PartialEq, Clone, PartialOrd)]
pub enum Token {
    ID(Vec<u8>),
    IConst(i32),
    Type(Type),
    Assign,
    OpPlus,
    OpMinus,
    OpMul,
    OpDiv,
    OpModulo,
    Merge,
    Last,
    Delay,
    Time,
    Count,
    NewLine,
    BraceL,
    BraceR,
    Comma,
    Define,
    In,
    Out,
    Colon
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::ID(_identifier) => write!(f, "Identifier"),
            Token::IConst(_int_val) => write!(f, "Integer Constant"),
            Token::Type(_type_val) => write!(f, "Type"),
            Token::Assign => write!(f, "="),
            Token::OpPlus => write!(f, "+"),
            Token::OpMinus => write!(f, "-"),
            Token::OpMul => write!(f, "*"),
            Token::OpDiv => write!(f, "/"),
            Token::OpModulo => write!(f, "%"),
            Token::Merge => write!(f, "merge"),
            Token::Last => write!(f, "last"),
            Token::Delay => write!(f, "delay"),
            Token::Time => write!(f, "time"),
            Token::Count => write!(f, "count"),
            Token::NewLine => write!(f, "\\n"),
            Token::BraceL => write!(f, "("),
            Token::BraceR => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Define => write!(f, "def"),
            Token::In => write!(f, "in"),
            Token::Colon => write!(f, ":"),
            Token::Out => write!(f, "out"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct Position {
    line : i32,
    pos : i32,
}

impl Position {
    pub fn new(line : i32, pos: i32) -> Position {
        Position {line, pos}
    }

    pub fn to_json_string(&self) -> String {
        format!("{{ \"line\": {}, \"pos\": {} }}", self.line, self.pos)
    }
}

impl std::cmp::PartialOrd for Position {
    fn partial_cmp(&self, other: &Position) -> Option<std::cmp::Ordering>{
        let linediff = self.line - other.line;
        if linediff > 0 {
            return Some(Ordering::Greater)
        } else if linediff < 0 {
            return Some(Ordering::Less)
        } else {
            let posdiff = self.pos - other.pos;
            if posdiff > 0 {
                return Some(Ordering::Greater)
            } else if posdiff < 0 {
                return Some(Ordering::Less)
            } else {
                return Some(Ordering::Equal)
            }
        }
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f,"line {}, col {}", self.line, self.pos) 
    }
}

struct Tokenizer<'a> {
    _source: &'a str,
    iter: Peekable<std::str::Bytes<'a>>,
    position: Position,
    tokens: Vec<(Token, Position)>,
    errors: Vec<(Position, String)>
}

impl Tokenizer<'_> {
    fn new(source: &str) -> Tokenizer {
        Tokenizer {
            _source: source,
            iter: source.bytes().peekable(),
            position: Position::new(1,1),
            tokens: vec!(),
            errors: vec!()
        }
    }

    fn peek(&mut self) -> Option<&u8> {
        self.iter.peek()
    }

    fn pop(&mut self) -> Option<u8> {
        self.position.pos += 1;
        self.iter.next() 
    }

    fn advance(&mut self) {
        self.position.pos += 1;
        self.iter.next();
    }

    fn push_advance(&mut self, token : Token)
    {
        self.tokens.push((token, self.position.clone()));
        self.advance();
    }

    fn new_line(&mut self) {
        // prohibit two consecutive new line tokens
        if !self.tokens.is_empty() {
            let (t, _) = self.tokens.last().unwrap();
            if t != &Token::NewLine {
                self.tokens.push((Token::NewLine, self.position.clone()))
            }
        }
        self.position.line += 1;
        self.position.pos = 1;
    }

    fn record_error(&mut self, error_msg: String) {
        self.errors.push((self.position, error_msg));
    }

    fn consume_whitespace(&mut self) {
        while self.peek().is_some() {
            match self.peek() {
                Some(c) => {
                    match c {
                        b' ' | b'\t' | b'\r' => self.advance(),
                        _ => break
                    }
                },
                None => break
            }
        }
    }

    fn capture_push_word(&mut self) {
        let mut word = vec!();
        let pos = self.position;
        word.push(self.pop().unwrap()); // not None because of check in tokenize()
        let mut event_type = false;
        while self.peek().is_some() {
            match self.peek() {
                Some(c) => match c {
                    b'a'..=b'z' 
                    | b'A'..=b'Z' 
                    | b'_' 
                    | b'0'..=b'9' => word.push(self.pop().unwrap()),
                    b'[' => {
                        word.push(self.pop().unwrap());
                        event_type = true;
                    },
                    b']' => {
                        if event_type {
                            word.push(self.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    _ => break
                }
                None => break
            }
        }
        if event_type {
            match &word[..] {
                b"Events[Unit]" => self.tokens.push((Token::Type(Type::UnitStream), pos)),
                b"Events[Int]" => self.tokens.push((Token::Type(Type::IntStream), pos)),
                _ => self.record_error(String::from("Failed to parse type"))
            }
        } else {
            self.tokens.push((get_word_token(word),pos));
        }
    }

    fn capture_push_number(&mut self) {
        let pos = self.position;

        let mut number = vec!();
        while self.peek().is_some() {
            match self.peek() {
                Some(c) => match c {
                    b'0'..=b'9' => number.push(self.pop().unwrap()),
                    _ => break
                }
                None => break
            }
        }

        let number = String::from_utf8_lossy(&number);
        // Guaranteed not to be none due to validation
        let number = i32::from_str_radix(&number, 10).unwrap();
        self.tokens.push((Token::IConst(number), pos));
    }

    fn consume_comment(&mut self) {
        // consume starting hashtag
        let mut char = self.pop();

        // consume until (including) end of line
        while char.is_some() {
            if self.peek() == Some(&b'\n') {
                self.pop();
                self.new_line();
                return;
            } else {
                char = self.pop();
            }
        }
    }

    fn has_error(&self) -> bool {
        !self.errors.is_empty() 
    }

    fn tokenize(&mut self) {
        while self.peek().is_some() {
            match self.peek() {
                Some(c) => {
                    match c {
                        b'#' => self.consume_comment(),
                        b'+' => self.push_advance(Token::OpPlus),
                        b'*' => self.push_advance(Token::OpMul),
                        b'-' => self.push_advance(Token::OpMinus),
                        b',' => self.push_advance(Token::Comma),
                        b'%' => self.push_advance(Token::OpModulo),
                        b'/' => self.push_advance(Token::OpDiv),
                        b'(' => self.push_advance(Token::BraceL),
                        b')' => self.push_advance(Token::BraceR),
                        b'=' => self.push_advance(Token::Assign),
                        b':' => self.push_advance(Token::Colon),
                        b'\n' => {
                            self.new_line();
                            self.iter.next();
                        },
                        b' ' | b'\r' | b'\t' => self.consume_whitespace(),
                        b'0'..=b'9' => self.capture_push_number(),
                        b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.capture_push_word(),
                        _ => {
                            self.record_error(String::from("Invalid character"));
                            self.advance();
                        }
                    }
                }
                None => () // finished 
            }
        }
    }
}

fn get_word_token(word : Vec<u8>) -> Token {
    match &word[..] {
        b"delay" => Token::Delay,
        b"last" => Token::Last,
        b"time" => Token::Time,
        b"merge" => Token::Merge,
        b"count" => Token::Count,
        b"def" => Token::Define,
        b"in" => Token::In,
        b"out" => Token::Out,
        _ => Token::ID(word)
    }
}


pub fn tokenize_internal(source : &str) -> Option<(Vec<(Token, Position)>, Position)> {
    let mut t = Tokenizer::new(source);
    t.tokenize();
    if t.has_error() {
        eprintln!("Lexer Error\n");
        for error in &t.errors {
            eprintln!("\t{}: {}", error.0, error.1);
        }
        return None;
    }
    Some((t.tokens, t.position))
}

pub fn tokenize(source : &str) -> Option<Vec<(Token, Position)>> {
    let tokens = tokenize_internal(source);
    // add newline if there is none at EOF
    return match tokens {
        Some((mut t,p)) => {
            if !t.is_empty() {
                if let Some((tok,_)) = t.last() {
                    if tok != &Token::NewLine {
                        t.push((Token::NewLine, p));
                    }
                }
            }
            Some(t)
        },
        None => None
    }
}

    
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn multiple_tokens() {
        use super::Token::*;
        use super::Type;
        let text = "in x : Events[Int] \n in y : Events[Int] \n def z = x + y \n out z";
        let full_list = tokenize_internal(text);
        let (list,_) = full_list.unwrap();
        let t_list = list.iter().map(|(a, _b)| a ).collect::<Vec<_>>();
        assert_eq!(t_list, vec!(
            &In,
            &ID(b"x".to_vec()),
            &Colon,
            &Type(Type::IntStream),
            &NewLine,
            &In,
            &ID(b"y".to_vec()),
            &Colon,
            &Type(Type::IntStream),
            &NewLine,
            &Define,
            &ID(b"z".to_vec()),
            &Assign,
            &ID(b"x".to_vec()),
            &OpPlus,
            &ID(b"y".to_vec()),
            &NewLine,
            &Out,
            &ID(b"z".to_vec())
            )
        )
    }

    #[test]
    fn position() {
        use super::Token::*;
        let text = "def x = \n ( \n y \t = x + 15 / 2 \t / x y0 % \n g1 = \t  # asda*//*\n 3423 \n \n merge(x,y)";
        let (full_list, _) = tokenize_internal(text).unwrap();
        assert_eq!(full_list[0], (Define, Position::new(1,1)));
        assert_eq!(full_list[7], (Assign, Position::new(3,6)));
        assert_eq!(full_list[21], (IConst(3423), Position::new(5,2)));
        assert_eq!(full_list[23], (Merge, Position::new(7,2)));
    }

    #[test]
    fn single_token() {
        macro_rules! ut {
            ($tup : expr) => {
                {
                    let (t, _ ) = $tup;
                    t
                }
            }
        }
        assert_eq!(ut!(tokenize_internal("+").unwrap()), vec!((Token::OpPlus, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("-").unwrap()), vec!((Token::OpMinus, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal(",").unwrap()), vec!((Token::Comma, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("(").unwrap()), vec!((Token::BraceL, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal(")").unwrap()), vec!((Token::BraceR, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("*").unwrap()), vec!((Token::OpMul, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("=").unwrap()), vec!((Token::Assign, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("/").unwrap()), vec!((Token::OpDiv, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("%").unwrap()), vec!((Token::OpModulo, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("G3RdA").unwrap()), vec!((Token::ID(b"G3RdA".to_vec()), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("hug_0").unwrap()), vec!((Token::ID(b"hug_0".to_vec()), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("y").unwrap()), vec!((Token::ID(b"y".to_vec()), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("x").unwrap()), vec!((Token::ID(b"x".to_vec()), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("Events[Unit]").unwrap()), vec!((Token::Type(Type::UnitStream), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("Events[Int]").unwrap()), vec!((Token::Type(Type::IntStream), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("def").unwrap()), vec!((Token::Define, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("in").unwrap()), vec!((Token::In, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("out").unwrap()), vec!((Token::Out, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal(":").unwrap()), vec!((Token::Colon, Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("1336231").unwrap()), vec!((Token::IConst(1336231), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("1").unwrap()), vec!((Token::IConst(1), Position::new(1, 1))));
        assert_eq!(ut!(tokenize_internal("\n").unwrap()), vec!());
        assert_eq!(ut!(tokenize_internal(" ").unwrap()), vec!());
        assert_eq!(ut!(tokenize_internal("\t").unwrap()), vec!());
        assert_eq!(ut!(tokenize_internal("# asdf\n").unwrap()), vec!());
        assert_eq!(ut!(tokenize_internal("#asdf").unwrap()), vec!());
    }

    #[test]
    fn newline() {
        let t = tokenize(")").unwrap();
        assert_eq!(t, vec!((Token::BraceR, Position::new(1,1)),(Token::NewLine, Position::new(1,2))));
        let u = tokenize("#").unwrap();
        assert_eq!(u, vec!());
    }

}