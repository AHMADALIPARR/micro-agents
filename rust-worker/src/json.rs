//! Minimal JSON for the line protocol (flat objects of strings / numbers /
//! booleans / null; arrays of strings on output). No serde: the crate builds
//! offline with zero crates.io dependencies.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

impl Val {
    pub fn as_str(&self) -> Option<&str> {
        if let Val::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Val::Num(n) = self { if *n >= 0.0 { Some(*n as u64) } else { None } } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Val::Bool(b) = self { Some(*b) } else { None }
    }
}

pub fn parse_object(s: &str) -> Result<HashMap<String, Val>, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut m = HashMap::new();
    ws(b, &mut i);
    expect(b, &mut i, b'{')?;
    ws(b, &mut i);
    if peek(b, i) == Some(b'}') {
        return Ok(m);
    }
    loop {
        ws(b, &mut i);
        let k = string(b, &mut i)?;
        ws(b, &mut i);
        expect(b, &mut i, b':')?;
        ws(b, &mut i);
        let v = value(b, &mut i)?;
        m.insert(k, v);
        ws(b, &mut i);
        match peek(b, i) {
            Some(b',') => i += 1,
            Some(b'}') => return Ok(m),
            _ => return Err(format!("expected , or }} at {}", i)),
        }
    }
}

fn peek(b: &[u8], i: usize) -> Option<u8> {
    b.get(i).copied()
}
fn ws(b: &[u8], i: &mut usize) {
    while matches!(peek(b, *i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        *i += 1;
    }
}
fn expect(b: &[u8], i: &mut usize, c: u8) -> Result<(), String> {
    if peek(b, *i) == Some(c) { *i += 1; Ok(()) } else { Err(format!("expected '{}' at {}", c as char, i)) }
}

fn string(b: &[u8], i: &mut usize) -> Result<String, String> {
    expect(b, i, b'"')?;
    let mut out = Vec::new();
    loop {
        match peek(b, *i) {
            None => return Err("unterminated string".into()),
            Some(b'"') => {
                *i += 1;
                return String::from_utf8(out).map_err(|e| e.to_string());
            }
            Some(b'\\') => {
                *i += 1;
                let c = peek(b, *i).ok_or("bad escape")?;
                *i += 1;
                match c {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i..*i + 4).ok_or("bad \\u")?).map_err(|e| e.to_string())?;
                        let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                        *i += 4;
                        let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                        let mut tmp = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    }
                    _ => return Err("bad escape".into()),
                }
            }
            Some(c) => {
                out.push(c);
                *i += 1;
            }
        }
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Val, String> {
    match peek(b, *i) {
        Some(b'"') => Ok(Val::Str(string(b, i)?)),
        Some(b't') if b[*i..].starts_with(b"true") => { *i += 4; Ok(Val::Bool(true)) }
        Some(b'f') if b[*i..].starts_with(b"false") => { *i += 5; Ok(Val::Bool(false)) }
        Some(b'n') if b[*i..].starts_with(b"null") => { *i += 4; Ok(Val::Null) }
        Some(c) if c == b'-' || c.is_ascii_digit() => {
            let st = *i;
            while matches!(peek(b, *i), Some(c) if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit()) {
                *i += 1;
            }
            std::str::from_utf8(&b[st..*i]).unwrap().parse::<f64>().map(Val::Num).map_err(|e| e.to_string())
        }
        _ => Err(format!("unexpected value at {}", i)),
    }
}

pub fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub fn esc_list(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| esc(s)).collect::<Vec<_>>().join(","))
}
