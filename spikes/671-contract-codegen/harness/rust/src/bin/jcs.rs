//! Print serde_json_canonicalizer output per line of harness/jcs-vectors.txt.
use std::{env, fs};

fn main() {
    let path = env::args().nth(1).expect("usage: jcs <vectors>");
    for line in fs::read_to_string(path).unwrap().lines().filter(|l| !l.is_empty()) {
        match serde_json::from_str::<serde_json::Value>(line) {
            Err(e) => println!("ERROR parse {e}"),
            Ok(v) => match serde_json_canonicalizer::to_string(&v) {
                Ok(s) => println!("{s}"),
                Err(e) => println!("ERROR {e}"),
            },
        }
    }
}
