//! Rust checks: `jsonschema` verdicts and the generated types' parse
//! verdicts + JCS round trip. One JSON line per sample.
//! Run: cargo run -- <out>
#![allow(dead_code)]

mod quicktype_types;
mod typify_types;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{env, fs, path::Path};

fn round_trip<T: DeserializeOwned + Serialize>(raw: &str, original: &Value) -> Value {
    match serde_json::from_str::<T>(raw) {
        Err(_) => json!({ "parse": false }),
        Ok(parsed) => {
            let back = serde_json::to_value(&parsed).expect("serialise");
            let equal = serde_json_canonicalizer::to_string(&back).unwrap()
                == serde_json_canonicalizer::to_string(original).unwrap();
            json!({ "parse": true, "jcs_equal": equal })
        }
    }
}

fn main() {
    let out = env::args().nth(1).expect("usage: harness <out>");
    let spike = Path::new(env!("CARGO_MANIFEST_DIR")).join(env::var("SPIKE_DIR").expect("SPIKE_DIR"));
    let schema: Value =
        serde_json::from_str(&fs::read_to_string(spike.join("schema/contracts.schema.json")).unwrap()).unwrap();
    let validator = jsonschema::draft202012::new(&schema).expect("schema compiles");
    for kind in ["valid", "invalid"] {
        let dir = Path::new(&out).join("samples").join(kind);
        let mut files: Vec<_> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        for path in files {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let key = name.split('.').next().unwrap().to_string();
            let value: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let catalog = json!({ key: value });
            let raw = catalog.to_string();
            let row = json!({
                "lang": "rs",
                "sample": format!("{kind}/{name}"),
                "jsonschema": validator.is_valid(&catalog),
                "typify": round_trip::<typify_types::AreneonContracts>(&raw, &catalog),
                "quicktype": round_trip::<quicktype_types::AreneonContracts>(&raw, &catalog),
            });
            println!("{row}");
        }
    }
    // request_hash = SHA-256 of the canonical request without rqid.
    let valid = Path::new(&out).join("samples").join("valid");
    let mut fixtures: Vec<_> = fs::read_dir(&valid).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("request_fixture.")).collect();
    fixtures.sort();
    for path in fixtures {
        let fixture: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let mut request = fixture["request"].clone();
        request.as_object_mut().unwrap().remove("rqid");
        let digest = sha256::digest(serde_json_canonicalizer::to_string(&request).unwrap());
        eprintln!("{}", json!({ "lang": "rs", "request_hash": path.file_name().unwrap().to_string_lossy(),
            "matches": digest == fixture["request_hash"].as_str().unwrap() }));
    }
}
