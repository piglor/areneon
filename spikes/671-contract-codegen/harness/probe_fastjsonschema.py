"""Does fastjsonschema honour 2020-12-only keywords? Compared with jsonschema."""

import fastjsonschema
import jsonschema

S = "https://json-schema.org/draft/2020-12/schema"
cases = {
    "prefixItems": (
        {"$schema": S, "type": "array", "prefixItems": [{"type": "integer"}], "items": False},
        ["x"],
    ),
    "unevaluatedProperties": (
        {"$schema": S, "type": "object", "properties": {"a": {}}, "unevaluatedProperties": False},
        {"a": 1, "b": 2},
    ),
    "dependentRequired": ({"$schema": S, "dependentRequired": {"a": ["b"]}}, {"a": 1}),
}
for k, (s, inst) in cases.items():
    try:
        fastjsonschema.compile(s)(inst)
        f = "accepted"
    except fastjsonschema.JsonSchemaException:
        f = "rejected"
    except Exception as e:
        f = f"error {type(e).__name__}: {e}"
    print(
        k,
        "fastjsonschema:",
        f,
        "| jsonschema:",
        "accepted" if jsonschema.Draft202012Validator(s).is_valid(inst) else "rejected",
    )
