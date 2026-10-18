"""Print RFC 8785 (rfc8785 package) output per line of harness/jcs-vectors.txt."""

import json
import sys

import rfc8785

with open(sys.argv[1], encoding="utf-8") as vectors:
    for line in vectors:
        try:
            print(rfc8785.dumps(json.loads(line)).decode("utf-8"))
        except Exception as exc:  # recorded per vector
            print(f"ERROR {type(exc).__name__}")
