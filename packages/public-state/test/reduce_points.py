"""Reference states for the interpreter test: the Python reducer's state at chosen cut points.

Reads jobs as JSON on stdin ``{"engine_data": path, "jobs": [{"id", "player", "lines", "team",
"cuts"}]}`` and writes ``{id: {cut: state}}``. Test support only: it decides nothing.
"""

import json
import sys

from areneon_ml.reducer import ReducerError, Vocabulary, reduce_channel

request = json.load(sys.stdin)
with open(request["engine_data"], encoding="utf-8") as handle:
    vocab = Vocabulary.from_engine_data(json.load(handle))
out = {}
for job in request["jobs"]:
    states = {}
    for cut in job["cuts"]:
        try:
            states[str(cut)] = reduce_channel(
                job["player"], job["lines"][:cut], job["team"], vocab
            ).to_dict()
        except ReducerError as error:  # a refusal is a result: the interpreter must flag there
            states[str(cut)] = {"refused": str(error)}
    out[job["id"]] = states
json.dump(out, sys.stdout)
