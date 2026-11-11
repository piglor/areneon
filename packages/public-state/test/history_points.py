"""The Python reducer's history facts at every request, for the TS/Python parity test.

Reads jobs as JSON on stdin ``{"engine_data": path, "jobs": [{"id", "player", "lines", "team"}]}``
and writes ``{id: [{"line": index, "facts": {...}}]}``: the facts before each `|request|` line (the
cut the live-engine parity test and the world facts use). A reducer refusal ends a job with
``{"refused": text}`` after the points it reached. Test support only: it decides nothing.
"""

import json
import sys

from areneon_ml.reducer import ChannelReducer, ReducerError, Vocabulary

request = json.load(sys.stdin)
with open(request["engine_data"], encoding="utf-8") as handle:
    vocab = Vocabulary.from_engine_data(json.load(handle))
out = {}
for job in request["jobs"]:
    points = []
    reducer = ChannelReducer(job["player"], job["team"], vocab)
    refused = None
    for index, raw in enumerate(job["lines"]):
        if raw.startswith("|request|") and reducer.state.history is not None:
            points.append({"line": index, "facts": reducer.state.history.facts()})
        try:
            reducer.feed(raw)
        except ReducerError as error:
            refused = str(error)
            break
    out[job["id"]] = {"points": points, "refused": refused}
json.dump(out, sys.stdout)
