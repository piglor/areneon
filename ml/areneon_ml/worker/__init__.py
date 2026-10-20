"""The policy worker (ADR-010 §2): a child process that answers PolicyQuery lines on stdio.

It opens no socket, reads checkpoints only by id from the run directory, and
sees only the observation and candidates of a query: never a battle seed, an
opponent's hidden information or a CloneJudge verdict.
"""
