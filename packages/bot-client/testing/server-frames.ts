import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  actorRule,
  decodeRequest,
  type Engine,
  legalChoiceSet,
  type PokemonSet,
  resolveFormat,
  type VersionManifest,
  validateTeam,
} from "@areneon/showdown-bridge";

/**
 * Recorded server frames for the protocol reader's tests (CC0, see
 * fixtures/LICENSE and the README in FRAMES_DIR). No server is started and
 * nothing connects anywhere: the battle frames are the pinned engine's own
 * per-player output, framed the way the pinned server frames it.
 */
export const FRAMES_DIR = fileURLToPath(new URL("../../../fixtures/bot-client/frames", import.meta.url));

const CORPUS_CASE = fileURLToPath(
  new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/choices/maybe-trapped.json", import.meta.url),
);

/** A battle room id of the shape the server gives one (`server/rooms.ts`). */
export const BATTLE_ROOM = "battle-gen9championsvgc2026regmc-1";

/**
 * A choice the engine refuses as `[Invalid choice]`: slot 1 has no move 9.
 * The tests use the engine's real refusal of it to stand in for an
 * enumerator defect, which by construction never happens on its own.
 */
export const INVALID_CHOICE = "move 9, move 1 1";

/** Sent while p1 has a `wait` request: the sim answers that it is not p1's turn. */
export const OUT_OF_TURN_CHOICE = "default";

/** The engine-derived battle, as one player's connection receives it. */
export interface BattleFrames {
  provenance: string;
  manifest: VersionManifest;
  seed: string;
  room: string;
  player: "p1";
  /** Every choice p1's connection sent, in order, with the frame count seen before it. */
  sent: { after_frame: number; choice: string }[];
  frames: string[];
}

/** Frames only the server layer sends, written from the pinned server source. */
export interface ServerLayerFrames {
  provenance: string;
  room: string;
  frames: Record<string, { source: string; frame: string }>;
}

export interface FrameFixtures {
  "battle.json": BattleFrames;
  "server-layer.json": ServerLayerFrames;
}

interface CorpusCase {
  seed: string;
  teams: Record<"p1" | "p2", PokemonSet[]>;
  log: { player: "p1" | "p2"; choice: string }[];
}

type Player = "p1" | "p2";

/**
 * Plays the choice corpus's maybe-trapped battle (foe Mega Gengar may have
 * Shadow Tag) on the pinned in-process sim and frames p1's own stream as the
 * pinned server sends it to p1's connection.
 *
 * p1 replays the case's log, then submits a pending switch (the engine
 * answers `[Unavailable choice]` and an updated request), then
 * `INVALID_CHOICE` (`[Invalid choice]`), then plays the shared actor rule;
 * on its first `wait` request it sends `OUT_OF_TURN_CHOICE`. p2 replays the
 * log, then plays the actor rule. The battle runs to its end.
 */
export async function captureFrames(engine: Engine): Promise<FrameFixtures> {
  const spec = JSON.parse(readFileSync(CORPUS_CASE, "utf8")) as CorpusCase;
  const format = resolveFormat(engine, "gen9championsvgc2026regmc");
  for (const player of ["p1", "p2"] as const) {
    const verdict = validateTeam(engine, format, spec.teams[player]);
    if (!verdict.valid) throw new Error(`${player} team refused: ${verdict.problems.join("; ")}`);
  }
  const battleStream = new engine.sim.BattleStream();
  const streams = engine.sim.getPlayerStreams(battleStream);
  const queues: Record<Player, string[]> = {
    p1: spec.log.filter((e) => e.player === "p1").map((e) => e.choice),
    p2: spec.log.filter((e) => e.player === "p2").map((e) => e.choice),
  };
  /** Each player's stream: one chunk per message, `|t:|` lines dropped. */
  const chunks: Record<Player, string[][]> = { p1: [], p2: [] };
  const sent: BattleFrames["sent"] = [];
  let stage: "log" | "pending" | "invalid" | "rule" = "log";
  let afterInvalid = "";
  let outOfTurnSent = false;

  const choose = (player: Player, raw: string, seq: number): string => {
    const decoded = decodeRequest(raw);
    const legal = legalChoiceSet({
      request: { ...decoded, request_seq: seq },
      showdown_commit: engine.commit,
      judge: null,
    });
    const logged = queues[player].shift();
    if (logged !== undefined) return logged;
    if (player === "p2") return actorRule(legal);
    if (stage === "log" && legal.pending.length > 0) {
      stage = "pending";
      return legal.pending[0] as string;
    }
    if (stage === "pending") {
      stage = "invalid";
      afterInvalid = actorRule(legal);
      return INVALID_CHOICE;
    }
    stage = "rule";
    return actorRule(legal);
  };

  const play = async (player: Player): Promise<void> => {
    const stream = streams[player];
    let seq = 0;
    for await (const chunk of stream) {
      const lines = chunk.split("\n").filter((line) => !line.startsWith("|t:|"));
      chunks[player].push(lines);
      const send = (choice: string): void => {
        if (player === "p1") sent.push({ after_frame: chunks.p1.length, choice });
        stream.write(choice);
      };
      for (const line of lines) {
        if (line.startsWith("|request|")) {
          const raw = line.slice("|request|".length);
          seq += 1;
          if ((JSON.parse(raw) as { wait?: boolean }).wait === true) {
            if (player === "p1" && !outOfTurnSent) {
              outOfTurnSent = true;
              send(OUT_OF_TURN_CHOICE);
            }
            continue;
          }
          if (battleStream.battle?.ended === true) continue;
          send(choose(player, raw, seq));
        } else if (line.startsWith("|error|[Invalid choice]") && player === "p1" && stage === "invalid") {
          stage = "rule";
          send(afterInvalid);
        } else if (line.startsWith("|win|") || line === "|tie") {
          return;
        }
      }
    }
  };

  const done = Promise.all([play("p1"), play("p2")]);
  streams.omniscient.write(
    [
      `>start ${JSON.stringify({ formatid: format.manifest.format_id, seed: spec.seed })}`,
      ...(["p1", "p2"] as const).map(
        (p) => `>player ${p} ${JSON.stringify({ name: p, team: engine.sim.Teams.pack(spec.teams[p]) })}`,
      ),
    ].join("\n"),
  );
  await done;
  streams.omniscient.writeEnd();

  const frames = serverFrames(chunks);
  const battle: BattleFrames = {
    provenance:
      "synthetic: the pinned engine's own p1 stream of a seeded battle between synthetic CC0 teams (the choice corpus's maybe-trapped case), framed as the pinned server frames it; see README.md",
    manifest: format.manifest,
    seed: spec.seed,
    room: BATTLE_ROOM,
    player: "p1",
    sent,
    frames,
  };
  return { "battle.json": battle, "server-layer.json": serverLayerFrames(battle) };
}

const requestOf = (lines: string[]): string | undefined =>
  lines.find((line) => line.startsWith("|request|"))?.slice("|request|".length);

/**
 * Frames p1's chunks as the pinned server sends them to p1's connection.
 * `server/room-battle.ts` (`receive`, :769-819) sends each `update` as one
 * `>roomid` frame (`room.add` + `room.update`), and each `sideupdate` line
 * on its own through `sendRoom`. Before sending a `|request|` it increments
 * the battle-wide `rqid` and appends it to the request JSON (:803-815).
 *
 * The counter is shared by both players. The sim emits one request per side
 * per round, p1 then p2 (`Battle.sendUpdates`), and an `update` request after
 * an `[Unavailable choice]` to one side only, so p1's rqids follow from both
 * players' request lists as long as p2 never gets an `update` request.
 */
function serverFrames(chunks: Record<Player, string[][]>): string[] {
  const updates = (player: Player): boolean[] =>
    chunks[player].flatMap((lines) => {
      const raw = requestOf(lines);
      return raw === undefined ? [] : [(JSON.parse(raw) as { update?: boolean }).update === true];
    });
  const p1Rounds = updates("p1").filter((update) => !update).length;
  if (updates("p2").includes(true) || updates("p2").length !== p1Rounds) {
    throw new Error("p2's requests do not pair with p1's rounds: p1's rqids would be ambiguous");
  }
  let rqid = 0;
  // p2's request of the current round, sent right after p1's and not yet counted.
  let p2Owed = false;
  return chunks.p1.flatMap((lines) => {
    const raw = requestOf(lines);
    if (raw !== undefined) {
      const request = JSON.parse(raw) as { update?: boolean; rqid?: number };
      if (p2Owed) rqid += 1;
      rqid += 1;
      p2Owed = request.update !== true;
      request.rqid = rqid;
      return [`>${BATTLE_ROOM}\n|request|${JSON.stringify(request)}`];
    }
    const body = lines.filter((line) => line !== "");
    return body.length === 0 ? [] : [`>${BATTLE_ROOM}\n${body.join("\n")}`];
  });
}

/**
 * Frames the server layer sends on its own, written from the pinned server
 * source (cited per frame). The requests in them are copied from the
 * engine-derived battle, so a re-send carries a real request.
 */
function serverLayerFrames(battle: BattleFrames): ServerLayerFrames {
  const room = `>${battle.room}\n`;
  // The first move request p1 answered (not an update): what a
  // rejoin re-sends, with the choice p1 had sent for it.
  const requests = battle.frames
    .map((frame, index) => ({ index, raw: requestOf(frame.split("\n")) }))
    .filter((r): r is { index: number; raw: string } => r.raw !== undefined);
  const answered = requests.find((r) => {
    const parsed = JSON.parse(r.raw) as { active?: unknown; update?: boolean };
    return parsed.active !== undefined && parsed.update !== true && battle.sent.some((s) => s.after_frame === r.index + 1);
  });
  if (answered === undefined) throw new Error("no answered move request to re-send");
  const choice = (battle.sent.find((s) => s.after_frame === answered.index + 1) as { choice: string }).choice;
  return {
    provenance:
      "synthetic: written by Areneon from the pinned server source cited per frame; requests copied from battle.json; see README.md",
    room: battle.room,
    frames: {
      global_updateuser: {
        source: "PROTOCOL.md (`>ROOMID` omitted for the global room); server/users.ts:965-975 (`getUpdateuserText`; settings abbreviated)",
        frame: '|updateuser| Areneon-bot-0000|0|1|{"blockChallenges":false}',
      },
      other_room: {
        source: "PROTOCOL.md (`>ROOMID` header, `|j|` join line)",
        frame: ">lobby\n|j| human",
      },
      request_null: {
        source: "server/room-battle.ts:1021 (`onLeave`: `|request|null`)",
        frame: `${room}|request|null`,
      },
      request_empty: {
        source: "the empty-body form of `|request|`, treated like `|request|null`",
        frame: `${room}|request|`,
      },
      resend_with_sentchoice: {
        source: "server/room-battle.ts:954-961 (`onConnect`: current request, then `|sentchoice|`)",
        frame: `${room}|request|${answered.raw}\n|sentchoice|${choice}`,
      },
      resend_without_sentchoice: {
        source: "server/room-battle.ts:954-961 (`onConnect` before any choice: the request alone)",
        frame: `${room}|request|${answered.raw}`,
      },
      nothing_to_choose: {
        source: "server/room-battle.ts:628 (`choose` with no open request)",
        frame: `${room}|error|[Invalid choice] There's nothing to choose`,
      },
      too_late: {
        source: "server/room-battle.ts:634 (`choose` with a stale rqid, or every player waiting)",
        frame: `${room}|error|[Invalid choice] Sorry, too late to make a different move; the next turn has already started`,
      },
    },
  };
}
