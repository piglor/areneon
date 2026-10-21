"use strict";
// The replay interpreter (ADR-017 sections 4 and 4b). A classic script, not a module: Python inlines
// its bytes into the page's one hashed script, and the tests run the same bytes in a vm context.
// It is written from this repository's fixtures, the pinned engine's sim/SIM-PROTOCOL.md and the
// Python reducer's grammar notes (ml/areneon_ml/reducer): no line of any Showdown client was used.
//
// It turns ONE player's own channel into a state after every line. The state has the reducer's own
// shape (`own` and `opponent`, snake_case fields), so a test compares the two. Rules it keeps:
//   - Player perspective only. It reads nothing but the channel it is given; a hidden half is never
//     filled in. An opponent HP that is not out of 100 and a `|split|` marker are refused (flagged).
//   - Unknown line kinds and anything the grammar below does not cover are never dropped: the line is
//     kept verbatim and flagged.
//   - It decides nothing about the game. It shows what the log says; ids come from the log's own
//     names, and the optional `dex` (embedded by the page builder for the species in the game) only
//     gives base species and types for the colours.

/**
 * @typedef {{ own: boolean, name: string | null, base_species: string, species: string | null,
 *   temp_species: string | null, level: number | null, gender: string | null, shiny: boolean,
 *   hp_num: number | null, hp_den: number | null, status: string | null, fainted: boolean,
 *   brought: boolean | null, active_slot: string | null, boosts: Record<string, number>,
 *   volatiles: string[], moves_revealed: string[], item: string | null, item_known: boolean,
 *   item_lost: boolean, ability: string | null, ability_suppressed: boolean, mega: boolean,
 *   mega_stone: string | null, stats: Record<string, number> | null, moves_own: string[] | null,
 *   pp: Record<string, number>, can_mega_evo: boolean, phantom?: boolean }} Mon
 * @typedef {{ player: string, username: string | null, teamsize: number | null, roster: Mon[],
 *   slots: Record<string, number | null>, conditions: Record<string, number> }} Side
 * @typedef {{ player: string, format_name: string | null, gametype: string | null, turn: number,
 *   started: boolean, request_kind: string | null, weather: string | null, terrain: string | null,
 *   field_conditions: string[], winner: string | null, own: Side, opponent: Side,
 *   labels: Record<string, string>, omniscient?: boolean }} State
 * @typedef {{ raw: string, kind: string, args: string[], tags: Record<string, string> }} Line
 * @typedef {{ species?: Record<string, { name?: string, base: string, types: string[] }>,
 *   terrains?: string[] }} Dex
 * @typedef {{ index: number, line: string, kind: string, caption: string | null, flagged: string | null,
 *   request: number, state: State }} Frame
 */
var ArReplay = (function () {
  var STAT_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];
  var STAT_NAMES = {
    atk: "Attack", def: "Defense", spa: "Sp. Atk", spd: "Sp. Def", spe: "Speed",
    accuracy: "accuracy", evasion: "evasiveness",
  };
  var MAX_BOOST = 6;
  var SLOTS = { singles: "a", doubles: "ab", triples: "abc" };
  // Lines that change no state: kept, shown, never flagged.
  var NO_STATE = {
    "": 1, gen: 1, rule: 1, rated: 1, upkeep: 1, uhtml: 1, uhtmlchange: 1, cant: 1, "-crit": 1,
    "-supereffective": 1, "-resisted": 1, "-immune": 1, "-fail": 1, "-anim": 1, "-block": 1,
    "-notarget": 1, "-miss": 1, "-hint": 1, "-message": 1, "-center": 1, "-combine": 1,
    "-waiting": 1, "-prepare": 1, "-mustrecharge": 1, "-nothing": 1, "-hitcount": 1,
    "-singlemove": 1, "-singleturn": 1, "-fieldactivate": 1, "-ohko": 1, "-zbroken": 1,
    teampreview: 1,
  };
  var FREE_TEXT = { uhtml: 1, uhtmlchange: 1, "-message": 1, "-hint": 1, rule: 1, tier: 1, error: 1 };
  var TAG = /^\[([a-z]+)\]\s*(.*)$/;
  var REF = /^(p[1-4])([a-d])?: (.+)$/;
  var SIDE = /^(p[1-4]): /;
  var LEVEL = /^L(\d{1,3})$/;
  var HP = /^(\d+)(?:\/(\d+)([gyr])?)?$/;
  var PREFIXED = /^(move|ability|item): (.+)$/;
  var VOLATILE = /^(?:(move|ability|item): )?([A-Za-z][A-Za-z0-9 '.-]*)$/;
  var ERROR_TEXT = /^\[(?:Unavailable|Invalid) choice\] \S[^\n\r]*$/;
  var ABILITY_COPIES = { abilitytrace: 1, moveroleplay: 1 };
  var ILLUSION = "illusion";

  /** @param {string} text */
  function toId(text) {
    return String(text).toLowerCase().replace(/[^a-z0-9]/g, "");
  }

  /** A line the grammar cannot take: the interpreter keeps the line and flags it. */
  class Flag extends Error {}

  /** @param {string} reason @returns {never} */
  function flag(reason) {
    throw new Flag(reason);
  }

  /** A guard: the line is flagged unless `ok`. One place for the branch every check would repeat. */
  /** @param {any} ok @param {string} reason @returns {asserts ok} */
  function need(ok, reason) {
    if (!ok) { throw new Flag(reason); }
  }

  /** @param {string} raw @returns {Line} */
  function parseLine(raw) {
    need(!(raw.charAt(0) !== "|"), "a protocol line starts with '|'");
    var rest = raw.slice(1);
    var cut = rest.indexOf("|");
    var kind = cut < 0 ? rest : rest.slice(0, cut);
    if (kind === "request") { return { raw: raw, kind: kind, args: [cut < 0 ? "" : rest.slice(cut + 1)], tags: {} }; }
    var parts = rest.split("|").slice(1);
    if (FREE_TEXT[kind]) { return { raw: raw, kind: kind, args: parts, tags: {} }; }
    /** @type {string[]} */
    var args = [];
    /** @type {Record<string, string>} */
    var tags = {};
    var seenTag = false;
    parts.forEach(function (part) {
      var m = TAG.exec(part);
      if (m) { tags[String(m[1])] = String(m[2]); seenTag = true; return; }
      if (seenTag) { flag("a positional field after a tag"); }
      args.push(part);
    });
    return { raw: raw, kind: kind, args: args, tags: tags };
  }

  /** @returns {Record<string, number>} */
  function zeroBoosts() {
    /** @type {Record<string, number>} */
    var out = {};
    STAT_KEYS.forEach(function (k) { out[k] = 0; });
    return out;
  }

  /** @param {boolean} own @returns {Mon} */
  function newMon(own) {
    return {
      own: own, name: null, base_species: "", species: null, temp_species: null, level: null,
      gender: null, shiny: false, hp_num: null, hp_den: null, status: null, fainted: false,
      brought: null, active_slot: null, boosts: zeroBoosts(), volatiles: [], moves_revealed: [],
      item: null, item_known: false, item_lost: false, ability: null, ability_suppressed: false,
      mega: false, mega_stone: null, stats: null, moves_own: null, pp: {}, can_mega_evo: false,
    };
  }

  /** @param {string} player @returns {Side} */
  function newSide(player) {
    return { player: player, username: null, teamsize: null, roster: [], slots: {}, conditions: {} };
  }

  /** @template T @param {T} value @returns {T} */
  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  /** @param {string[]} list @param {string} value */
  function addSorted(list, value) {
    if (list.indexOf(value) < 0) { list.push(value); list.sort(); }
  }

  /** @param {string} text @returns {number} */
  function integer(text) {
    if (!/^-?\d+$/.test(text)) { flag("'" + text + "' is not an integer"); }
    return parseInt(text, 10);
  }

  /** The field at `i`, or an empty string: one place for the fallback every handler needs. */
  /** @param {Line} line @param {number} i */
  function arg(line, i) {
    return line.args[i] || "";
  }

  /** @param {Line} line @param {number} low @param {number | null} high */
  function arity(line, low, high) {
    if (line.args.length < low) { flag(line.kind + " needs at least " + low + " fields"); }
    if (high !== null && line.args.length > high) { flag(line.kind + " carries more than " + high + " fields"); }
  }

  /**
   * One interpreter for one player's own channel, or (`omniscient`) for the engine's omniscient log
   * of a recorded evaluation game: both sides' exact values, as an owner's local evaluation view.
   * An omniscient state is p1 as `own` and p2 as `opponent`; every Pokemon is built from the
   * `|poke|` lines, so it needs no request, and `state.omniscient` is true. The engine's omniscient stream
   * carries the secret line already and no `|split|` marker, so a marker is refused here too (fail closed).
   * @param {{ player: string, dex?: Dex, omniscient?: boolean }} options
   */
  function createInterpreter(options) {
    var player = options.player;
    var omni = options.omniscient === true;
    if (player !== "p1" && player !== "p2") { throw new Error("unsupported player"); }
    if (omni && player !== "p1") { throw new Error("an omniscient view is p1's"); }
    var dex = options.dex || {};
    var speciesTable = dex.species || {};
    var terrains = dex.terrains || [];
    /** @type {State} */
    var state = {
      player: player, format_name: null, gametype: null, turn: 0, started: false, request_kind: null,
      weather: null, terrain: null, field_conditions: [], winner: null,
      own: newSide(player), opponent: newSide(player === "p1" ? "p2" : "p1"), labels: {},
    };
    if (omni) { state.omniscient = true; }
    /** @type {Record<string, string>} */
    var shown = {};
    /** @type {Record<string, { index: number, before: Mon | null }>} */
    var disguise = {};
    /** @type {any[]} */
    var requests = [];

    /** The name a log line printed for an id (`raindance` was printed `RainDance`): display only. */
    /** @param {string} text @returns {string} the id */
    function named(text) {
      var p = PREFIXED.exec(text);
      var name = p ? (String(p[2])) : text;
      var id = toId(name);
      if (id !== "" && name !== id) { state.labels[id] = name; }
      return id;
    }

    /** @param {Line} line @param {string} id @returns {Side} */
    function sideOf(id, line) {
      if (id === state.own.player) { return state.own; }
      if (id === state.opponent.player) { return state.opponent; }
      return flag("unsupported side '" + id + "' in " + line.kind);
    }

    /** @param {string} text @param {Line} line @returns {{ side: Side, letter: string | null, name: string }} */
    function ref(text, line) {
      var m = REF.exec(text);
      need(!(!m), "'" + text + "' is not a Pokemon id");
      return { side: sideOf(String(m[1]), line), letter: m[2] || null, name: String(m[3]) };
    }

    /** @param {string} text @param {Line} line @returns {Mon} */
    function mon(text, line) {
      var r = ref(text, line);
      if (r.letter !== null) {
        var index = r.side.slots[r.letter];
        if (index !== undefined && index !== null && shown[r.side.player + r.letter] === r.name) {
          return /** @type {Mon} */ (r.side.roster[index]);
        }
      }
      for (var i = 0; i < r.side.roster.length; i += 1) {
        var held = /** @type {Mon} */ (r.side.roster[i]);
        if (held.name === r.name) { return held; }
      }
      return flag("'" + text + "' names a Pokemon that has not been introduced");
    }

    /** @param {Side} side @param {Mon} m */
    function indexOf(side, m) {
      return side.roster.indexOf(m);
    }

    /** @param {string} text @returns {{ species: string, base: string, hidden: boolean, level: number, gender: string, shiny: boolean }} */
    function details(text) {
      var parts = text.split(", ");
      var name = parts.shift() || "";
      var hidden = name.slice(-2) === "-*";
      var species = toId(hidden ? name.slice(0, -2) : name);
      var level = 100;
      var gender = "";
      var shiny = false;
      parts.forEach(function (part) {
        var lm = LEVEL.exec(part);
        if (lm) { level = parseInt(lm[1] || "100", 10); }
        else if (part === "M" || part === "F") { gender = part; }
        else if (part === "shiny") { shiny = true; }
        else { flag("unknown details field '" + part + "'"); }
      });
      var entry = speciesTable[species];
      return { species: species, base: entry ? entry.base : species, hidden: hidden, level: level, gender: gender, shiny: shiny };
    }

    /** @param {string} text @param {Mon} m @returns {{ num: number, den: number | null, status: string | null }} */
    function parseHp(text, m) {
      var cut = text.indexOf(" ");
      var number = cut < 0 ? text : text.slice(0, cut);
      var status = cut < 0 ? "" : text.slice(cut + 1);
      var match = HP.exec(number);
      need(!(!match), "'" + text + "' is not an HP");
      var num = parseInt(match[1] || "0", 10);
      var den = match[2] ? parseInt(match[2], 10) : null;
      if (den === null) {
        need(!(status !== "fnt" || num !== 0), "'" + text + "' is not an HP");
        return { num: 0, den: null, status: null };
      }
      need(!(!m.own && !omni && den !== 100), "an opponent HP that is not a percentage (refused: a secret half)");
      need(!(num > den || den === 0), "'" + text + "' is not an HP");
      if (match[3] && !omni && (m.own || !({ 50: "gy", 20: "yr" }[num] || "").includes(match[3]))) {
        return flag("'" + text + "' has a colour class an HP cannot carry");
      }
      if (status === "fnt") {
        need(!(num !== 0), "'" + text + "' is not an HP");
        status = "";
      } else if (status) { status = toId(status); }
      return { num: num, den: den, status: status || null };
    }

    /** @param {Mon} m @param {string} text @param {boolean} [withStatus] */
    function setHp(m, text, withStatus) {
      var hp = parseHp(text, m);
      m.hp_num = hp.num;
      m.hp_den = hp.den !== null ? hp.den : (m.hp_den || (m.own ? null : 100));
      if (withStatus !== false) { m.status = hp.status; }
      m.fainted = hp.num === 0;
    }

    /** @param {Mon} m */
    function switchOut(m) {
      m.active_slot = null;
      m.boosts = zeroBoosts();
      m.volatiles = [];
      m.temp_species = null;
      m.ability_suppressed = false;
    }

    /** @param {string} text @returns {{ kind: string, id: string }} */
    function effect(text) {
      var p = PREFIXED.exec(text);
      if (p) { return { kind: String(p[1]), id: named(text) }; }
      return { kind: "bare", id: named(text) };
    }

    /** @param {Mon} m @param {Line} line */
    function revealFrom(m, line) {
      var source = line.tags.from;
      if (source === undefined || line.tags.of !== undefined) { return; }
      var p = PREFIXED.exec(source);
      if (!p) { return; }
      if (p[1] === "item") { m.item = toId(String(p[2])); m.item_known = true; }
      else if (p[1] === "ability") { m.ability = toId(String(p[2])); }
    }

    /** @param {Side} side @param {string} name @param {{ species: string, base: string }} d @returns {number | null} */
    function locate(side, name, d) {
      for (var i = 0; i < side.roster.length; i += 1) {
        if (/** @type {Mon} */ (side.roster[i]).name === name) { return i; }
      }
      if (side.roster.length > 0 && /** @type {Mon} */ (side.roster[0]).own) { return null; }
      for (var j = 0; j < side.roster.length; j += 1) {
        var m = /** @type {Mon} */ (side.roster[j]);
        if (m.name === null && m.base_species === d.base && (m.species === null || m.species === d.species)) { return j; }
      }
      return null;
    }

    /** @param {Side} side @param {string} name @param {{ species: string, base: string }} d @returns {Mon} */
    function bind(side, name, d) {
      var found = locate(side, name, d);
      if (found !== null) {
        var held = /** @type {Mon} */ (side.roster[found]);
        if (held.name === null) { held.name = name; }
        return held;
      }
      need(!(side.roster.length > 0 && /** @type {Mon} */ (side.roster[0]).own), "'" + name + "' is not in the own team");
      need(!(side.teamsize !== null && side.roster.length >= side.teamsize), "'" + name + "' is more Pokemon than the team size");
      var created = newMon(false);
      created.name = name;
      created.base_species = d.base;
      side.roster.push(created);
      return created;
    }

    // ---- requests ------------------------------------------------------------------------------

    /** @param {any} entry @returns {Mon} */
    function ownByIdent(entry) {
      var ident = String(entry.ident);
      var m = SIDE.exec(ident);
      var name = m ? ident.slice(m[0].length) : "";
      for (var i = 0; i < state.own.roster.length; i += 1) {
        if (/** @type {Mon} */ (state.own.roster[i]).name === name) { return /** @type {Mon} */ (state.own.roster[i]); }
      }
      var created = newMon(true);
      created.name = name;
      state.own.roster.push(created);
      return created;
    }

    /** An own request names the real Pokemon in each position: an Illusion user shown as another
     * Pokemon is found here, before its `|replace|` (the own channel shows the disguise). */
    /** @param {any[]} entries */
    function settleOwn(entries) {
      var side = state.own;
      entries.forEach(function (entry, i) {
        var letter = String.fromCharCode(97 + i);
        var key = side.player + letter;
        var d = disguise[key];
        if (!d) { return; }
        var real = ownByIdent(entry);
        var isIllusion = toId(entry.ability || entry.baseAbility || "") === ILLUSION;
        if (!entry.active || indexOf(side, real) === d.index || !isIllusion) { delete disguise[key]; return; }
        settle(side, letter, real);
      });
    }

    /** @param {Line} line */
    function onRequest(line) {
      var body;
      try { body = JSON.parse(arg(line, 0)); } catch { return flag("a request that is not JSON"); }
      var sideData = body && body.side;
      need(!(!sideData || typeof sideData !== "object"), "a request with no side");
      need(!(sideData.id !== state.own.player), "a request for the other player (refused: a secret half)");
      need(!omni, "a request in an omniscient log (a player's own half)");
      var kind = body.wait ? "wait" : body.teamPreview ? "preview" : "forceSwitch" in body ? "forceSwitch" : "active" in body ? "move" : null;
      need(!(kind === null), "a request with no phase");
      var entries = sideData.pokemon || [];
      requests.push(body);
      if (kind !== "preview") { settleOwn(entries); }
      var listed = entries.map(function (/** @type {any} */ entry) {
        var m = ownByIdent(entry);
        var d = details(entry.details);
        m.base_species = d.base;
        m.species = d.species;
        m.level = d.level;
        m.gender = d.gender || null;
        m.shiny = d.shiny;
        setHp(m, entry.condition);
        m.item = entry.item ? toId(entry.item) : null;
        m.item_known = true;
        m.ability = toId(entry.ability || entry.baseAbility || "") || null;
        m.stats = clone(entry.stats || {});
        m.moves_own = (entry.moves || []).map(toId);
        return m;
      });
      if (kind !== "preview") {
        state.own.roster.forEach(function (m) { m.brought = listed.indexOf(m) >= 0; });
      }
      (body.active || []).forEach(function (/** @type {any} */ active, /** @type {number} */ index) {
        var m = listed[index];
        if (!m) { flag("a request with more active entries than Pokemon"); return; }
        (active.moves || []).forEach(function (/** @type {any} */ move) {
          if (typeof move.pp === "number") { m.pp[toId(move.id)] = move.pp; }
          if (typeof move.move === "string") { named(move.move); }
        });
        m.can_mega_evo = !!(active.canMegaEvo || active.canMegaEvoX || active.canMegaEvoY);
      });
      state.request_kind = kind;
    }

    // ---- Illusion --------------------------------------------------------------------------------

    /** Carry what `shown` gained since `before` over to `real`. */
    /** @param {Mon} sh @param {Mon} before @param {Mon} real @param {string | null} letter */
    function carry(sh, before, real, letter) {
      real.hp_num = sh.hp_num; real.hp_den = sh.hp_den;
      real.status = sh.status; real.fainted = sh.fainted;
      if (sh.item !== before.item || sh.item_known !== before.item_known || sh.item_lost !== before.item_lost) {
        real.item = sh.item; real.item_known = sh.item_known; real.item_lost = sh.item_lost;
      }
      if (sh.ability !== before.ability) { real.ability = sh.ability; }
      if (sh.mega !== before.mega) { real.mega = sh.mega; real.mega_stone = sh.mega_stone; }
      real.boosts = clone(sh.boosts); real.volatiles = sh.volatiles.slice();
      real.temp_species = sh.temp_species; real.ability_suppressed = sh.ability_suppressed;
      sh.moves_revealed.forEach(function (move) {
        if (real.moves_revealed.indexOf(move) < 0 && before.moves_revealed.indexOf(move) < 0) { real.moves_revealed.push(move); }
      });
      real.active_slot = letter; real.brought = true;
    }

    /** The Pokemon at `letter` is `real`: carry over what was seen there and put the one it was shown as back. */
    /** @param {Side} side @param {string} letter @param {Mon} real */
    function settle(side, letter, real) {
      var key = side.player + letter;
      var d = disguise[key];
      delete disguise[key];
      need(!(!d || d.before === null), "the disguise was not on the roster");
      var sh = /** @type {Mon} */ (side.roster[d.index]);
      need(!(real.active_slot !== null && real.active_slot !== letter), "the replaced Pokemon is already in play");
      carry(sh, d.before, real, letter);
      var restored = side.roster[d.index] = d.before;
      if (restored.active_slot !== null) { switchOut(restored); }
      side.slots[letter] = indexOf(side, real);
    }

    /** @param {Line} line */
    function onReplace(line) {
      arity(line, 2, 3);
      var r = ref(arg(line, 0), line);
      var side = r.side;
      var letter = r.letter;
      var slotIndex = letter === null ? undefined : side.slots[letter];
      need(!(letter === null || slotIndex === undefined || slotIndex === null), "replace names a position nobody holds");
      var d = details(arg(line, 1));
      var key = side.player + letter;
      var real;
      if (disguise[key]) {
        var shownIndex = /** @type {{ index: number }} */ (disguise[key]).index;
        var named = side.roster.filter(function (m, i) { return i !== shownIndex && !m.phantom && m.name === r.name; });
        var unnamed = side.roster.filter(function (m, i) {
          return i !== shownIndex && !m.phantom && !m.own && m.name === null && m.base_species === d.base &&
            (m.species === null || m.species === d.species);
        });
        var found = named.length > 0 ? named : unnamed;
        need(!(found.length !== 1), "a replace that names no single Pokemon of the roster");
        real = /** @type {Mon} */ (found[0]);
        settle(side, letter, real);
      } else {
        real = /** @type {Mon} */ (side.roster[slotIndex]);
        need(!(!(real.own && real.name === r.name)), "a replace with no disguise on record");
      }
      need(!(real.base_species !== d.base), "the Pokemon shown in a replace is not the roster's");
      real.name = r.name; real.brought = true;
      real.species = d.species; real.level = d.level;
      real.gender = d.gender || null; real.shiny = d.shiny;
      real.ability = ILLUSION;
      shown[key] = r.name;
      if (line.args[2] !== undefined) { setHp(real, line.args[2]); }
    }

    // ---- handlers --------------------------------------------------------------------------------

    /** @param {Line} line */
    function onSwitch(line) {
      arity(line, 3, 3);
      var r = ref(arg(line, 0), line);
      var side = r.side;
      var letter = r.letter;
      need(!(letter === null || !(letter in side.slots)), "a switch needs an active position");
      var d = details(arg(line, 1));
      var located = locate(side, r.name, d);
      var before = located === null ? null : clone(/** @type {Mon} */ (side.roster[located]));
      var m = bind(side, r.name, d);
      var key = side.player + letter;
      if (m.fainted && !m.own) {
        // A fainted Pokemon cannot come back: the one that fainted was an Illusion user shown as this one.
        var dead = clone(m);
        dead.phantom = true;
        side.roster.push(dead);
        var slot = indexOf(side, m);
        m = side.roster[slot] = newMon(false);
        m.name = dead.name; m.base_species = dead.base_species; m.species = dead.species;
        before = clone(m);
      }
      var twin = false;
      if (m.active_slot !== null && m.active_slot !== letter) {
        // Shown in two positions: one of the two is an Illusion user, and the channel does not say
        // which. This one gets an entry of its own and the view says it is not reconciled.
        twin = true;
        var extra = m.own ? clone(m) : newMon(false);
        if (!m.own) { extra.name = m.name; extra.base_species = m.base_species; } else { extra.moves_revealed = []; }
        extra.phantom = true;
        side.roster.push(extra);
        m = extra;
        before = clone(m);
      }
      disguise[key] = { index: indexOf(side, m), before: before };
      var previous = side.slots[letter];
      if (previous !== undefined && previous !== null) { switchOut(/** @type {Mon} */ (side.roster[previous])); }
      side.slots[letter] = indexOf(side, m);
      shown[key] = r.name;
      switchOut(m);
      m.active_slot = letter; m.brought = true;
      m.species = d.species; m.level = d.level;
      m.gender = d.gender || null; m.shiny = d.shiny;
      if (!m.own && m.base_species === "") { m.base_species = d.base; }
      setHp(m, arg(line, 2));
      if (twin) { flag("a Pokemon shown in two positions (an Illusion twin) is not reconciled"); }
    }

    /** @param {Line} line */
    function onSwap(line) {
      arity(line, 2, 2);
      var r = ref(arg(line, 0), line);
      var side = r.side;
      var target = String.fromCharCode(97 + integer(arg(line, 1)));
      var letter = r.letter;
      need(!(letter === null || !(letter in side.slots) || !(target in side.slots)), "swap names an unknown position");
      need(!(side.slots[letter] === null || shown[side.player + letter] !== r.name), "swap of a Pokemon that is not in that position");
      var tmp = side.slots[letter];
      side.slots[letter] = /** @type {number | null} */ (side.slots[target]);
      side.slots[target] = /** @type {number | null} */ (tmp);
      var a = side.player + letter;
      var b = side.player + target;
      var sa = shown[a]; var sb = shown[b];
      delete shown[a]; delete shown[b];
      if (sa !== undefined) { shown[b] = sa; }
      if (sb !== undefined) { shown[a] = sb; }
      var da = disguise[a]; var db = disguise[b];
      delete disguise[a]; delete disguise[b];
      if (da !== undefined) { disguise[b] = da; }
      if (db !== undefined) { disguise[a] = db; }
      [letter, target].forEach(function (s) {
        var idx = side.slots[s];
        if (idx !== undefined && idx !== null) { /** @type {Mon} */ (side.roster[idx]).active_slot = s; }
      });
    }

    /** @param {Line} line */
    function onFaint(line) {
      arity(line, 1, 1);
      var r = ref(arg(line, 0), line);
      var m = mon(arg(line, 0), line);
      m.fainted = true; m.hp_num = 0;
      if (r.letter !== null && r.side.slots[r.letter] === indexOf(r.side, m)) {
        r.side.slots[r.letter] = null;
        delete disguise[r.side.player + r.letter];
      }
      switchOut(m);
      m.status = null;
    }

    /** @param {Line} line @param {boolean} withStatus */
    function onHp(line, withStatus) {
      arity(line, 2, 2);
      var m = mon(arg(line, 0), line);
      setHp(m, arg(line, 1), withStatus);
      if (withStatus) { revealFrom(m, line); }
    }

    /** @param {Line} line */
    function onBoost(line) {
      arity(line, 3, 3);
      var m = mon(arg(line, 0), line);
      var stat = arg(line, 1);
      need(!(STAT_KEYS.indexOf(stat) < 0), "unknown stat '" + stat + "'");
      var amount = integer(arg(line, 2));
      var current = /** @type {number} */ (m.boosts[stat]);
      var value = line.kind === "-boost" ? current + amount : line.kind === "-unboost" ? current - amount : amount;
      need(!(Math.abs(value) > MAX_BOOST), "boost " + value + " is outside -6..6");
      m.boosts[stat] = value;
      revealFrom(m, line);
    }

    /** @param {Line} line @returns {string} */
    function sideConditionName(line) {
      arity(line, 2, 2);
      need(!(!SIDE.test(arg(line, 0))), "'" + line.args[0] + "' is not a side");
      return effect(arg(line, 1)).id;
    }

    /** @param {Line} line @returns {Side} */
    function sideOfArg(line) {
      var m = SIDE.exec(arg(line, 0));
      return sideOf((m && m[1]) || "", line);
    }

    /** @type {Record<string, (line: Line) => void>} */
    var handlers = {
      player: function (line) {
        arity(line, 1, 4);
        var side = sideOf(arg(line, 0), line);
        if (line.args.length > 1) { side.username = line.args[1] || null; }
      },
      error: function (line) {
        if (!ERROR_TEXT.test(line.args.join("|"))) { flag("an |error| line that is not a choice refusal"); }
      },
      tier: function (line) { arity(line, 1, 1); state.format_name = line.args[0] || null; },
      gametype: function (line) {
        arity(line, 1, 1);
        var letters = SLOTS[/** @type {keyof typeof SLOTS} */ (line.args[0])];
        need(!(!letters), "unsupported gametype '" + line.args[0] + "'");
        state.gametype = line.args[0] || null;
        [state.own, state.opponent].forEach(function (side) {
          side.slots = {};
          letters.split("").forEach(function (l) { side.slots[l] = null; });
        });
      },
      clearpoke: function () {
        (omni ? [state.own, state.opponent] : [state.opponent]).forEach(function (side) {
          side.roster = [];
          Object.keys(side.slots).forEach(function (k) { side.slots[k] = null; });
          [shown, disguise].forEach(function (table) {
            Object.keys(table).forEach(function (k) { if (k.slice(0, 2) === side.player) { delete table[k]; } });
          });
        });
      },
      poke: function (line) {
        arity(line, 2, 3);
        var flagText = line.args[2];
        need(!(flagText !== undefined && flagText !== "" && flagText !== "item"), "unknown item flag '" + flagText + "'");
        var side = sideOf(arg(line, 0), line);
        var d = details(arg(line, 1));
        if (side === state.own && !omni) { return; }
        var m = newMon(false);
        m.base_species = d.base;
        m.species = d.hidden ? null : d.species;
        m.level = d.level; m.gender = d.gender || null; m.shiny = d.shiny;
        side.roster.push(m);
      },
      teamsize: function (line) { arity(line, 2, 2); sideOf(arg(line, 0), line).teamsize = integer(arg(line, 1)); },
      start: function () { state.started = true; },
      turn: function (line) { arity(line, 1, 1); state.turn = integer(arg(line, 0)); },
      win: function (line) {
        arity(line, 1, 1);
        var label = [["own", state.own], ["opponent", state.opponent]].filter(function (pair) {
          var side = /** @type {Side} */ (pair[1]);
          return side.username !== null && side.username === line.args[0];
        })[0];
        need(!(!label), "'" + line.args[0] + "' is neither player");
        state.winner = omni ? /** @type {Side} */ (label[1]).player : /** @type {string} */ (label[0]);
      },
      tie: function () { state.winner = "tie"; },
      request: onRequest,
      switch: onSwitch,
      drag: onSwitch,
      detailschange: function (line) {
        arity(line, 2, 3);
        var m = mon(arg(line, 0), line);
        var d = details(arg(line, 1));
        m.species = d.species; m.level = d.level; m.gender = d.gender || null; m.shiny = d.shiny;
        if (line.args[2] !== undefined) { setHp(m, line.args[2]); }
      },
      "-formechange": function (line) { arity(line, 2, 3); mon(arg(line, 0), line).temp_species = toId(arg(line, 1)); },
      swap: onSwap,
      replace: onReplace,
      "-activate": function (line) {
        arity(line, 1, 4);
        if (line.args.length > 1 && toId(arg(line, 1)) === "moveskillswap") {
          mon(arg(line, 0), line).ability = null;
          if (line.tags.of !== undefined) { mon(line.tags.of, line).ability = null; }
        }
      },
      move: function (line) {
        arity(line, 2, 3);
        var m = mon(arg(line, 0), line);
        var move = named(arg(line, 1));
        var target = line.args[2];
        if (target && target !== "null") { ref(target, line); }
        if (line.tags.from === undefined && m.moves_revealed.indexOf(move) < 0) { m.moves_revealed.push(move); }
      },
      faint: onFaint,
      "-damage": function (line) { onHp(line, true); },
      "-heal": function (line) { onHp(line, true); },
      "-sethp": function (line) { onHp(line, false); },
      "-status": function (line) {
        arity(line, 2, 2);
        var m = mon(arg(line, 0), line);
        m.status = toId(arg(line, 1));
        revealFrom(m, line);
      },
      "-curestatus": function (line) { arity(line, 2, 2); mon(arg(line, 0), line).status = null; },
      "-cureteam": function (line) {
        arity(line, 1, 1);
        ref(arg(line, 0), line).side.roster.forEach(function (m) { m.status = null; });
      },
      "-boost": onBoost,
      "-unboost": onBoost,
      "-setboost": onBoost,
      "-swapboost": function (line) {
        arity(line, 2, 3);
        var a = mon(arg(line, 0), line);
        var b = mon(arg(line, 1), line);
        var stats = line.args[2] === undefined ? STAT_KEYS : line.args[2].split(",").map(function (s) { return s.trim(); });
        stats.forEach(function (stat) {
          if (STAT_KEYS.indexOf(stat) < 0) { flag("unknown stat '" + stat + "'"); }
          var t = /** @type {number} */ (a.boosts[stat]);
          a.boosts[stat] = /** @type {number} */ (b.boosts[stat]);
          b.boosts[stat] = t;
        });
      },
      "-invertboost": function (line) {
        arity(line, 1, 1);
        var m = mon(arg(line, 0), line);
        STAT_KEYS.forEach(function (k) { m.boosts[k] = -(/** @type {number} */ (m.boosts[k])) || 0; });
      },
      "-clearboost": function (line) { arity(line, 1, 1); mon(arg(line, 0), line).boosts = zeroBoosts(); },
      "-clearallboost": function () {
        [state.own, state.opponent].forEach(function (side) { side.roster.forEach(function (m) { m.boosts = zeroBoosts(); }); });
      },
      "-clearpositiveboost": function (line) {
        arity(line, 3, 3);
        var m = mon(arg(line, 0), line);
        mon(arg(line, 1), line);
        STAT_KEYS.forEach(function (k) { m.boosts[k] = Math.min(/** @type {number} */ (m.boosts[k]), 0); });
      },
      "-clearnegativeboost": function (line) {
        arity(line, 1, 1);
        var m = mon(arg(line, 0), line);
        STAT_KEYS.forEach(function (k) { m.boosts[k] = Math.max(/** @type {number} */ (m.boosts[k]), 0); });
      },
      "-copyboost": function (line) {
        arity(line, 2, 2);
        var copier = mon(arg(line, 0), line);
        copier.boosts = clone(mon(arg(line, 1), line).boosts);
      },
      "-weather": function (line) { arity(line, 1, 1); state.weather = line.args[0] === "none" ? null : named(arg(line, 0)); },
      "-fieldstart": function (line) {
        arity(line, 1, 1);
        var id = effect(arg(line, 0)).id;
        if (terrains.indexOf(id) >= 0) { state.terrain = id; } else { addSorted(state.field_conditions, id); }
      },
      "-fieldend": function (line) {
        arity(line, 1, 1);
        var id = effect(arg(line, 0)).id;
        if (terrains.indexOf(id) >= 0) { if (state.terrain === id) { state.terrain = null; } }
        else { state.field_conditions = state.field_conditions.filter(function (c) { return c !== id; }); }
      },
      "-sidestart": function (line) {
        var id = sideConditionName(line);
        var side = sideOfArg(line);
        side.conditions[id] = (side.conditions[id] || 0) + 1;
      },
      "-sideend": function (line) { var id = sideConditionName(line); delete sideOfArg(line).conditions[id]; },
      "-swapsideconditions": function () {
        var t = state.own.conditions; state.own.conditions = state.opponent.conditions; state.opponent.conditions = t;
      },
      "-start": function (line) {
        arity(line, 2, 3);
        var m = mon(arg(line, 0), line);
        var text = arg(line, 1);
        var v = VOLATILE.exec(text);
        need(!(!v || text.length > 64), "'" + text + "' is not a volatile id");
        var id = named(String(v[2]));
        if (v[1] === "ability") { m.ability = id; }
        addSorted(m.volatiles, id);
      },
      "-end": function (line) {
        arity(line, 2, 3);
        var m = mon(arg(line, 0), line);
        var text = arg(line, 1);
        var v = VOLATILE.exec(text);
        need(!(!v || text.length > 64), "'" + text + "' is not a volatile id");
        var id = toId(String(v[2]));
        m.volatiles = m.volatiles.filter(function (x) { return x !== id; });
      },
      "-item": function (line) {
        arity(line, 2, 2);
        var m = mon(arg(line, 0), line);
        m.item = named(arg(line, 1)); m.item_known = true; m.item_lost = false;
      },
      "-enditem": function (line) {
        arity(line, 2, 2);
        var m = mon(arg(line, 0), line);
        m.item = null; m.item_known = true; m.item_lost = true;
      },
      "-ability": function (line) {
        arity(line, 2, 3);
        var m = mon(arg(line, 0), line);
        m.ability = named(arg(line, 1)); m.ability_suppressed = false;
        var from = line.tags.from;
        var holder = line.tags.of;
        if (from !== undefined && holder !== undefined && ABILITY_COPIES[/** @type {keyof typeof ABILITY_COPIES} */ (toId(from))]) {
          mon(holder, line).ability = m.ability;
        }
      },
      "-endability": function (line) { arity(line, 1, 1); mon(arg(line, 0), line).ability_suppressed = true; },
      "-transform": function (line) {
        arity(line, 2, 2);
        var m = mon(arg(line, 0), line);
        var t = mon(arg(line, 1), line);
        m.temp_species = t.temp_species || t.species;
        addSorted(m.volatiles, "transform");
      },
      "-mega": function (line) {
        arity(line, 3, 3);
        var m = mon(arg(line, 0), line);
        m.mega = true; m.mega_stone = named(arg(line, 2)); m.can_mega_evo = false;
      },
    };

    // ---- captions --------------------------------------------------------------------------------

    /** @param {string} text @returns {string} the Pokemon's name without its position */
    function who(text) {
      var m = REF.exec(text);
      return m ? (String(m[3])) : text;
    }

    /** @param {string} text @returns {string} */
    function words(text) {
      var p = PREFIXED.exec(text);
      return p ? (String(p[2])) : text;
    }

    /** @param {Line} line @returns {string | null} */
    function caption(line) {
      var a = line.args;
      var A = function (/** @type {number} */ i) { return a[i] || ""; };
      switch (line.kind) {
        case "turn": return "Turn " + A(0);
        case "switch": return who(A(0)) + " (" + A(1).split(",")[0] + ") was sent out";
        case "drag": return who(A(0)) + " was dragged out";
        case "move": return who(A(0)) + " used " + A(1) + (A(2) && A(2) !== "null" ? " on " + who(A(2)) : "");
        case "-damage": return who(A(0)) + " lost HP" + (line.tags.from ? " (" + words(line.tags.from) + ")" : "");
        case "-heal": return who(A(0)) + " regained HP" + (line.tags.from ? " (" + words(line.tags.from) + ")" : "");
        case "-sethp": return who(A(0)) + "'s HP was set";
        case "faint": return who(A(0)) + " fainted";
        case "-status": return who(A(0)) + " was afflicted: " + A(1);
        case "-curestatus": return who(A(0)) + " was cured of " + A(1);
        case "-cureteam": return "The team was cured";
        case "-boost": return who(A(0)) + "'s " + (STAT_NAMES[/** @type {keyof typeof STAT_NAMES} */ (A(1))] || A(1)) + " rose" + (A(2) === "1" ? "" : " by " + A(2));
        case "-unboost": return who(A(0)) + "'s " + (STAT_NAMES[/** @type {keyof typeof STAT_NAMES} */ (A(1))] || A(1)) + " fell" + (A(2) === "1" ? "" : " by " + A(2));
        case "-setboost": return who(A(0)) + "'s " + (STAT_NAMES[/** @type {keyof typeof STAT_NAMES} */ (A(1))] || A(1)) + " was set to " + A(2);
        case "-clearallboost": return "All stat changes were reset";
        case "-clearboost": return who(A(0)) + "'s stat changes were reset";
        case "-supereffective": return "It's super effective!";
        case "-resisted": return "It's not very effective";
        case "-immune": return who(A(0)) + " is unaffected";
        case "-crit": return "A critical hit!";
        case "-miss": return who(A(0)) + "'s attack missed";
        case "-fail": return "But it failed";
        case "cant": return who(A(0)) + " can't move" + (A(1) ? " (" + words(A(1)) + ")" : "");
        case "-weather": return A(0) === "none" ? "The weather cleared" : "Weather: " + A(0);
        case "-fieldstart": return words(A(0)) + " began";
        case "-fieldend": return words(A(0)) + " ended";
        case "-sidestart": return words(A(1)) + " began on " + words(A(0));
        case "-sideend": return words(A(1)) + " ended on " + words(A(0));
        case "-start": return who(A(0)) + ": " + words(A(1)) + " began";
        case "-end": return who(A(0)) + ": " + words(A(1)) + " ended";
        case "-item": return who(A(0)) + " has " + A(1);
        case "-enditem": return who(A(0)) + " lost its " + A(1);
        case "-ability": return who(A(0)) + " has " + A(1);
        case "-mega": return who(A(0)) + " Mega Evolved";
        case "detailschange": return who(A(0)) + " changed forme: " + A(1).split(",")[0];
        case "-formechange": return who(A(0)) + " changed forme: " + A(1);
        case "-transform": return who(A(0)) + " transformed into " + who(A(1));
        case "replace": return who(A(0)) + " was really " + A(1).split(",")[0] + " (the disguise ended)";
        case "swap": return who(A(0)) + " changed places";
        case "win": return A(0) + " won the battle";
        case "tie": return "The battle ended in a tie";
        case "error": return "Choice refused: " + a.join("|");
        case "-message": return a.join("|");
        case "-hint": return a.join("|");
        case "-activate": return who(A(0)) + ": " + words(A(1));
        case "-prepare": return who(A(0)) + " is preparing " + A(1);
        case "-block": return who(A(0)) + " was protected";
        case "-hitcount": return "Hit " + A(1) + " time(s)";
        case "-notarget": return "But there was no target";
        case "-mustrecharge": return who(A(0)) + " must recharge";
        case "-singleturn": return who(A(0)) + ": " + words(A(1));
        case "-singlemove": return who(A(0)) + ": " + words(A(1));
        case "-waiting": return who(A(0)) + " is waiting for " + who(A(1));
        case "-fieldactivate": return words(A(0)) + " took effect";
        case "-ohko": return "It's a one-hit KO!";
        case "-copyboost": return who(A(0)) + " copied " + who(A(1)) + "'s stat changes";
        case "-swapboost": return who(A(0)) + " swapped stat changes with " + who(A(1));
        case "-invertboost": return who(A(0)) + "'s stat changes were inverted";
        case "-clearpositiveboost": return who(A(0)) + "'s stat boosts were removed";
        case "-clearnegativeboost": return who(A(0)) + "'s stat drops were removed";
        case "-swapsideconditions": return "The side conditions were swapped";
        case "-endability": return who(A(0)) + "'s ability was suppressed";
        default: return null;
      }
    }

    /**
     * Feed one raw protocol line. Returns what to show for it.
     * @param {string} raw
     * @returns {{ kind: string, caption: string | null, flagged: string | null }}
     */
    function feed(raw) {
      var kind = "";
      try {
        if (raw.slice(0, 7) === "|split|") { kind = "split"; return flag("a |split| marker holds the opponent's secret half (refused)"); }
        var line = parseLine(raw);
        kind = line.kind;
        var text = caption(line);
        if (NO_STATE[line.kind] === 1 && !handlers[line.kind]) { return { kind: kind, caption: text, flagged: null }; }
        var handler = handlers[line.kind];
        need(!(!handler), "no rule for line type '" + line.kind + "'");
        handler(line);
        return { kind: kind, caption: text, flagged: null };
      } catch (error) {
        if (error instanceof Flag) { return { kind: kind, caption: null, flagged: error.message }; }
        throw error;
      }
    }

    return {
      feed: feed,
      /** @returns {State} a copy of the state now */
      snapshot: function () { return clone(state); },
      /** @returns {number} how many `|request|` lines were taken */
      requestCount: function () { return requests.length; },
      /** @returns {any[]} the requests taken, parsed, in order */
      requests: function () { return requests; },
    };
  }

  /**
   * Interpret a whole channel: one frame per line, the state after it. Frame 0 is the state before
   * any line. A state equal to the one before it is shared, not copied.
   * @param {string} player
   * @param {string[]} lines
   * @param {Dex} [dex]
   * @param {boolean} [omniscient] the log is the engine's omniscient one (player `p1`, both sides exact)
   * @returns {{ frames: Frame[], flags: { index: number, line: string, reason: string }[], requests: any[] }}
   */
  function interpretChannel(player, lines, dex, omniscient) {
    var interp = createInterpreter({ player: player, dex: dex || {}, omniscient: omniscient === true });
    var first = interp.snapshot();
    /** @type {Frame[]} */
    var frames = [{ index: -1, line: "", kind: "", caption: null, flagged: null, request: 0, state: first }];
    var lastText = JSON.stringify(first);
    var last = first;
    /** @type {{ index: number, line: string, reason: string }[]} */
    var flags = [];
    lines.forEach(function (raw, index) {
      var result = interp.feed(raw);
      var snap = interp.snapshot();
      var text = JSON.stringify(snap);
      if (text !== lastText) { last = snap; lastText = text; }
      if (result.flagged !== null) { flags.push({ index: index, line: raw, reason: result.flagged }); }
      frames.push({ index: index, line: raw, kind: result.kind, caption: result.caption, flagged: result.flagged, request: interp.requestCount(), state: last });
    });
    return { frames: frames, flags: flags, requests: interp.requests() };
  }

  return { toId: toId, createInterpreter: createInterpreter, interpretChannel: interpretChannel, parseLine: parseLine };
})();

if (typeof module !== "undefined") { module.exports = ArReplay; } // the tests load this file; the page does not
