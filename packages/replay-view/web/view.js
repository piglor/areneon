"use strict";
// The replay view (ADR-017 section 4b): an SVG board, controls, caption bar, decision panel and
// log panel drawn from the interpreter's states (interpreter.js, loaded first in the same script).
// Page security (ADR-017 section 4c): every string reaches the page through `textContent`, and every
// node is made with `createElement` / `createElementNS`. No markup string is ever parsed (the page
// uses none of the DOM's HTML-writing properties or methods), there is no inline handler, and
// `make()` refuses to set a `style` or `on*` attribute. Animation only eases between interpreter states; the end of each easing is
// exactly the state's own value, so stepping, jumping and printing show the same positions.
//
// Data in: one JSON block (`replay-data`) read by `boot`. Nothing is fetched.

/**
 * @typedef {{ choice: string, request_seq: number, policy: string }} Decision
 * @typedef {{ player: string, label: string, lines: string[], team: any[], decisions: Decision[],
 *   result: string }} ViewData
 * @typedef {{ title: string, views: ViewData[], dex: any, links: { label: string, href: string }[],
 *   notes: string[] }} ReplayData
 */
var ArView = (function () {
  var SVG = ["http", ":", "/", "/", "www.w3.org", "/2000/svg"].join(""); // the SVG namespace name: a name, never fetched, in pieces so a scan of a page for a URL stays exact
  var W = 640;
  var HP_W = 110;
  var FORBIDDEN_ATTR = /^(style|on.*)$/i;

  /**
   * Make an element; text goes in as a text node (textContent), never as markup.
   * @param {Document} doc
   * @param {string} tag
   * @param {Record<string, string | number> | null} attrs
   * @param {(Node | string | null)[]} [kids]
   * @param {boolean} [svg]
   * @returns {any}
   */
  function make(doc, tag, attrs, kids, svg) {
    var node = svg ? doc.createElementNS(SVG, tag) : doc.createElement(tag);
    Object.keys(attrs || {}).forEach(function (name) {
      if (FORBIDDEN_ATTR.test(name)) { throw new Error("attribute not allowed: " + name); }
      node.setAttribute(name, String(/** @type {any} */ (attrs)[name]));
    });
    (kids || []).forEach(function (kid) {
      if (kid === null) { return; }
      node.appendChild(typeof kid === "string" ? doc.createTextNode(kid) : kid);
    });
    return node;
  }

  /** @param {string} text @param {number} max */
  function clip(text, max) {
    return text.length > max ? text.slice(0, max - 1) + "…" : text;
  }

  /** @param {string} id @param {Record<string, string>} labels */
  function pretty(id, labels) {
    return (labels && labels[id]) || id;
  }

  // ---- alignment of two channels --------------------------------------------------------------------

  var PRIVATE = { request: 1, error: 1, uhtml: 1, uhtmlchange: 1, "": 1 };

  /** @param {string} raw */
  function kindOf(raw) {
    var rest = raw.slice(1);
    var cut = rest.indexOf("|");
    return cut < 0 ? rest : rest.slice(0, cut);
  }

  /** @param {string} raw */
  function keyOf(raw) {
    var parts = raw.slice(1).split("|");
    return parts[0] + "|" + (parts[1] || "");
  }

  /**
   * For each frame of the master channel, the frame of the other channel that shows the same moment.
   * Lines both players see are matched by kind and subject; a line only one player gets (a request,
   * a refusal) pairs with the same kind in the other channel when there is one.
   * @param {string[]} master
   * @param {string[]} other
   * @returns {number[]} frame i of the master -> frame of the other (lines consumed)
   */
  function align(master, other) {
    var map = [0];
    var j = 0;
    master.forEach(function (raw) {
      var kind = kindOf(raw);
      if (PRIVATE[/** @type {keyof typeof PRIVATE} */ (kind)]) {
        if (j < other.length && kindOf(other[j] || "") === kind) { j += 1; }
      } else {
        while (j < other.length && PRIVATE[/** @type {keyof typeof PRIVATE} */ (kindOf(other[j] || ""))]) { j += 1; }
        if (j < other.length && keyOf(other[j] || "") === keyOf(raw)) { j += 1; }
        else {
          var found = -1;
          for (var k = j; k < Math.min(other.length, j + 20); k += 1) {
            if (keyOf(other[k] || "") === keyOf(raw)) { found = k; break; }
          }
          if (found >= 0) { j = found + 1; }
        }
      }
      map.push(j);
    });
    map[map.length - 1] = other.length; // the last line of the battle: the other view is at its end too
    return map;
  }

  // ---- the choice string of a decision -------------------------------------------------------------

  /**
   * Split a recorded choice into one entry per slot: `move 2 1` / `switch 3` / `pass`, or the team
   * order of a Team Preview (`team 3145`). Anything else is kept whole and highlights nothing.
   * @param {string} choice
   * @returns {{ kind: string, index: number, target: string, extra: string, raw: string }[]}
   */
  function parseChoice(choice) {
    return choice.split(", ").map(function (part) {
      var words = part.split(" ");
      var kind = words[0] || "";
      if ((kind === "move" || kind === "switch") && /^\d+$/.test(words[1] || "")) {
        return { kind: kind, index: parseInt(words[1] || "0", 10), target: /^-?\d+$/.test(words[2] || "") ? (words[2] || "") : "", extra: words.slice(/^-?\d+$/.test(words[2] || "") ? 3 : 2).join(" "), raw: part };
      }
      if (kind === "team" && /^\d+$/.test(words[1] || "")) { return { kind: kind, index: 0, target: words[1] || "", extra: "", raw: part }; }
      return { kind: kind === "pass" ? "pass" : "other", index: 0, target: "", extra: "", raw: part };
    });
  }

  // ---- the board ----------------------------------------------------------------------------------

  /**
   * @param {any} mon
   * @param {any} dex
   * @returns {string[]} type ids ([] when unknown)
   */
  function typesOf(mon, dex) {
    var id = mon.temp_species || mon.species;
    var entry = id && dex.species ? dex.species[id] : null;
    return entry && entry.types ? entry.types.slice(0, 2) : [];
  }

  /** @param {any} mon @param {any} dex @param {Record<string, string>} labels */
  function nameOf(mon, dex, labels) {
    var id = mon.temp_species || mon.species || mon.base_species;
    var entry = id && dex.species ? dex.species[id] : null;
    return (entry && entry.name) || pretty(id || "", labels) || "?";
  }

  /** @param {number | null} num @param {number | null} den */
  function fraction(num, den) {
    return num === null || !den ? 1 : Math.max(0, Math.min(1, num / den));
  }

  /** @param {number} f */
  function hpClass(f) {
    return f > 0.5 ? "hp-g" : f > 0.2 ? "hp-y" : "hp-r";
  }

  /** @param {any} mon */
  function hpText(mon) {
    if (mon.hp_num === null) { return "HP ?"; }
    if (mon.fainted) { return "fainted"; }
    var pct = Math.round(100 * fraction(mon.hp_num, mon.hp_den));
    return mon.own && mon.hp_den ? mon.hp_num + "/" + mon.hp_den + " (" + pct + "%)" : pct + "%";
  }

  /** @param {any} mon @param {Record<string, string>} labels */
  function detailText(mon, labels) {
    var bits = [mon.name || "?"];
    if (mon.species) { bits.push(mon.species); }
    if (mon.item_known) { bits.push("item: " + (mon.item ? pretty(mon.item, labels) : "none")); }
    if (mon.ability) { bits.push("ability: " + pretty(mon.ability, labels)); }
    if (mon.moves_revealed.length) { bits.push("moves: " + mon.moves_revealed.map(function (m) { return pretty(m, labels); }).join(", ")); }
    if (mon.mega) { bits.push("Mega" + (mon.mega_stone ? " (" + pretty(mon.mega_stone, labels) + ")" : "")); }
    return bits.join("; ");
  }

  /**
   * One Pokemon token: two type colours as halves of a disc, or a dashed blank one.
   * @param {Document} doc @param {number} cx @param {number} cy @param {number} r
   * @param {string[]} types @param {boolean} blank
   */
  function tokenShape(doc, cx, cy, r, types, blank) {
    var g = make(doc, "g", { class: "token" }, [], true);
    if (blank) {
      g.appendChild(make(doc, "circle", { cx: cx, cy: cy, r: r, class: "blank" }, [], true));
      g.appendChild(make(doc, "text", { x: cx, y: cy + 5, class: "tk", "text-anchor": "middle" }, ["?"], true));
      return g;
    }
    var a = types[0] || "unknown";
    var b = types[1] || null;
    if (b === null) {
      g.appendChild(make(doc, "circle", { cx: cx, cy: cy, r: r, class: "ty-" + a }, [], true));
    } else {
      g.appendChild(make(doc, "path", { d: "M " + cx + " " + (cy - r) + " A " + r + " " + r + " 0 0 0 " + cx + " " + (cy + r) + " Z", class: "ty-" + a }, [], true));
      g.appendChild(make(doc, "path", { d: "M " + cx + " " + (cy - r) + " A " + r + " " + r + " 0 0 1 " + cx + " " + (cy + r) + " Z", class: "ty-" + b }, [], true));
    }
    g.appendChild(make(doc, "circle", { cx: cx, cy: cy, r: r, class: "ring" }, [], true));
    return g;
  }

  /**
   * The board of one channel: built once, then `update`d from states.
   * @param {Document} doc
   * @param {{ dex: any, animate: () => boolean, raf: (f: (t: number) => void) => void, now: () => number }} env
   * @param {string} perspective e.g. "p1"
   */
  function createBoard(doc, env, perspective) {
    var svg = make(doc, "svg", { viewBox: "0 0 " + W + " 400", class: "board", role: "img", "aria-label": "Battle board as " + perspective + " sees it" }, [], true);
    var title = make(doc, "title", null, ["Battle board, " + perspective + "'s own view"], true);
    svg.appendChild(title);
    svg.appendChild(make(doc, "rect", { x: 0, y: 0, width: W, height: 400, rx: 10, class: "bg" }, [], true));
    svg.appendChild(make(doc, "line", { x1: 20, y1: 200, x2: W - 20, y2: 200, class: "mid" }, [], true));
    var benchTop = make(doc, "g", { class: "bench" }, [], true);
    var benchBottom = make(doc, "g", { class: "bench" }, [], true);
    var fieldRow = make(doc, "g", { class: "fieldrow" }, [], true);
    var sideTop = make(doc, "g", { class: "sidecond" }, [], true);
    var sideBottom = make(doc, "g", { class: "sidecond" }, [], true);
    var who = make(doc, "text", { x: 14, y: 22, class: "who" }, [""], true);
    var whoBottom = make(doc, "text", { x: 14, y: 392, class: "who" }, [""], true);
    var turn = make(doc, "text", { x: W - 14, y: 22, class: "turn", "text-anchor": "end" }, [""], true);
    var slots = { top: [], bottom: [] };
    /** @type {Record<string, any[]>} */
    var slotGroups = slots;
    [["top", 56], ["bottom", 242]].forEach(function (pair) {
      [0, 1].forEach(function (i) {
        var g = make(doc, "g", { class: "slot empty", transform: "translate(" + (60 + i * 280) + " " + pair[1] + ")" }, [], true);
        var holder = make(doc, "g", { class: "tokenholder" }, [], true);
        var name = make(doc, "text", { x: 70, y: 22, class: "nm" }, [""], true);
        var level = make(doc, "text", { x: 70, y: 40, class: "lv" }, [""], true);
        var bar = make(doc, "rect", { x: 70, y: 50, width: HP_W, height: 8, rx: 4, class: "hp hp-g" }, [], true);
        var track = make(doc, "rect", { x: 70, y: 50, width: HP_W, height: 8, rx: 4, class: "hptrack" }, [], true);
        var hp = make(doc, "text", { x: 70, y: 76, class: "hpt" }, [""], true);
        var status = make(doc, "text", { x: 190, y: 59, class: "status", "text-anchor": "start" }, [""], true);
        var chips = make(doc, "text", { x: 70, y: 94, class: "chips" }, [""], true);
        var tip = make(doc, "title", null, [""], true);
        [tip, holder, name, level, track, bar, hp, status, chips].forEach(function (n) { g.appendChild(n); });
        svg.appendChild(g);
        /** @type {any} */ (slotGroups[/** @type {string} */ (pair[0])]).push({ g: g, holder: holder, name: name, level: level, bar: bar, hp: hp, status: status, chips: chips, tip: tip, shownKey: "", hpNow: 1 });
      });
    });
    [benchTop, benchBottom, fieldRow, sideTop, sideBottom, who, whoBottom, turn].forEach(function (n) { svg.appendChild(n); });

    /** @param {any} node */
    function clear(node) {
      while (node.firstChild) { node.removeChild(node.firstChild); }
    }

    /** @param {any} slot @param {number} to @param {boolean} animated */
    function setBar(slot, to, animated) {
      var from = slot.hpNow;
      slot.hpNow = to;
      var token = (slot.token = (slot.token || 0) + 1);
      if (!animated || !env.animate() || from === to) { slot.bar.setAttribute("width", String(Math.round(HP_W * to * 100) / 100)); return; }
      var start = env.now();
      var step = function (/** @type {number} */ t) {
        if (slot.token !== token) { return; }
        var p = Math.min(1, (t - start) / 450);
        var eased = 1 - Math.pow(1 - p, 3);
        var value = p >= 1 ? to : from + (to - from) * eased;
        slot.bar.setAttribute("width", String(Math.round(HP_W * value * 100) / 100));
        if (p < 1) { env.raf(step); }
      };
      env.raf(step);
    }

    /**
     * @param {any} slot @param {any} mon @param {Record<string, string>} labels @param {boolean} animated
     */
    function fillSlot(slot, mon, labels, animated) {
      if (!mon) {
        slot.g.setAttribute("class", "slot empty");
        clear(slot.holder);
        slot.shownKey = "";
        slot.name.textContent = "";
        slot.level.textContent = "";
        slot.hp.textContent = "";
        slot.status.textContent = "";
        slot.chips.textContent = "";
        slot.tip.textContent = "";
        setBar(slot, 0, false);
        return;
      }
      var types = typesOf(mon, env.dex);
      var key = (mon.temp_species || mon.species || mon.base_species) + "|" + types.join(",");
      if (slot.shownKey !== key) {
        clear(slot.holder);
        slot.holder.appendChild(tokenShape(doc, 34, 34, 28, types, false));
        slot.shownKey = key;
        slot.hpNow = fraction(mon.hp_num, mon.hp_den);
      }
      var f = fraction(mon.hp_num, mon.hp_den);
      slot.g.setAttribute("class", "slot" + (mon.fainted ? " fainted" : "") + (mon.own ? " own" : ""));
      slot.bar.setAttribute("class", "hp " + hpClass(f));
      slot.name.textContent = clip(nameOf(mon, env.dex, labels), 16) + (mon.gender ? " " + (mon.gender === "M" ? "♂" : "♀") : "");
      slot.level.textContent = (mon.level !== null ? "L" + mon.level : "") + (mon.mega ? "  Mega" : "") + (mon.temp_species && mon.temp_species !== mon.species ? "  as " + pretty(mon.temp_species, labels) : "");
      slot.hp.textContent = hpText(mon);
      slot.status.textContent = mon.status ? mon.status.toUpperCase() : "";
      slot.status.setAttribute("class", "status st-" + (mon.status || "none"));
      var boosts = Object.keys(mon.boosts).filter(function (k) { return mon.boosts[k] !== 0; }).sort().map(function (k) { return k + (mon.boosts[k] > 0 ? " +" : " ") + mon.boosts[k]; });
      var vols = mon.volatiles.map(function (v) { return pretty(v, labels); });
      slot.chips.textContent = clip(boosts.concat(vols).join("  "), 44);
      slot.tip.textContent = detailText(mon, labels);
      setBar(slot, mon.fainted ? 0 : f, animated);
    }

    /** @param {any} side @param {any} node @param {number} y @param {any} state @param {Record<string, string>} labels @param {boolean} top */
    function fillBench(side, node, y, state, labels, top) {
      clear(node);
      var x = 20;
      var inPlay = {};
      var items = [];
      side.roster.forEach(function (/** @type {any} */ m) {
        if (m.phantom) { return; }
        if (state.started && !m.brought) { return; }
        if (m.active_slot !== null) { return; }
        items.push(m);
      });
      items.forEach(function (/** @type {any} */ m) {
        var shape = tokenShape(doc, x + 12, y, 12, typesOf(m, env.dex), false);
        var cls = "benchmon" + (m.fainted ? " fainted" : "");
        var g = make(doc, "g", { class: cls }, [shape, make(doc, "title", null, [detailText(m, labels) + "; " + hpText(m)], true)], true);
        node.appendChild(g);
        x += 32;
      });
      if (state.started && side.teamsize !== null) {
        var seen = side.roster.filter(function (/** @type {any} */ m) { return m.brought && !m.phantom; }).length;
        for (var k = seen; k < side.teamsize; k += 1) {
          node.appendChild(make(doc, "g", { class: "benchmon" }, [tokenShape(doc, x + 12, y, 12, [], true), make(doc, "title", null, ["not revealed"], true)], true));
          x += 32;
        }
      }
      void inPlay; void top;
    }

    /** @param {any} state @param {boolean} animated */
    function update(state, animated) {
      var labels = state.labels || {};
      var own = state.own;
      var opp = state.opponent;
      who.textContent = (opp.username || opp.player) + " (opponent, as revealed)";
      whoBottom.textContent = (own.username || own.player) + " (this view)";
      turn.textContent = state.turn ? "Turn " + state.turn + (state.winner ? " · finished" : "") : state.started ? "Start" : "Team Preview";
      [["top", opp], ["bottom", own]].forEach(function (pair) {
        var side = /** @type {any} */ (pair[1]);
        var group = /** @type {any[]} */ (slotGroups[/** @type {string} */ (pair[0])]);
        ["a", "b"].forEach(function (letter, i) {
          var index = side.slots[letter];
          fillSlot(group[i], index === undefined || index === null ? null : side.roster[index], labels, animated);
        });
      });
      fillBench(opp, benchTop, 40, state, labels, true);
      fillBench(own, benchBottom, 374, state, labels, false);
      clear(fieldRow);
      var badges = [];
      if (state.weather) { badges.push(["weather", pretty(state.weather, labels)]); }
      if (state.terrain) { badges.push(["terrain", pretty(state.terrain, labels)]); }
      state.field_conditions.forEach(function (/** @type {string} */ c) { badges.push(["field", pretty(c, labels)]); });
      badges.forEach(function (b, i) {
        var text = /** @type {string} */ (b[1]);
        fieldRow.appendChild(make(doc, "g", { class: "badge " + b[0], transform: "translate(" + (20 + i * 150) + " 185)" }, [
          make(doc, "rect", { x: 0, y: 0, width: 140, height: 28, rx: 14 }, [], true),
          make(doc, "text", { x: 70, y: 19, "text-anchor": "middle" }, [clip(text, 18)], true),
        ], true));
      });
      [[sideTop, opp, 156], [sideBottom, own, 216]].forEach(function (triple) {
        var node = /** @type {any} */ (triple[0]);
        var side = /** @type {any} */ (triple[1]);
        clear(node);
        Object.keys(side.conditions).sort().forEach(function (c, i) {
          var count = side.conditions[c];
          node.appendChild(make(doc, "g", { class: "badge side", transform: "translate(" + (20 + i * 128) + " " + triple[2] + ")" }, [
            make(doc, "rect", { x: 0, y: 0, width: 120, height: 24, rx: 12 }, [], true),
            make(doc, "text", { x: 60, y: 16, "text-anchor": "middle" }, [clip(pretty(c, labels) + (count > 1 ? " ×" + count : ""), 16)], true),
          ], true));
        });
      });
    }

    return { svg: svg, update: update };
  }

  // ---- decision panel ---------------------------------------------------------------------------

  /**
   * @param {Document} doc
   * @param {any} request a parsed `|request|` body
   * @param {Decision | null} decision
   * @param {Record<string, string>} labels
   */
  function decisionPanel(doc, request, decision, labels) {
    var box = make(doc, "div", { class: "decision" }, []);
    if (!request) { box.appendChild(make(doc, "p", { class: "muted" }, ["No request yet."])); return box; }
    var picks = decision ? parseChoice(decision.choice) : [];
    var head = decision ? "Chosen: " + decision.choice + "  (policy " + decision.policy + ")" : "No decision is recorded for this request.";
    if (request.wait) { box.appendChild(make(doc, "p", { class: "muted" }, ["Waiting for the other player: nothing to choose."])); return box; }
    box.appendChild(make(doc, "p", { class: "chosen-line" }, [head]));
    var side = (request.side && request.side.pokemon) || [];
    if (request.teamPreview) {
      var order = picks[0] && picks[0].kind === "team" ? picks[0].target.split("").map(Number) : [];
      var list = make(doc, "ol", { class: "opts" }, []);
      side.forEach(function (/** @type {any} */ p, /** @type {number} */ i) {
        var at = order.indexOf(i + 1);
        list.appendChild(make(doc, "li", { class: at >= 0 ? "opt chosen" : "opt" }, [(p.ident || "").replace(/^p[1-4]: /, "") + (at >= 0 ? "  ← slot " + (at + 1) : "")]));
      });
      box.appendChild(make(doc, "p", { class: "muted" }, ["Team Preview: bring " + (request.maxChosenTeamSize || "?") + ". The order is the first slots in the chosen string."]));
      box.appendChild(list);
      return box;
    }
    var moveSlots = request.active || [];
    var forced = request.forceSwitch || [];
    var count = Math.max(moveSlots.length, forced.length);
    for (var slot = 0; slot < count; slot += 1) {
      var pick = picks[slot];
      var mon = side[slot];
      var title = "Slot " + "ab".charAt(slot) + ": " + (mon ? (mon.ident || "").replace(/^p[1-4]: /, "") : "?");
      var wrap = make(doc, "div", { class: "slotopts" }, [make(doc, "h4", null, [title])]);
      var active = moveSlots[slot];
      if (active) {
        var moves = make(doc, "ul", { class: "opts" }, []);
        var movePick = pick;
        (active.moves || []).forEach(function (/** @type {any} */ m, /** @type {number} */ i) {
          var chosen = !!movePick && movePick.kind === "move" && movePick.index === i + 1;
          moves.appendChild(make(doc, "li", { class: "opt" + (chosen ? " chosen" : "") + (m.disabled ? " disabled" : "") }, [
            m.move + (m.pp !== undefined ? " (" + m.pp + "/" + m.maxpp + ")" : "") + (m.disabled ? " disabled" : "") + (chosen && movePick && movePick.target ? "  → target " + movePick.target : "") + (chosen && movePick && movePick.extra ? "  " + movePick.extra : ""),
          ]));
        });
        wrap.appendChild(moves);
      } else if (forced[slot]) {
        wrap.appendChild(make(doc, "p", { class: "muted" }, ["Must switch."]));
      } else {
        wrap.appendChild(make(doc, "p", { class: "muted" }, ["Nothing to choose."]));
      }
      if (pick && (pick.kind === "switch" || pick.kind === "pass" || pick.kind === "other")) {
        var target = pick.kind === "switch" && side[pick.index - 1] ? (side[pick.index - 1].ident || "").replace(/^p[1-4]: /, "") : "";
        wrap.appendChild(make(doc, "p", { class: "chosen-line" }, [pick.kind === "switch" ? "Switch to " + (target || "#" + pick.index) : pick.kind === "pass" ? "Pass" : "Choice: " + pick.raw]));
      }
      box.appendChild(wrap);
    }
    var bench = make(doc, "ul", { class: "opts bench" }, []);
    side.forEach(function (/** @type {any} */ p, /** @type {number} */ i) {
      if (p.active) { return; }
      var chosen = picks.some(function (x) { return x.kind === "switch" && x.index === i + 1; });
      bench.appendChild(make(doc, "li", { class: "opt" + (chosen ? " chosen" : "") }, ["#" + (i + 1) + " " + (p.ident || "").replace(/^p[1-4]: /, "") + " (" + p.condition + ")"]));
    });
    box.appendChild(make(doc, "p", { class: "muted" }, ["On the bench (switch targets):"]));
    box.appendChild(bench);
    void labels;
    return box;
  }

  // ---- the page -----------------------------------------------------------------------------------

  /**
   * Build the whole replay inside `root`.
   * @param {Document} doc
   * @param {any} root
   * @param {ReplayData} data
   * @param {{ raf?: (f: (t: number) => void) => void, now?: () => number, setTimeout?: (f: () => void, ms: number) => any,
   *   clearTimeout?: (h: any) => void, reducedMotion?: () => boolean }} [options]
   */
  function init(doc, root, data, options) {
    var opts = options || {};
    var raf = opts.raf || function (f) { f(0); };
    var now = opts.now || function () { return 0; };
    var later = opts.setTimeout || function () { return 0; };
    var cancel = opts.clearTimeout || function () {};
    var reduced = opts.reducedMotion || function () { return false; };
    var animateOn = true;
    var env = { dex: data.dex || {}, animate: function () { return animateOn && !reduced(); }, raf: raf, now: now };

    var runs = data.views.map(function (v) { return ArReplay.interpretChannel(v.player, v.lines, data.dex); });
    if (!runs[0]) { throw new Error("a replay needs at least one view"); }
    var master = runs[0];
    var maps = runs.map(function (run, i) { return i === 0 ? null : align(data.views[0] ? data.views[0].lines : [], data.views[i] ? data.views[i].lines : []); });
    var last = master.frames.length - 1;
    var position = 0;
    var playing = false;
    var timer = null;
    var speed = 1;

    var views = data.views.map(function (v, i) {
      var board = createBoard(doc, env, v.player);
      var caption = make(doc, "p", { class: "caption" }, [""]);
      var decision = make(doc, "div", { class: "decisionbox" }, []);
      var logList = make(doc, "div", { class: "log" }, []);
      var rawList = make(doc, "div", { class: "log raw" }, []);
      var run = runs[i];
      var frames = run ? run.frames : [];
      var logRows = [];
      var rawRows = [];
      frames.forEach(function (f, k) {
        if (k === 0) { return; }
        var cls = f.flagged ? "line flagged" : "line";
        rawRows.push(make(doc, "div", { class: cls }, [f.line + (f.flagged ? "   [not recognised: " + f.flagged + "]" : "")]));
        if (f.caption || f.flagged) {
          logRows.push({ frame: k, node: make(doc, "div", { class: cls }, [f.flagged ? "Not recognised: " + f.line : f.caption || ""]) });
        }
      });
      logRows.forEach(function (r) { logList.appendChild(r.node); });
      rawRows.forEach(function (n) { rawList.appendChild(n); });
      var teamBox = make(doc, "details", { class: "team" }, [make(doc, "summary", null, ["Own team (" + (v.team || []).length + " sets)"])]);
      (v.team || []).forEach(function (/** @type {any} */ set) {
        teamBox.appendChild(make(doc, "p", { class: "set" }, [
          (set.name && set.name !== set.species ? set.name + " (" + set.species + ")" : set.species) + " @ " + (set.item || "no item") + " · " + set.ability + " · " + (set.moves || []).join(", "),
        ]));
      });
      var flags = make(doc, "p", { class: "flagnote" }, [run && run.flags.length ? run.flags.length + " line(s) were not recognised and are shown flagged." : ""]);
      var panel = make(doc, "section", { class: "viewpanel", "data-player": v.player }, [
        make(doc, "h2", null, [v.label]),
        board.svg, caption, flags,
        make(doc, "details", { class: "decisionwrap", open: "" }, [make(doc, "summary", null, ["Decision at this request"]), decision]),
        teamBox,
        make(doc, "details", { class: "logwrap" }, [make(doc, "summary", null, ["Log"]), logList]),
        make(doc, "details", { class: "logwrap" }, [make(doc, "summary", null, ["Raw log"]), rawList]),
        make(doc, "p", { class: "muted" }, ["Result: " + v.result]),
      ]);
      return { data: v, board: board, caption: caption, decision: decision, panel: panel, frames: frames, run: run, logRows: logRows, rawRows: rawRows, logList: logList, rawList: rawList, lastCaption: "", shownFrame: -1 };
    });

    // controls
    var slider = make(doc, "input", { type: "range", min: 0, max: last, value: 0, "aria-label": "Position in the battle log" }, []);
    var status = make(doc, "span", { class: "muted" }, [""]);
    var turns = [];
    master.frames.forEach(function (f, i) {
      var t = f.state.turn;
      if (t > 0 && (turns.length === 0 || /** @type {any} */ (turns[turns.length - 1]).turn !== t)) { turns.push({ turn: t, frame: i }); }
    });
    var jump = make(doc, "select", { "aria-label": "Jump to a turn" }, [make(doc, "option", { value: "0" }, ["Start"])].concat(turns.map(function (t) {
      return make(doc, "option", { value: String(t.frame) }, ["Turn " + t.turn]);
    })));
    var speedSel = make(doc, "select", { "aria-label": "Playback speed" }, [0.5, 1, 2, 4].map(function (s) {
      return make(doc, "option", s === 1 ? { value: String(s), selected: "" } : { value: String(s) }, [s + "×"]);
    }));
    var anim = make(doc, "input", { type: "checkbox", checked: "", id: "anim-toggle" }, []);
    var animLabel = make(doc, "label", { for: "anim-toggle" }, [anim, " Animate HP changes"]);
    var viewSel = make(doc, "select", { "aria-label": "Which channel to show" }, [make(doc, "option", { value: "all" }, ["Both channels"])].concat(data.views.map(function (v, i) {
      return make(doc, "option", { value: String(i) }, [v.player + " only"]);
    })));
    var btn = function (/** @type {string} */ text, /** @type {string} */ label) {
      return make(doc, "button", { type: "button", "aria-label": label }, [text]);
    };
    var toStart = btn("⏮", "Back to the start");
    var back = btn("◀", "Previous event");
    var play = btn("▶", "Play");
    var fwd = btn("▶▶", "Next event");
    var toEnd = btn("⏭", "To the end");
    var bar = make(doc, "div", { class: "controls" }, [toStart, back, play, fwd, toEnd, slider, jump, speedSel, viewSel, animLabel, status]);
    root.appendChild(bar);
    if (data.notes && data.notes.length) {
      root.appendChild(make(doc, "div", { class: "notes" }, data.notes.map(function (n) { return make(doc, "p", { class: "note" }, [n]); })));
    }
    var grid = make(doc, "div", { class: "views" }, views.map(function (v) { return v.panel; }));
    root.appendChild(grid);

    /** @param {number} i */
    function hasCaption(i) {
      var f = master.frames[i];
      return !!f && !!(f.caption || f.flagged);
    }

    /** @param {number} i @param {boolean} animated */
    function show(i, animated) {
      position = Math.max(0, Math.min(last, i));
      slider.value = String(position);
      var turnFrame = 0;
      turns.forEach(function (t) { if (t.frame <= position) { turnFrame = t.frame; } });
      jump.value = String(turnFrame);
      var m = master.frames[position];
      status.textContent = (position) + " / " + last + (m && m.state.turn ? " · turn " + m.state.turn : "");
      views.forEach(function (v, vi) {
        var idx = vi === 0 ? position : Math.min(v.frames.length - 1, (maps[vi] || [])[position] || 0);
        var f = v.frames[idx];
        if (!f) { return; }
        v.board.update(f.state, animated);
        var text = "";
        for (var k = idx; k >= 1; k -= 1) {
          var g = v.frames[k];
          if (g && (g.caption || g.flagged)) { text = g.flagged ? "Not recognised: " + g.line : g.caption || ""; break; }
        }
        v.caption.textContent = text;
        var flagged = !!f.flagged;
        v.caption.setAttribute("class", flagged ? "caption flagged" : "caption");
        var requestNo = f.request;
        var request = v.run && requestNo > 0 ? v.run.requests[requestNo - 1] : null;
        var decision = null;
        v.data.decisions.forEach(function (d) { if (d.request_seq === requestNo) { decision = d; } });
        while (v.decision.firstChild) { v.decision.removeChild(v.decision.firstChild); }
        var heading = request ? "Request " + requestNo + " of " + (v.run ? v.run.requests.length : 0) + ", turn " + f.state.turn : "";
        v.decision.appendChild(make(doc, "h3", null, [heading || "Before the first request"]));
        v.decision.appendChild(decisionPanel(doc, request, decision, f.state.labels || {}));
        // the log: lines after the position are dimmed, so a reader sees where the replay is
        v.logRows.forEach(function (r) { r.node.setAttribute("class", (r.node.getAttribute("class") || "line").replace(" future", "") + (r.frame > idx ? " future" : "")); });
        v.rawRows.forEach(function (n, k) { n.setAttribute("class", (n.getAttribute("class") || "line").replace(/ (future|cur)/g, "") + (k + 1 > idx ? " future" : k + 1 === idx ? " cur" : "")); });
      });
    }

    function nextEvent(from, dir) {
      var i = from + dir;
      while (i > 0 && i < last && !hasCaption(i)) { i += dir; }
      return Math.max(0, Math.min(last, i));
    }

    function stop() {
      playing = false;
      play.textContent = "▶";
      play.setAttribute("aria-label", "Play");
      if (timer !== null) { cancel(timer); timer = null; }
    }

    function tick() {
      timer = null;
      if (!playing) { return; }
      if (position >= last) { stop(); return; }
      show(nextEvent(position, 1), true);
      if (position >= last) { stop(); return; }
      timer = later(tick, Math.round(1100 / speed));
    }

    function start() {
      if (position >= last) { show(0, false); }
      playing = true;
      play.textContent = "⏸";
      play.setAttribute("aria-label", "Pause");
      timer = later(tick, Math.round(1100 / speed));
    }

    toStart.addEventListener("click", function () { stop(); show(0, false); });
    toEnd.addEventListener("click", function () { stop(); show(last, false); });
    back.addEventListener("click", function () { stop(); show(nextEvent(position, -1), false); });
    fwd.addEventListener("click", function () { stop(); show(nextEvent(position, 1), true); });
    play.addEventListener("click", function () { if (playing) { stop(); } else { start(); } });
    slider.addEventListener("input", function () { stop(); show(parseInt(slider.value, 10), false); });
    jump.addEventListener("change", function () { stop(); show(parseInt(jump.value, 10), false); });
    speedSel.addEventListener("change", function () { speed = parseFloat(speedSel.value) || 1; });
    anim.addEventListener("change", function () { animateOn = !!anim.checked; });
    viewSel.addEventListener("change", function () {
      views.forEach(function (v, i) { v.panel.hidden = viewSel.value !== "all" && viewSel.value !== String(i); });
    });
    doc.addEventListener("keydown", function (/** @type {any} */ e) {
      if (e.key === "ArrowRight") { stop(); show(nextEvent(position, 1), true); }
      else if (e.key === "ArrowLeft") { stop(); show(nextEvent(position, -1), false); }
      else if (e.key === " " && e.target && e.target.tagName !== "BUTTON" && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") { if (e.preventDefault) { e.preventDefault(); } if (playing) { stop(); } else { start(); } }
    });

    show(0, false);
    return {
      show: function (/** @type {number} */ i) { stop(); show(i, false); },
      position: function () { return position; },
      last: last,
      views: views,
      runs: runs,
      maps: maps,
      turns: turns,
      next: function () { stop(); show(nextEvent(position, 1), true); },
      prev: function () { stop(); show(nextEvent(position, -1), false); },
      play: start,
      pause: stop,
      controls: { slider: slider, jump: jump, speed: speedSel, view: viewSel, anim: anim, play: play, fwd: fwd, back: back, toStart: toStart, toEnd: toEnd },
    };
  }

  /**
   * Read the page's one data block and build the view in `#replay-root`.
   * @param {Document} doc
   * @param {any} win
   */
  function boot(doc, win) {
    var block = doc.getElementById("replay-data");
    var root = doc.getElementById("replay-root");
    if (!block || !root) { return null; }
    var data = JSON.parse(block.textContent || "null");
    var page = init(doc, root, data, {
      raf: function (f) { win.requestAnimationFrame(f); },
      now: function () { return win.performance.now(); },
      setTimeout: function (f, ms) { return win.setTimeout(f, ms); },
      clearTimeout: function (h) { win.clearTimeout(h); },
      reducedMotion: function () { return !!(win.matchMedia && win.matchMedia("(prefers-reduced-motion: reduce)").matches); },
    });
    // `#f=120` opens the replay at frame 120: a link to a moment of the game.
    var at = /^#f=(\d{1,6})$/.exec((win.location && win.location.hash) || "");
    if (at) { page.show(parseInt(at[1] || "0", 10)); }
    return page;
  }

  return { init: init, boot: boot, make: make, align: align, parseChoice: parseChoice, createBoard: createBoard, decisionPanel: decisionPanel };
})();

if (typeof document !== "undefined" && typeof window !== "undefined") { ArView.boot(document, window); }
