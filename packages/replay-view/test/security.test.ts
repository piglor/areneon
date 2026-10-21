import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { FakeDocument, FakeNode } from "./dom.ts";
import { WEB, fullDex, loadView } from "./support.ts";

const { view } = loadView();
const LS = String.fromCharCode(0x2028);
const PS = String.fromCharCode(0x2029);
const NICK = "</script><img src=x onerror=alert(1)>";
const OPP = 'x" onmouseover="alert(2)" a=\'b\'';
const LOG = `a </script> b <!-- c ${LS} d ${PS} e "q" 's' & <b>`;
const POLICY = "policy</script><svg onload=alert(3)>";
const LABEL = 'lab"el<x>';
const CANARIES = [NICK, OPP, LOG, POLICY, LABEL];

const ALLOWED_TAGS = new Set([
  "div", "p", "h2", "h3", "h4", "section", "details", "summary", "ul", "ol", "li", "button", "select", "option",
  "input", "label", "span", "svg", "title", "rect", "line", "text", "g", "circle", "path",
].map((t) => t.toUpperCase()));
const ALLOWED_ATTRS = new Set([
  "class", "id", "type", "min", "max", "value", "aria-label", "for", "open", "checked", "selected", "data-player",
  "viewBox", "role", "x", "y", "x1", "x2", "y1", "y2", "width", "height", "rx", "cx", "cy", "r", "d", "transform", "text-anchor",
]);

function problems(root: FakeNode, doc: FakeDocument): string[] {
  const found: string[] = [];
  for (const tag of doc.created) if (!ALLOWED_TAGS.has(tag.toUpperCase())) found.push(`a tag the template lacks: ${tag}`);
  for (const node of root.all()) {
    for (const [name, value] of node.attrs) {
      if (!ALLOWED_ATTRS.has(name)) found.push(`an attribute the template lacks: ${name}`);
      if (/^(style|on)/i.test(name)) found.push(`a style or handler attribute: ${name}`);
      if (CANARIES.some((c) => value.includes(c))) found.push(`a canary in an attribute value: ${name}`);
      if (name === "class" && !/^[A-Za-z0-9 _-]*$/.test(value)) found.push(`a class that is not a plain token list: ${value}`);
    }
  }
  return found;
}

const request = (nick: string) =>
  "|request|" +
  JSON.stringify({
    active: [{ moves: [{ move: LOG, id: "tackle", pp: 5, maxpp: 8, target: "normal", disabled: false }] }, { moves: [] }],
    side: {
      name: "A",
      id: "p1",
      pokemon: [
        { ident: `p1: ${nick}`, details: "Venusaur, L50, F", condition: "150/150", active: true, stats: {}, moves: ["tackle"], baseAbility: "overgrow", item: "", ability: "overgrow" },
        { ident: "p1: Charizard", details: "Charizard, L50, F", condition: "150/150", active: true, stats: {}, moves: ["tackle"], baseAbility: "blaze", item: "", ability: "blaze" },
      ],
    },
  });

function hostileData() {
  const lines = [
    "|gametype|doubles",
    "|player|p1|A||",
    `|player|p2|${OPP}||`,
    "|clearpoke",
    "|poke|p2|Blastoise, L50, M|",
    request(NICK),
    "|teamsize|p1|2",
    "|teamsize|p2|1",
    "|start",
    `|switch|p1a: ${NICK}|Venusaur, L50, F|150/150`,
    "|switch|p1b: Charizard|Charizard, L50, F|150/150",
    `|switch|p2a: ${OPP}|Blastoise, L50, M|100/100`,
    "|turn|1",
    request(NICK),
    `|move|p2a: ${OPP}|${LOG}|p1a: ${NICK}`,
    `|-damage|p1a: ${NICK}|50/150|[from] item: ${LOG}`,
    `|-status|p1a: ${NICK}|par`,
    `|-start|p1a: ${NICK}|Substitute`,
    `|-message|${LOG}`,
    `|${LOG}|unknown kind`,
    `|-weather|${LOG}`,
    "|turn|2",
  ];
  const dex = fullDex();
  return {
    title: LOG,
    views: [{
      player: "p1",
      label: LABEL,
      lines,
      team: [{ name: NICK, species: "Venusaur", item: LOG, ability: LABEL, moves: [LOG, NICK] }],
      decisions: [{ choice: `move 1 1, ${LOG}`, request_seq: 2, policy: POLICY }],
      result: LABEL,
    }],
    dex,
    links: [],
    notes: [LOG],
  };
}

describe("hostile strings stay text (ADR-017 4c)", () => {
  it("every canary reaches the page only as text, in allowed elements and attributes", () => {
    const doc = new FakeDocument();
    const root = doc.createElement("div");
    const page = view.init(doc, root, hostileData());
    page.show(page.last);
    for (let i = 0; i < page.last; i += 7) page.show(i);
    page.show(page.last);
    expect(problems(root, doc)).toEqual([]);
    const text = root.all().map((n) => n.text ?? "").join("\n");
    for (const canary of [NICK, LOG, POLICY, LABEL]) expect(text, canary).toContain(canary);
    expect(text).toContain("Not recognised:");
    expect(root.byClass("flagged").length).toBeGreaterThan(0);
    // the raw log keeps every line, hostile ones included, as text
    expect(root.byClass("raw")[0]!.all().map((n) => n.text ?? "").join("")).toContain(LOG);
  });

  it("the checker catches what it checks for (negative controls)", () => {
    const doc = new FakeDocument();
    const root = doc.createElement("div");
    const bad = doc.createElement("div");
    bad.setAttribute("class", NICK);
    bad.setAttribute("onclick", "x()");
    bad.setAttribute("style", "color:red");
    bad.setAttribute("title", NICK);
    bad.setAttribute("aria-label", NICK);
    root.appendChild(bad);
    doc.createElement("script");
    const found = problems(root, doc).join("\n");
    expect(found).toContain("a tag the template lacks: script");
    expect(found).toContain("a style or handler attribute: onclick");
    expect(found).toContain("a style or handler attribute: style");
    expect(found).toContain("an attribute the template lacks: title");
    expect(found).toContain("a canary in an attribute value: aria-label");
    expect(found).toContain("a class that is not a plain token list");
  });

  it("builds no node from data: a tag, an attribute name or a class", () => {
    const doc = new FakeDocument();
    const root = doc.createElement("div");
    view.init(doc, root, hostileData());
    expect(new Set(doc.created.map((t) => t.toLowerCase()))).toEqual(
      new Set(doc.created.map((t) => t.toLowerCase()).filter((t) => ALLOWED_TAGS.has(t.toUpperCase()))),
    );
  });
});

describe("the viewer's own code", () => {
  const sources = ["interpreter.js", "view.js"].map((f) => [f, readFileSync(join(WEB, f), "utf8")] as const);
  const css = readFileSync(join(WEB, "replay.css"), "utf8");

  it("writes through no HTML sink and calls out to nothing", () => {
    for (const [name, text] of sources) {
      for (const sink of [
        "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "Function(", "fetch(",
        "XMLHttpRequest", "import(", "WebSocket", "sendBeacon", "EventSource", "localStorage", "sessionStorage", "postMessage",
      ]) {
        expect(text, `${name} uses ${sink}`).not.toContain(sink);
      }
      expect(text, name).not.toMatch(/https?:/);
      expect(text, name).not.toMatch(/\/\/[a-z]/);
      expect(text, name).not.toMatch(/<\/script|<!--/i);
      expect(text, name).not.toContain("\r");
    }
  });

  it("sets an attribute only through the guard", () => {
    const text = sources.map(([, s]) => s).join("\n");
    // every setAttribute call in the view names a fixed attribute: no style, no handler
    const calls = [...text.matchAll(/setAttribute\(([^,)]+)/g)].map((m) => m[1]);
    for (const name of calls) expect(name, `setAttribute(${name})`).toMatch(/^("[A-Za-z-]+"|name)$/);
    expect(text).not.toMatch(/setAttribute\(\s*["'](style|on[a-z]+)["']/i);
    expect(text).not.toMatch(/\.style\b/);
  });

  it("the style block has no url, import or remote font", () => {
    expect(css).not.toMatch(/url\(|@import|@font-face|https?:/i);
    expect(css).not.toContain("\r");
  });
});
