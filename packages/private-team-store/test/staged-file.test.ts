import { describe, expect, it } from "vitest";
import { ImportPlanError } from "../src/import-error.ts";
import { parseStagedFile } from "../src/staged-file.ts";

const ID = "0123456789abcdef0123456789abcdef";
const BODY = "Gallade @ Wise Glasses\nAbility: Steadfast\n";
const code = (text: string): string | null => {
  try {
    parseStagedFile(text, 4);
    return null;
  } catch (error) {
    if (!(error instanceof ImportPlanError)) throw error;
    return error.code;
  }
};

describe("parseStagedFile", () => {
  it("returns a headerless file whole", () => {
    expect(parseStagedFile(BODY, 1)).toEqual({ team_id: null, tags: null, paste: BODY });
  });

  it("reads team_id and tags, tags sorted, and removes the header", () => {
    expect(parseStagedFile(`#! team_id=${ID} tags=zeta,alpha,m-1\n${BODY}`, 1)).toEqual({ team_id: ID, tags: ["alpha", "m-1", "zeta"], paste: BODY });
  });

  it("takes either key alone", () => {
    expect(parseStagedFile(`#! team_id=${ID}\n${BODY}`, 1)).toEqual({ team_id: ID, tags: null, paste: BODY });
    expect(parseStagedFile(`#! tags=a\n${BODY}`, 1)).toEqual({ team_id: null, tags: ["a"], paste: BODY });
  });

  it("reads CRLF and bare CR line ends", () => {
    expect(parseStagedFile(`#! tags=a\r\nGallade\r\nAbility: X\r\n`, 1).paste).toBe("Gallade\nAbility: X\n");
  });

  it("takes sixteen tags and no more", () => {
    const tags = (n: number): string => Array.from({ length: n }, (_, k) => `t${String.fromCharCode(97 + k)}`).join(",");
    expect(parseStagedFile(`#! tags=${tags(16)}\n${BODY}`, 1).tags).toHaveLength(16);
    expect(code(`#! tags=${tags(17)}\n${BODY}`)).toBe("header_malformed");
  });

  it.each([
    ["a bare #", `#\n${BODY}`],
    ["a plain comment first", `# my team\n${BODY}`],
    ["an uppercase id", `#! team_id=${ID.toUpperCase()}\n${BODY}`],
    ["a short id", `#! team_id=${ID.slice(1)}\n${BODY}`],
    ["tags before the id", `#! tags=a team_id=${ID}\n${BODY}`],
    ["an empty tags value", `#! tags=\n${BODY}`],
    ["a space after a comma", `#! tags=a, b\n${BODY}`],
    ["an uppercase tag", `#! tags=Aa\n${BODY}`],
    ["a tag starting with a hyphen", `#! tags=-a\n${BODY}`],
    ["a 33-character tag", `#! tags=${"a".repeat(33)}\n${BODY}`],
    ["a repeated tag", `#! tags=a,b,a\n${BODY}`],
    ["an unknown key", `#! team_id=${ID} note=x\n${BODY}`],
    ["trailing space", `#! team_id=${ID} \n${BODY}`],
    ["no space after #!", `#!team_id=${ID}\n${BODY}`],
  ])("refuses %s in the header", (_name, text) => {
    expect(code(text)).toBe("header_malformed");
  });

  it.each([
    ["a second header", `#! tags=a\n#! tags=b\n${BODY}`],
    ["a comment after the header", `#! tags=a\n${BODY}# note\n`],
    ["a comment in a headerless file", `${BODY}# note\n`],
    ["an indented comment", `${BODY}   # note\n`],
    ["a header that is not the first line", `\n#! tags=a\n${BODY}`],
  ])("refuses %s", (_name, text) => {
    expect(code(text)).toBe("comment_line");
  });

  it("reports the file position and nothing from the text", () => {
    try {
      parseStagedFile("#! team_id=QQQ-CANARY\n", 4);
      expect.unreachable();
    } catch (error) {
      expect((error as ImportPlanError).position).toEqual({ subject: "file", index: 4 });
      expect((error as Error).message).not.toContain("CANARY");
    }
  });
});
