// interpreter.cjs is a classic script whose typedefs make the checker treat it as a module, so the
// global it defines for view.cjs is declared here for the type check only.
declare var ArReplay: {
  interpretChannel(player: string, lines: string[], dex?: any): {
    frames: any[];
    flags: { index: number; line: string; reason: string }[];
    requests: any[];
  };
  [name: string]: any;
};
