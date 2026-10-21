/** A small DOM stand-in: enough for the view's `createElement`, attributes, text and listeners. */
export class FakeNode {
  tagName: string;
  ns: string | null;
  attrs = new Map<string, string>();
  children: FakeNode[] = [];
  listeners = new Map<string, ((e: any) => void)[]>();
  text: string | null = null;
  hidden = false;
  value = "";
  checked = true;
  parent: FakeNode | null = null;
  constructor(tag: string, ns: string | null = null) {
    this.tagName = tag.toUpperCase();
    this.ns = ns;
  }
  get nodeName(): string {
    return this.tagName;
  }
  setAttribute(name: string, value: string): void {
    this.attrs.set(name, String(value));
    if (name === "value") this.value = String(value);
  }
  getAttribute(name: string): string | null {
    return this.attrs.get(name) ?? null;
  }
  appendChild(node: FakeNode): FakeNode {
    node.parent = this;
    this.children.push(node);
    return node;
  }
  removeChild(node: FakeNode): FakeNode {
    this.children = this.children.filter((c) => c !== node);
    return node;
  }
  get firstChild(): FakeNode | null {
    return this.children[0] ?? null;
  }
  set textContent(value: string) {
    this.children = [];
    this.text = value;
  }
  get textContent(): string {
    return (this.text ?? "") + this.children.map((c) => c.textContent).join("");
  }
  addEventListener(type: string, fn: (e: any) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  fire(type: string, event: any = {}): void {
    for (const fn of this.listeners.get(type) ?? []) fn({ target: this, ...event });
  }
  all(): FakeNode[] {
    return [this, ...this.children.flatMap((c) => c.all())];
  }
  find(predicate: (n: FakeNode) => boolean): FakeNode[] {
    return this.all().filter(predicate);
  }
  byClass(name: string): FakeNode[] {
    return this.find((n) => (n.attrs.get("class") ?? "").split(" ").includes(name));
  }
}

class TextNode extends FakeNode {
  constructor(value: string) {
    super("#text");
    this.text = value;
  }
}

export class FakeDocument extends FakeNode {
  ids = new Map<string, FakeNode>();
  created: string[] = [];
  constructor() {
    super("#document");
  }
  createElement(tag: string): FakeNode {
    this.created.push(tag);
    return new FakeNode(tag);
  }
  createElementNS(ns: string, tag: string): FakeNode {
    this.created.push(tag);
    return new FakeNode(tag, ns);
  }
  createTextNode(value: string): FakeNode {
    return new TextNode(value);
  }
  getElementById(id: string): FakeNode | null {
    return this.ids.get(id) ?? null;
  }
}
