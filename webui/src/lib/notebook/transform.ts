/**
 * Cell compiler: TypeScript cell source → plain JS the executor's engine runs.
 *
 * 1. Strip types with sucrase (line-preserving, so engine error lines match).
 * 2. Turn top-level declarations into assignments to hoisted globals, so cells
 *    share one scope like a notebook and re-running a cell redefines its names
 *    instead of failing with "already declared".
 * 3. Wrap the body in `__nb.run(cellId, async (show) => { ... })` for top-level
 *    await and per-cell output routing. The hoisted `var` line goes on line 1,
 *    so line numbers are unchanged.
 *
 * Limitation: function declarations become assignments, so a function can only
 * be called after its definition has run (no hoisting within a cell).
 */
import { transform } from "sucrase";
import * as acorn from "acorn";

type Node = acorn.Node & Record<string, any>;

function collectNames(pattern: Node, out: Set<string>) {
  switch (pattern.type) {
    case "Identifier":
      out.add(pattern.name);
      break;
    case "ObjectPattern":
      for (const p of pattern.properties) {
        collectNames(p.type === "RestElement" ? p.argument : p.value, out);
      }
      break;
    case "ArrayPattern":
      for (const e of pattern.elements) if (e) collectNames(e, out);
      break;
    case "RestElement":
      collectNames(pattern.argument, out);
      break;
    case "AssignmentPattern":
      collectNames(pattern.left, out);
      break;
  }
}

export class CellCompileError extends Error {}

/** Names the notebook API defines. Keep in sync with RESERVED_NAMES in
 * crates/dies-notebook/src/lib.rs. */
export const RESERVED_NAMES = new Set([
  "frame", "now", "robot", "emit", "show", "wait", "until", "trial", "Plot", "v2", "deg",
  "toDeg", "clamp", "wrapAngle", "stats", "__nb",
]);

export function compileCell(cellId: string, code: string): string {
  let js: string;
  try {
    js = transform(code, {
      transforms: ["typescript"],
      disableESTransforms: true,
    }).code;
  } catch (e: any) {
    throw new CellCompileError(`TypeScript: ${e?.message ?? e}`);
  }

  let ast: acorn.Program;
  try {
    ast = acorn.parse(js, {
      ecmaVersion: "latest",
      sourceType: "script",
      allowAwaitOutsideFunction: true,
      allowReturnOutsideFunction: true,
    });
  } catch (e: any) {
    throw new CellCompileError(`Syntax: ${e?.message ?? e}`);
  }

  const names = new Set<string>();
  const edits: { start: number; end: number; text: string }[] = [];
  for (const node of ast.body as Node[]) {
    if (node.type === "VariableDeclaration") {
      const parts: string[] = [];
      for (const d of node.declarations as Node[]) {
        collectNames(d.id, names);
        const pattern = js.slice(d.id.start, d.id.end);
        if (d.init) {
          const init = js.slice(d.init.start, d.init.end);
          parts.push(d.id.type === "Identifier" ? `${pattern} = ${init}` : `(${pattern} = ${init})`);
        } else if (d.id.type === "Identifier") {
          parts.push(`${pattern} = undefined`);
        }
      }
      edits.push({ start: node.start, end: node.end, text: parts.length ? `${parts.join(", ")};` : ";" });
    } else if (
      (node.type === "FunctionDeclaration" || node.type === "ClassDeclaration") &&
      node.id
    ) {
      names.add(node.id.name);
      edits.push({ start: node.start, end: node.start, text: `${node.id.name} = ` });
      edits.push({ start: node.end, end: node.end, text: ";" });
    }
  }

  for (const name of names) {
    if (RESERVED_NAMES.has(name)) {
      throw new CellCompileError(`"${name}" is a built-in name. Pick another name.`);
    }
  }

  let body = js;
  for (const e of edits.sort((a, b) => b.start - a.start || b.end - a.end)) {
    body = body.slice(0, e.start) + e.text + body.slice(e.end);
  }

  const hoist = names.size ? `var ${[...names].join(", ")}; ` : "";
  return `${hoist}__nb.run(${JSON.stringify(cellId)}, async (show) => {${body}\n});`;
}
