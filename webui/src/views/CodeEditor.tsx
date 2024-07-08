import React, { useRef, useEffect, useState } from "react";
import * as monaco from "monaco-editor";
import * as acorn from "acorn";
import * as acornLoose from "acorn-loose";
import * as walk from "acorn-walk";
import { Play, Save } from "lucide-react";
import { useLocalStorage } from "@uidotdev/usehooks";
import { Button } from "@/components/ui/button";
import { useResizeObserver } from "@/lib/useResizeObserver";
import * as math from "mathjs";

interface CodeEditorProps {
  globals: Record<string, any>;
  onRun?: (value: string) => void;
}

const MAX_CODE_HEIGHT = 200;

const CodeEditor: React.FC<CodeEditorProps> = ({ globals, onRun }) => {
  const editorRef = useRef<HTMLDivElement>(null);
  const contRef = useRef<HTMLDivElement>(null);
  const editorObjRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const [storedCode, setStoredCode] = useLocalStorage("code", "");
  const [dirty, setDirty] = useState(false);

  const globalsRef = useRef(globals);
  globalsRef.current = globals;
  const onRunRef = useRef(onRun);
  onRunRef.current = onRun;

  useResizeObserver({
    ref: contRef,
    onResize: () => {
      if (editorObjRef.current) {
        editorObjRef.current.layout();
      }
    },
  });

  const handleRun = () => {
    if (editorObjRef.current && onRunRef.current) {
      setDirty(false);
      setStoredCode(editorObjRef.current.getValue());
      onRunRef.current(editorObjRef.current.getValue());
    }
  };

  useEffect(() => {
    if (editorRef.current) {
      if (storedCode.length > 0) {
        onRunRef.current && onRunRef.current(storedCode);
      }

      const newEditor = monaco.editor.create(editorRef.current, {
        value: storedCode,
        language: "typescript",
        minimap: { enabled: false },
        lineNumbers: "off",
        roundedSelection: false,
        scrollBeyondLastLine: false,
        readOnly: false,
        theme: "vs-dark",
        suggestOnTriggerCharacters: true,
      });

      editorObjRef.current = newEditor;

      const updateEditorHeight = () => {
        const contentHeight = Math.min(
          MAX_CODE_HEIGHT,
          Math.max(1, newEditor.getContentHeight())
        );
        editorRef.current!.style.height = `${contentHeight}px`;
        newEditor.layout();
      };

      newEditor.onDidContentSizeChange(updateEditorHeight);
      updateEditorHeight();

      // Add event listener for Ctrl+Enter
      newEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
        setDirty(false);
        setStoredCode(newEditor.getValue());
        onRunRef.current && onRunRef.current(newEditor.getValue());
      });

      // Add event listener for changes
      newEditor.onDidChangeModelContent(() => {
        setDirty(true);
      });

      monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
        target: monaco.languages.typescript.ScriptTarget.ESNext,
        allowNonTsExtensions: true,
      });

      // Trigger completion on focus
      newEditor.onDidFocusEditorText(() => {
        newEditor.trigger("keyboard", "editor.action.triggerSuggest", {});
      });

      const standardGlobals = {
        math,
      };

      const allGlobals = { ...standardGlobals, ...globalsRef.current };

      const getCompletionItems = (
        obj: any
      ): monaco.languages.CompletionItem[] => {
        const completionItems: monaco.languages.CompletionItem[] = [];

        const keys = Object.keys(Object.getOwnPropertyDescriptors(obj));
        for (const key of keys) {
          const value = obj[key];
          const completionItem: monaco.languages.CompletionItem = {
            label: key,
            kind: getCompletionItemKind(value),
            insertText: Array.isArray(obj) ? `[${key}]` : key,
            detail: typeof value === "function" ? "(method)" : "(property)",
          } as monaco.languages.CompletionItem;

          if (typeof value === "object" && value !== null) {
            completionItems.push(completionItem);
          } else {
            completionItems.push(completionItem);
          }
        }

        return completionItems;
      };

      const disposable = monaco.languages.registerCompletionItemProvider(
        "typescript",
        {
          triggerCharacters: [".", "✖", ""],
          provideCompletionItems: (model, position) => {
            const textUntilPosition = model.getValueInRange({
              startLineNumber: 1,
              startColumn: 1,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            });

            try {
              const ast = acornLoose.parse(textUntilPosition, {
                ecmaVersion: 2020,
              });

              let lastNode: acorn.Node | null = null;
              walk.full(ast, (node) => {
                if (
                  lastNode === null &&
                  node.start <= textUntilPosition.length &&
                  node.end >= textUntilPosition.length
                ) {
                  lastNode = node;
                }
              });
              lastNode = lastNode as acorn.Node | null;

              if (lastNode && lastNode.type === "MemberExpression") {
                let parts: string[] = [];
                let node: any = lastNode;
                if (node.property.name !== "✖") {
                  parts.push(node.property.name);
                }
                while (node.object) {
                  if (node.object.type === "MemberExpression") {
                    if (
                      node.object.property &&
                      node.object.property.type === "Identifier"
                    ) {
                      parts = [node.object.property.name, ...parts];
                    }
                    node = node.object;
                  } else if (node.object.type === "Identifier") {
                    parts = [node.object.name, ...parts];
                    break;
                  } else {
                    break;
                  }
                }

                let currentObj: any = allGlobals;
                for (const part of parts) {
                  currentObj = currentObj ? currentObj[part] : undefined;
                }

                if (currentObj) {
                  return {
                    suggestions: getCompletionItems(currentObj),
                  };
                }
              }

              if (lastNode && lastNode.type === "Identifier") {
                const currentObj =
                  allGlobals[(lastNode as any).name as keyof typeof allGlobals];
                if (currentObj && typeof currentObj === "object") {
                  return {
                    suggestions: getCompletionItems(currentObj),
                  };
                }
              }

              // If not in a member expression or object not found, return all globals
              return {
                suggestions: getCompletionItems(allGlobals),
              };
            } catch (error) {
              console.error("Parsing error:", error);
              return {
                suggestions: getCompletionItems(allGlobals),
              };
            }
          },
        }
      );

      return () => {
        disposable.dispose();
        newEditor.dispose();
      };
    }
  }, []);

  const getCompletionItemKind = (
    value: any
  ): monaco.languages.CompletionItemKind => {
    if (typeof value === "function") {
      return monaco.languages.CompletionItemKind.Function;
    } else if (typeof value === "object") {
      return monaco.languages.CompletionItemKind.Module;
    } else {
      return monaco.languages.CompletionItemKind.Variable;
    }
  };

  return (
    <div
      ref={contRef}
      className="relative w-full flex flex-row items-start gap-2"
    >
      <div ref={editorRef} className="w-full min-h-10 *:z-50" />

      <Button onClick={handleRun} variant={dirty ? "default" : "ghost"}>
        <Save />
      </Button>
    </div>
  );
};

export default CodeEditor;
