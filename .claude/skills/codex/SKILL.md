---
name: claude-ex
description: >
  Local codebase intelligence via MCP. Use for: finding code, understanding
  architecture, tracing dependencies, impact analysis, finding callers,
  understanding what a file/function does in context. Triggers: "what calls",
  "who uses", "what depends on", "where is", "how does X work", "what breaks if",
  "find", "search codebase", "show me", refactoring, architecture questions.
  PREFER these MCP tools over grep/ripgrep for structural queries.
---

# claude-ex — Codebase Intelligence (MCP)

This project has a live code index exposed via MCP. The MCP tools are
**much faster and more precise than grep** for structural questions.

## MCP Tools Available

Use these tools via the MCP connection. They answer in <5ms.

### search_code
Find symbols by name, description, or content. Results ranked by structural
importance (PageRank). Use for any "find X" or "where is X" question.

### get_symbol
Full context for a single symbol: its code, what it depends on, what depends
on it, what else is in the same file. Use before modifying any symbol.

### get_callers
Who calls this function/method. Use before renaming, changing signatures,
or removing a function.

### get_dependents
What files are transitively affected if a file changes. Use before any
refactor that changes exports or file structure.

### get_dependencies
What a symbol imports/uses. Understand what it needs before moving or
modifying it.

### get_architecture
Project overview: top symbols, module map, language breakdown.
Use when you need to understand the overall structure.

## When to prefer MCP tools over grep
- "What calls processPayment?" → get_callers (not grep — grep misses indirect references)
- "What breaks if I change auth.ts?" → get_dependents (not grep — grep can't trace transitive deps)
- "Find the main payment handling code" → search_code (PageRank-weighted, finds the important one)
- "Show me the PaymentService" → get_symbol (includes dependencies + dependents, not just code)

## When to use grep instead
- Simple string search: "find all TODOs" → grep
- Regex patterns: "find all console.log" → grep
- File listing: "show all test files" → find
