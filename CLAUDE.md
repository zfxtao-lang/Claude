# CLAUDE.md

## Project Overview

This is a **Chat History Viewer** (聊天记录查看器) — a single-file, zero-dependency web application for viewing, searching, editing, and exporting AI chat conversation histories. The UI is in Chinese (zh-CN) and styled to resemble a WeChat-like messaging interface.

## Repository Structure

```
.
├── chat-viewer.html   # The entire application (HTML + CSS + JS in one file, ~1640 lines)
├── sample-chat.json   # Sample conversation data for testing/demo
├── README.md          # Brief project description
└── CLAUDE.md          # This file
```

This is a **single-file architecture** — all HTML, CSS, and JavaScript live in `chat-viewer.html`. There is no build system, no bundler, no external dependencies, and no package manager.

## Technology Stack

- **Pure HTML/CSS/JavaScript** — no frameworks, no libraries
- **CSS custom properties** for theming (light/dark mode via `[data-theme="dark"]`)
- **Vanilla DOM manipulation** — uses `document.getElementById` (aliased as `$()`) and `innerHTML`
- **Markdown rendering** — custom `formatText()` function handles code blocks, inline code, bold, italic, tables, lists, and links
- **File API** — drag-and-drop and file input for importing JSON conversation files

## Key Features

- **Multi-format JSON import**: Supports ChatGPT export format, generic conversation arrays, and custom formats
- **Conversation sidebar**: Filterable, sortable conversation list with statistics
- **Message display**: User/assistant/system messages with thinking block toggle
- **Full-text search**: In-conversation search with highlight navigation (Ctrl+F)
- **Message editing**: Inline edit and delete individual messages
- **Selective export**: Checkbox-based message selection for partial export
- **Export formats**: Markdown, JSON, and plain text
- **Dark/light theme**: Toggle with Ctrl+D, persisted in localStorage
- **Mobile responsive**: Adaptive layout with bottom navigation bar for small screens
- **Drag-and-drop**: File import via drag-and-drop overlay

## Code Architecture (chat-viewer.html)

### CSS (lines 7–709)
- CSS custom properties in `:root` and `[data-theme="dark"]` for full theming
- Layout uses Flexbox for the app shell (sidebar + main area)
- Mobile breakpoint at `768px` with responsive overrides
- Component styles: sidebar, chat messages (user/assistant/system), search bar, modals, toasts

### HTML (lines 711–830)
- `.app` container with `.sidebar` and `.main` panels
- Sidebar: filter input, sort buttons, conversation list, stats
- Main: chat header, search bar, select toolbar, message area (`#chatBody`)
- Overlays: drop overlay, export modal, toast notification, back-to-top button

### JavaScript (lines 832–1643)
Key global state:
- `conversations` — array of parsed conversation objects
- `currentConv` — index of currently displayed conversation
- `currentMessages` — messages of the active conversation
- `selectMode` / `selectedMessages` — selection state for export

Important functions:
| Function | Purpose |
|---|---|
| `handleFiles(files)` | Entry point for file import |
| `parseAndLoad(text, filename)` | Parses JSON text, detects format, loads conversations |
| `parseChatGPTConversation(data)` | Handles ChatGPT's tree-structured export format |
| `parseGenericConversation(data)` | Handles flat array conversation format |
| `renderConvList()` | Renders sidebar conversation list with filtering/sorting |
| `loadConversation(index)` | Loads a conversation and triggers message rendering |
| `renderMessages()` | Renders all messages in the chat body |
| `formatText(text)` | Converts markdown-like text to HTML (code blocks, tables, lists, etc.) |
| `performSearch()` | Full-text search with regex highlight |
| `doExport(format)` | Exports conversation as Markdown, JSON, or plain text |
| `toggleTheme()` | Switches between light and dark theme |

### JSON Data Format (sample-chat.json)

The sample data is an array of conversation objects:
```json
[
  {
    "title": "Conversation Title",
    "created_at": "2025-12-15T10:30:00Z",
    "messages": [
      {
        "role": "user" | "assistant" | "system",
        "content": "string or array of content blocks",
        "timestamp": "ISO 8601 date string"
      }
    ]
  }
]
```

Assistant messages can have structured content with `thinking` and `text` blocks:
```json
{
  "role": "assistant",
  "content": [
    { "type": "thinking", "thinking": "..." },
    { "type": "text", "text": "..." }
  ]
}
```

## Development Workflow

### Running the App
Open `chat-viewer.html` directly in a browser — no server required. For testing, drag `sample-chat.json` onto the page or use the import button.

### Making Changes
Since this is a single-file app, all edits go into `chat-viewer.html`. The file is organized in three clear sections:
1. `<style>` — all CSS
2. `<body>` — all HTML structure
3. `<script>` — all JavaScript logic

### Testing
- Open `chat-viewer.html` in a browser and interact manually
- Test with `sample-chat.json` as sample data
- Verify both light and dark themes
- Test mobile layout by resizing the browser or using device emulation
- No automated test suite exists

## Conventions

- **Language**: UI text and comments are in Chinese (Simplified)
- **Commit messages**: Written in Chinese, prefixed with `feat:` or similar conventional commit types
- **Single-file pattern**: Do not split into separate CSS/JS files — the project intentionally ships as one self-contained HTML file
- **No external dependencies**: Do not add CDN links, npm packages, or external libraries
- **DOM access**: Use the `$(id)` shorthand (alias for `document.getElementById`) throughout the JavaScript
- **Theme variables**: All colors must use CSS custom properties from `:root` to support dark mode
- **Escaping**: Always use `escapeHtml()` when inserting user-provided text into innerHTML to prevent XSS
