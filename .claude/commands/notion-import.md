Import Notion Markdown exports into the Jekyll blog as notes.

Usage: /notion-import [path]

If a path argument is provided, use it as the input. Otherwise default to ~/Downloads/notionexports/.

## Full Workflow

### Step 1: Run the import script

```
python3 scripts/notion_import.py "$ARGUMENTS" --blog-root /Users/Vinnu/Desktop/blog
```

If $ARGUMENTS is empty, use `~/Downloads/notionexports/` as the input path.

The script handles:

- Extracting nested Notion zips
- Matching exports to existing `_notes/` stub files by slug
- Copying images to `assets/img/notes/<slug>/` (one subfolder per note)
- Rewriting image paths in markdown
- Fixing math for kramdown/MathJax compatibility:
  - Blank lines around `$$` display blocks, no blank lines inside them
  - Converting single `$` inline math to `$$` delimiters
  - Replacing `\rarr` with `\rightarrow`
  - Escaping `|` to `\vert` and `||` to `\Vert` inside inline math

Report which `_notes/` files were written and how many images were copied. Surface any warnings.

### Step 2: Rename images to be descriptive

After the script runs, images will have generic names like `Screenshot_2025-01-04_at_10.20.11_AM.png` or `image-1.png`. Read the markdown context around each image reference and rename both the file on disk and the reference in the markdown to a descriptive slug. Examples:

- `Screenshot_2025-01-04_at_10.20.11_AM.png` → `ppo-clipping-mechanism.png`
- `image-1.png` → `forward-kl-mode-covering.png`

Use lowercase kebab-case. The name should describe what the image shows based on the surrounding text.

### Step 3: Update `_pages/notes.md`

If importing a new note, add an entry to `_pages/notes.md`:

```markdown
### [Note Title](/notes/<slug>/)

---
```

### Step 4: Verify rendering

If the dev server is running, fetch the rendered page with curl and check for:

- Spurious `<table>` elements from pipe characters in math
- MathJax loading correctly
- Images loading correctly
