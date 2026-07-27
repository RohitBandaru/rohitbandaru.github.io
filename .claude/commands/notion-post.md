Turn a Notion Markdown export into a Jekyll blog post in `_posts/`.

Usage: /notion-post [path-to-zip]

For course/lecture notes that belong in `_notes/` instead, use `/notion-import`.

## Step 1: Pick the slug, image dir, and date

Read the export's title first, then decide three things before running anything:

- **slug** — the filename after the date, and therefore the URL (`/blog/<slug>/`).
  Title case with hyphens, punctuation dropped. Notion titles are often clumsy,
  so tighten them: `Foundation Models for Robotics (VLA)` →
  `Foundation-Models-for-Robotics-VLA`.
- **img-dir** — short snake_case name under `assets/img/blog/`. Existing ones are
  `vla`, `vae`, `world_models`, `transformer_pt1`. Keep it short; it appears in
  every image path.
- **date** — the publish date. The script otherwise guesses from the earliest
  screenshot filename, which is usually when research _started_, not when the
  post is going out. Pass `--date` explicitly.

## Step 2: Run the import

```
python3 scripts/notion_import.py <zip> --as post \
  --slug <Slug> --img-dir <name> --date YYYY-MM-DD \
  --blog-root /Users/Vinnu/Desktop/blog
```

This extracts the nested zip, copies images, fixes Notion math for
kramdown/MathJax, converts markdown images to `figure.liquid` includes (first
eager, rest lazy), and warns about any referenced image missing from disk.

**Do not ignore a missing-image warning.** A post shipped with a dangling image
path 404s silently in production — the `<picture>` fallback hides it from you
but not from readers.

## Step 3: Rename the images descriptively

The import leaves Notion's names: `Screenshot_2025-09-27_at_9.41.56_PM.png`,
`image-2.png`. For each one, read the surrounding paragraphs and the figure's
`alt`/`source`, then rename **both the file on disk and every reference in the
markdown**. Lowercase kebab-case, describing what the figure shows:

- `Screenshot_2025-09-27_at_9.41.56_PM.png` → `embodiment_distribution.png`
- `image-2.png` → `trajectory.png`

Then re-run the missing-image check to confirm nothing was left pointing at an
old name:

```
python3 -c "
import re,pathlib
p=pathlib.Path('_posts/<file>.md'); s=p.read_text()
print([x for x in re.findall(r'path=\"([^\"]+)\"',s) if not pathlib.Path(x).exists()])
"
```

## Step 4: Fill in the front matter

The script leaves `TODO` markers. Replace all of them:

- **description** — one or two sentences. This is the meta description _and_ it
  renders under the post on `/blog/`. Name the specific models covered; verify
  each one actually appears in the post before citing it.
- **tags** — **space separated, never commas.** A bare YAML string is split on
  whitespace, so `computer-vision, deep-learning` yields the tag
  `computer-vision,` whose archive page then collides with and overwrites the
  real `computer-vision` one. Reuse existing tags rather than inventing
  near-duplicates (`transformer`, not `transformers`).
- **thumbnail** — the figure that best represents the post. Shown on `/blog/`.
- **keywords** — optional, ignored by search engines. Skip it or keep it short.

## Step 5: Check the social card

`thumbnail` doubles as the Open Graph image unless overridden. Cards want
roughly 1.91:1 and at least 600px wide. Check it:

```
sips -g pixelWidth -g pixelHeight assets/img/blog/<dir>/<thumb>.png
```

- Portrait or very wide/short → set `og_image:` (plus `og_image_width:` and
  `og_image_height:`) to a better landscape figure from the post. `og_image`
  takes priority over `thumbnail`.
- Nothing in the post is big enough → set `twitter_card: summary` so it renders
  as a small card instead of a stretched one.

## Step 6: Link it into the rest of the blog

New posts start orphaned. Add links **only where the prose already refers to
something covered elsewhere** — do not bolt on a "related reading" list, the
related-posts block at the bottom is automatic.

Look for phrases like "in a previous blog post", "as we explored", "see part 1".
Use root-relative paths (`/blog/Vision-Language-Models/`), not absolute URLs.
Also consider adding a link _from_ an older post if it naturally sets this one up.

## Step 7: Verify before committing

```
npx prettier --write _posts/<file>.md
bundle exec jekyll build
```

Check the build output for:

- `Conflict: ... shared by multiple files` — a tag collision, almost always a
  stray comma in `tags:`
- Liquid errors from unescaped braces in code blocks
- Spurious `<table>` elements in the rendered HTML, caused by pipe characters
  inside math that should be `\vert` / `\Vert`

Then confirm the rendered page: exactly one `loading="eager"` image, every
figure resolving, and MathJax rendering the display blocks.
