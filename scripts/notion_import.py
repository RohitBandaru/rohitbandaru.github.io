#!/usr/bin/env python3
"""
Import Notion Markdown exports into Jekyll _notes/ or _posts/.

Handles:
  - Nested zips (Notion exports an outer zip containing an inner zip)
  - Image extraction and renaming (generic "image.png" → "image-1.png")
  - Image path rewriting to /assets/img/{notes,blog}/<slug>/
  - Date inference from screenshot filenames
  - Matching exported notes to existing _notes/ stub files
  - Post mode (--as post): _posts/YYYY-MM-DD-Slug.md, figure.liquid includes
    with the first image eager and the rest lazy, and a check that every
    referenced image actually exists on disk
  - Math formatting fixes for kramdown/MathJax compatibility:
    - Ensures blank lines around $$ display math blocks
    - Removes blank lines inside $$ blocks
    - Converts single $ inline math to $$ delimiters
    - Replaces \\rarr with \\rightarrow
    - Escapes | to \\vert and || to \\Vert in inline math

Usage:
    python3 scripts/notion_import.py <zip_or_directory> [--blog-root <path>]
    python3 scripts/notion_import.py <zip> --as post --slug <Slug> --img-dir <name>

Examples:
    python3 scripts/notion_import.py ~/Downloads/notionexports/
    python3 scripts/notion_import.py ~/Downloads/notionexports/mypage.zip
    python3 scripts/notion_import.py ~/Downloads/vla.zip --as post \\
        --slug Foundation-Models-for-Robotics-VLA --img-dir vla --date 2025-09-28
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import unquote


# Markdown image: ![caption](path)
# - caption may contain one level of [nested](links)
# - path may contain one level of (parens); Notion page folders often do,
#   e.g. "Foundation Models for Robotics (VLA) 1e93.../shot.png"
IMAGE_RE = r"!\[((?:[^\[\]]|\[[^\]]*\])*)\]\(((?:[^()]|\([^()]*\))*)\)"


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def extract_nested_zip(zip_path: Path, dest: Path) -> None:
    """Extract outer zip, then extract any inner zip found inside it."""
    with zipfile.ZipFile(zip_path) as outer:
        outer.extractall(dest)
    for inner in dest.glob("*.zip"):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(dest)
        inner.unlink()


def find_md_file(root: Path) -> Path | None:
    mds = list(root.rglob("*.md"))
    return mds[0] if mds else None


def find_image_dir(md_file: Path) -> Path | None:
    """Return the image subdirectory next to the .md file, if any."""
    for d in md_file.parent.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            return d
    return None


def infer_date(img_dir: Path | None) -> str:
    """Infer creation date from earliest screenshot filename, or return None."""
    if img_dir is None or not img_dir.is_dir():
        return None
    dates = []
    for f in img_dir.iterdir():
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m:
            dates.append(m.group(1))
    return min(dates) if dates else None


def copy_images(img_dir: Path, dest_dir: Path) -> dict[str, str]:
    """
    Copy images to dest_dir. Rename generic 'image.png', 'image 1.png' etc.
    to 'image-1.png', 'image-2.png'. Return mapping of old name → new name.
    """
    if not img_dir or not img_dir.is_dir():
        return {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    counter = 1
    for f in sorted(img_dir.iterdir()):
        if not f.is_file():
            continue
        if re.match(r"^image(\s+\d+)?\.\w+$", f.name, re.IGNORECASE):
            new_name = f"image-{counter}{f.suffix}"
            counter += 1
        else:
            new_name = f.name.replace(" ", "_")
        shutil.copy2(f, dest_dir / new_name)
        mapping[f.name] = new_name
    return mapping


def rewrite_image_refs(
    content: str, slug: str, mapping: dict[str, str], asset_root: str = "notes"
) -> str:
    """Replace Notion relative image paths with Jekyll /assets/img/<asset_root>/<slug>/ paths.

    The alt text pattern allows nested [brackets] (e.g. '[source](url) caption').
    Only rewrites paths that look like local relative paths (no http/https).
    """
    def replace(m):
        alt = m.group(1)
        raw = unquote(m.group(2))
        if raw.startswith("http"):
            return m.group(0)  # leave external URLs untouched
        filename = Path(raw).name
        new_name = mapping.get(filename, filename.replace(" ", "_"))
        return f"![{alt}](/assets/img/{asset_root}/{slug}/{new_name})"
    return re.sub(IMAGE_RE, replace, content)


def to_figure_includes(content: str) -> str:
    """Convert markdown images to al-folio figure.liquid includes.

    Notion captions usually carry the citation as a link, e.g.
        ![[source](https://arxiv.org/abs/1234)](/assets/img/blog/x/y.png)
        ![Task distribution in [Open X-Embodiment](https://arxiv.org/abs/1234)](...)

    The first link URL in the caption becomes source=; the caption text with
    links flattened becomes alt=. The first figure loads eagerly (it is usually
    above the fold and drives LCP); the rest load lazily.
    """
    seen = [0]

    def replace(m):
        caption, path = m.group(1), m.group(2)
        path = path.lstrip("/")
        link = re.search(r"\[[^\]]*\]\((https?://[^)]+)\)", caption)
        source = link.group(1) if link else ""
        # flatten [text](url) -> text, then drop a bare leading "source" label
        alt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", caption).strip()
        alt = re.sub(r"^\(?source\)?[:\s]*", "", alt, flags=re.IGNORECASE).strip()
        alt = alt.replace('"', "'").rstrip(" .")
        if not alt:
            alt = "TODO describe this figure"
        seen[0] += 1
        loading = "eager" if seen[0] == 1 else "lazy"
        parts = [
            f'{{% include figure.liquid loading="{loading}"',
            f'path="{path}"',
            f'alt="{alt}"',
            'class="img-fluid mx-auto d-block"',
            "width=600",
        ]
        if source:
            parts.append(f'source="{source}"')
        return " ".join(parts) + " %}"

    return re.sub(IMAGE_RE, replace, content)


def check_image_refs(content: str, blog_root: Path) -> list[str]:
    """Return referenced image paths that do not exist on disk.

    Guards against the failure mode where a post ships referencing an image
    that was never copied, which 404s silently in production.
    """
    missing = []
    for path in re.findall(r'path="([^"]+)"', content):
        if not (blog_root / path).exists():
            missing.append(path)
    for path in re.findall(r"!\[[^\]]*\]\((/assets/[^)]+)\)", content):
        if not (blog_root / path.lstrip("/")).exists():
            missing.append(path)
    return missing


def fix_math(content: str) -> str:
    """Fix Notion math for kramdown/MathJax compatibility.

    1. Ensure blank lines around $$ display math blocks
    2. Remove blank lines inside $$ blocks
    3. Convert single $...$ inline math to $$...$$
    4. Replace \\rarr with \\rightarrow
    5. Escape | and || inside inline $$ math
    """
    content = content.replace("\\rarr", "\\rightarrow")

    # Pass 1: Ensure blank lines around $$ and remove blank lines inside $$ blocks
    lines = content.split("\n")
    result = []
    in_display = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "$$":
            if not in_display:
                # Opening $$: ensure blank line before
                if result and result[-1].strip() != "":
                    result.append("")
                result.append(line)
                in_display = True
            else:
                # Closing $$: ensure blank line after
                result.append(line)
                in_display = False
                if i + 1 < len(lines) and lines[i + 1].strip() != "":
                    result.append("")
        elif in_display:
            # Inside display block: skip blank lines
            if stripped:
                result.append(line)
        else:
            result.append(line)
    content = "\n".join(result)

    # Pass 2: Convert single $...$ to $$...$$ on non-display lines
    lines = content.split("\n")
    new_lines = []
    in_display = False
    in_front = True
    front_count = 0
    for line in lines:
        if in_front:
            if line.strip() == "---":
                front_count += 1
                if front_count == 2:
                    in_front = False
            new_lines.append(line)
            continue
        stripped = line.strip()
        if stripped == "$$":
            in_display = not in_display
            new_lines.append(line)
            continue
        if in_display:
            new_lines.append(line)
            continue
        # Replace inline $...$ with $$...$$
        out = []
        i = 0
        while i < len(line):
            if line[i] == "$":
                if i + 1 < len(line) and line[i + 1] == "$":
                    out.append("$$")
                    i += 2
                    end = line.find("$$", i)
                    if end != -1:
                        out.append(line[i:end])
                        out.append("$$")
                        i = end + 2
                    else:
                        out.append(line[i:])
                        i = len(line)
                else:
                    end = i + 1
                    found = False
                    while end < len(line):
                        if line[end] == "$" and (end + 1 >= len(line) or line[end + 1] != "$"):
                            inner = line[i + 1 : end]
                            if inner.strip():
                                out.append("$$")
                                out.append(inner)
                                out.append("$$")
                                i = end + 1
                                found = True
                            break
                        end += 1
                    if not found:
                        out.append(line[i])
                        i += 1
            else:
                out.append(line[i])
                i += 1
        new_lines.append("".join(out))
    content = "\n".join(new_lines)

    # Pass 3: Escape | inside inline $$ math (not display blocks)
    lines = content.split("\n")
    final_lines = []
    in_display = False
    for line in lines:
        if line.strip() == "$$":
            in_display = not in_display
            final_lines.append(line)
            continue
        if in_display:
            final_lines.append(line)
            continue
        # Process inline $$...$$ and escape pipes
        out = []
        i = 0
        while i < len(line):
            if i + 1 < len(line) and line[i] == "$" and line[i + 1] == "$":
                i += 2
                end = line.find("$$", i)
                if end != -1:
                    inner = line[i:end]
                    inner = inner.replace("||", "\\Vert")
                    inner = re.sub(r"(?<!\\)(?<!\\Vert)\|(?!\|)", r"\\vert ", inner)
                    out.append("$$" + inner + "$$")
                    i = end + 2
                else:
                    out.append("$$")
                    out.append(line[i:])
                    i = len(line)
            else:
                out.append(line[i])
                i += 1
        final_lines.append("".join(out))
    return "\n".join(final_lines)


def extract_title(content: str) -> str:
    m = re.match(r"^# (.+)", content.strip())
    return m.group(1).strip() if m else ""


def strip_h1_title(content: str) -> str:
    return re.sub(r"^# .+\n+", "", content, count=1)


def find_target_stub(slug: str, notes_dir: Path) -> Path:
    """
    Match generated slug to an existing stub file. Tries:
      1. Exact match
      2. Existing slug is a substring of generated slug (handles title prefixes like
         'introduction-to-neuroscience' inside 'introduction-to-neuroscience-bing-wen-brunton')
      3. Generated slug is a substring of existing slug
    Falls back to creating a new file.
    """
    # 1. Exact match
    exact = notes_dir / f"{slug}.md"
    if exact.exists():
        return exact

    # 2 & 3. Substring match (normalize hyphens away for comparison)
    norm = lambda s: s.replace("-", "")
    best: Path | None = None
    best_len = 0
    for stub in notes_dir.glob("*.md"):
        existing = stub.stem
        ne, ns = norm(existing), norm(slug)
        if ne in ns or ns in ne:
            if len(existing) > best_len:
                best = stub
                best_len = len(existing)
    if best:
        return best

    return notes_dir / f"{slug}.md"


def read_existing_date(stub: Path) -> str | None:
    """Read the date from an existing stub's front matter."""
    if not stub.exists():
        return None
    content = stub.read_text(encoding="utf-8")
    m = re.search(r"^date:\s*(\d{4}-\d{2}-\d{2})", content, re.MULTILINE)
    if m and m.group(1) != "2024-01-01":
        return m.group(1)
    return None


def build_front_matter(title: str, date: str) -> str:
    return f"""---
layout: post
title: "{title}"
date: {date}
toc:
  sidebar: left
---\n"""


def build_post_front_matter(title: str, img_dir: str) -> str:
    """Front matter for _posts/.

    NOTE: tags must be space-separated. A bare YAML string is split on
    whitespace, so "a, b" yields the tags "a," and "b" -- the trailing-comma
    tag then collides with the real one and clobbers its archive page.
    """
    return f"""---
layout: post
title: "{title}"
description: "TODO one or two sentences; this is the meta description and shows on /blog/"
tags: TODO space separated no commas
thumbnail: assets/img/blog/{img_dir}/TODO.png
citation: true
toc:
  sidebar: left
---\n"""


def process_post_zip(
    zip_path: Path,
    blog_root: Path,
    slug: str | None,
    img_dir_name: str | None,
    date: str | None,
) -> None:
    """Import a Notion export into _posts/ as a blog post."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        extract_nested_zip(zip_path, tmp)

        md_file = find_md_file(tmp)
        if not md_file:
            print(f"  WARNING: No .md file found in {zip_path.name}")
            return

        content = md_file.read_text(encoding="utf-8")
        title = extract_title(content)
        if not title:
            print(f"  WARNING: Could not extract title from {md_file.name}")
            return

        src_img_dir = find_image_dir(md_file)
        # Posts keep title case in the filename; the permalink is /blog/:title/
        post_slug = slug or re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-")
        img_name = img_dir_name or slugify(post_slug).replace("-", "_")
        post_date = date or infer_date(src_img_dir) or "1970-01-01"

        dest_img_dir = blog_root / "assets" / "img" / "blog" / img_name
        img_mapping = copy_images(src_img_dir, dest_img_dir) if src_img_dir else {}

        body = strip_h1_title(content)
        body = rewrite_image_refs(body, img_name, img_mapping, asset_root="blog")
        body = fix_math(body)
        body = to_figure_includes(body)
        final = build_post_front_matter(title, img_name) + body.lstrip("\n")

        target = blog_root / "_posts" / f"{post_date}-{post_slug}.md"
        target.write_text(final, encoding="utf-8")

        print(f"  ✓ {target.relative_to(blog_root)}")
        if img_mapping:
            print(f"    {len(img_mapping)} image(s) → assets/img/blog/{img_name}/")
        missing = check_image_refs(final, blog_root)
        for p in missing:
            print(f"    WARNING: referenced image not found on disk: {p}")
        print("    TODO: fill in description, tags, thumbnail; rename images descriptively")


def process_zip(zip_path: Path, blog_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        extract_nested_zip(zip_path, tmp)

        md_file = find_md_file(tmp)
        if not md_file:
            print(f"  WARNING: No .md file found in {zip_path.name}")
            return

        content = md_file.read_text(encoding="utf-8")
        title = extract_title(content)
        if not title:
            print(f"  WARNING: Could not extract title from {md_file.name}")
            return

        slug = slugify(title)
        img_dir = find_image_dir(md_file)

        # Determine date: prefer screenshot-derived, then existing stub, then fallback
        date = infer_date(img_dir)
        notes_dir = blog_root / "_notes"
        target = find_target_stub(slug, notes_dir)
        if date is None:
            date = read_existing_date(target) or "2024-01-01"

        # Copy and rename images
        dest_img_dir = blog_root / "assets" / "img" / "notes" / target.stem
        img_mapping = copy_images(img_dir, dest_img_dir) if img_dir else {}

        # Build final content
        body = strip_h1_title(content)
        body = rewrite_image_refs(body, target.stem, img_mapping)
        body = fix_math(body)
        final = build_front_matter(title, date) + body.lstrip("\n")

        target.write_text(final, encoding="utf-8")
        rel = target.relative_to(blog_root)
        print(f"  ✓ {rel}  (date: {date})")
        if img_mapping:
            print(f"    {len(img_mapping)} image(s) → assets/img/notes/{target.stem}/")


def main():
    parser = argparse.ArgumentParser(
        description="Import Notion Markdown exports into Jekyll _notes/"
    )
    parser.add_argument("input", help="Zip file or directory of zip files")
    parser.add_argument(
        "--blog-root",
        default=str(Path(__file__).parent.parent),
        help="Path to Jekyll blog root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--as",
        dest="kind",
        choices=["note", "post"],
        default="note",
        help="Import into _notes/ (default) or _posts/",
    )
    parser.add_argument(
        "--slug",
        help="post only: filename slug after the date, e.g. Foundation-Models-for-Robotics-VLA",
    )
    parser.add_argument(
        "--img-dir",
        help="post only: short name under assets/img/blog/, e.g. vla",
    )
    parser.add_argument("--date", help="post only: YYYY-MM-DD, overrides inference")
    args = parser.parse_args()

    blog_root = Path(args.blog_root).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()

    required = "_posts" if args.kind == "post" else "_notes"
    if not (blog_root / required).exists():
        print(f"ERROR: No {required}/ directory found at {blog_root}")
        sys.exit(1)

    zips: list[Path] = []
    if input_path.is_dir():
        zips = sorted(input_path.glob("*.zip"))
    elif input_path.suffix == ".zip":
        zips = [input_path]
    else:
        print(f"ERROR: {input_path} is neither a zip file nor a directory")
        sys.exit(1)

    if not zips:
        print(f"No zip files found in {input_path}")
        sys.exit(1)

    if args.kind == "post" and len(zips) > 1 and (args.slug or args.img_dir):
        print("ERROR: --slug/--img-dir apply to a single export; pass one zip")
        sys.exit(1)

    print(f"Processing {len(zips)} zip(s) → {blog_root}\n")
    for z in zips:
        if args.kind == "post":
            process_post_zip(z, blog_root, args.slug, args.img_dir, args.date)
        else:
            process_zip(z, blog_root)
    print("\nDone!")


if __name__ == "__main__":
    main()
