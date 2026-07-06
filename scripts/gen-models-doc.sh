#!/usr/bin/env bash
# Generate docs/models.md from registry.toml — the single source of truth.
# No TOML dependency: the schema is flat [[model]] blocks, parsed with awk.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

{
  echo "# Supported models"
  echo
  echo "Generated from \`registry.toml\` — do not edit by hand"
  echo "(\`scripts/gen-models-doc.sh\`). Every row's metadata is re-preflighted"
  echo "nightly."
  echo
  echo "**Certification:** *verified* = full \`cima vet\` on hardware;"
  echo "*verified-by-family* = metadata preflight clean and a smaller family"
  echo "member is verified; *avoid* = known-defective, listed as a warning."
  echo
  echo "| Model | Family | Format | Capabilities | Min VRAM | Status | Vetted |"
  echo "|---|---|---|---|---|---|---|"
  awk '
    /^\[\[model\]\]/ { if (repo) print_row(); repo=fam=fmt=caps=vram=status=date="" ; next }
    /^\s*repo/       { repo=val() }
    /^\s*family/     { fam=val() }
    /^\s*format/     { fmt=val() }
    /^\s*caps/       { caps=arr() }
    /^\s*min_vram/   { vram=rawval() }
    /^\s*status/     { status=val() }
    /^\s*vet_date/   { date=val() }
    END { if (repo) print_row() }
    function val(){ s=$0; sub(/^[^=]*=[ ]*"/,"",s); sub(/".*/,"",s); return s }
    function rawval(){ s=$0; sub(/^[^=]*=[ ]*/,"",s); gsub(/[ ]/,"",s); return s }
    function arr(){ s=$0; sub(/^[^=]*=[ ]*\[/,"",s); sub(/\].*/,"",s); gsub(/"/,"",s); gsub(/,/,", ",s); return s }
    function print_row(){ printf "| `%s` | %s | %s | %s | %s GiB | %s | %s |\n", repo,fam,fmt,caps,vram,status,date }
  ' registry.toml
} > docs/models.md
echo "wrote docs/models.md"