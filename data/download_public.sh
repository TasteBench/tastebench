#!/usr/bin/env bash
# Fetch the publicly hosted datasets that the TasteBench pipeline depends on.
#
# - FooDB 2020-04-07  (https://foodb.ca/downloads)            → data/foodb_2020_04_07_csv/
# - FoodAtlas v3.2    is retired upstream — see notes at bottom of this file.
#                                                            → data/food_atlas/v3.2_20250211/
#
# Idempotent: re-running skips datasets whose target directory already exists
# and is non-empty.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Verified 2026-05-02: 200 OK; downloaded body sha256 captured below.
FOODB_URL="https://foodb.ca/public/system/downloads/foodb_2020_4_7_csv.tar.gz"
FOODB_SHA256="ed56abca084bce5cf7d7bb06fb037ca4b136f0d157ff4efdeeb227c43f024969"
FOODB_TGT="${HERE}/foodb_2020_04_07_csv"

FOODATLAS_URL="https://foodatlasdownloadsstack-downloadsbucketb54b8c20-hfoegjgwag4w.s3.us-west-1.amazonaws.com/bundles/foodatlas-v4.0/foodatlas-v4.0.zip"
FOODATLAS_SHA256="9418656937738bd42370ff3898d8aadbd0f57c013d5707fcc49aaba4262f7b0b"
FOODATLAS_TGT="${HERE}/food_atlas/v4.0"

verify_nonempty() {
  local d="$1"
  [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]
}

fetch() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Error: need curl or wget on PATH." >&2
    return 1
  fi
}

# ---------- FooDB ----------
if verify_nonempty "$FOODB_TGT"; then
  echo "[skip] FooDB: ${FOODB_TGT} already populated."
else
  echo "[fetch] FooDB → ${FOODB_TGT}  (~950 MB; the upstream file is named"
  echo "        .tar.gz but is actually uncompressed tar — verified 2026-05-02)"
  mkdir -p "$FOODB_TGT"
  tmp_tgz="$(mktemp -t foodb.XXXXXX.tar)"
  trap 'rm -f "$tmp_tgz"' EXIT
  if fetch "$FOODB_URL" "$tmp_tgz"; then
    # Integrity check against the canonical 2026-05-02 body.
    actual_sha=$(shasum -a 256 "$tmp_tgz" 2>/dev/null | awk '{print $1}')
    if [ -z "$actual_sha" ]; then
      actual_sha=$(sha256sum "$tmp_tgz" 2>/dev/null | awk '{print $1}')
    fi
    if [ "$actual_sha" != "$FOODB_SHA256" ]; then
      echo "[FAIL] FooDB SHA256 mismatch:" >&2
      echo "  expected: $FOODB_SHA256" >&2
      echo "  got:      $actual_sha" >&2
      echo "  upstream may have changed; verify before proceeding." >&2
      rm -f "$tmp_tgz"
      rmdir "$FOODB_TGT" 2>/dev/null || true
      exit 1
    fi
    # -xf (no -z): auto-detect compression. The upstream file is plain tar
    # despite its .tar.gz name, and GNU tar's -z would refuse it.
    tar -xf "$tmp_tgz" --strip-components=1 -C "$FOODB_TGT"
    echo "[done] FooDB extracted to $FOODB_TGT"
  else
    echo "[FAIL] FooDB download from $FOODB_URL" >&2
    rmdir "$FOODB_TGT" 2>/dev/null || true
    exit 1
  fi
  trap - EXIT
  rm -f "$tmp_tgz"
fi

# ---------- FoodAtlas v4.0 ----------
if verify_nonempty "$FOODATLAS_TGT"; then
  echo "[skip] FoodAtlas v4.0: ${FOODATLAS_TGT} already populated."
else
  echo "[fetch] FoodAtlas v4.0 → ${FOODATLAS_TGT}  (~82 MB)"
  mkdir -p "$FOODATLAS_TGT"
  tmp_zip="$(mktemp -t foodatlas.XXXXXX.zip)"
  trap 'rm -f "$tmp_zip"' EXIT
  if fetch "$FOODATLAS_URL" "$tmp_zip"; then
    actual_sha=$(shasum -a 256 "$tmp_zip" 2>/dev/null | awk '{print $1}')
    if [ -z "$actual_sha" ]; then
      actual_sha=$(sha256sum "$tmp_zip" 2>/dev/null | awk '{print $1}')
    fi
    if [ "$actual_sha" != "$FOODATLAS_SHA256" ]; then
      echo "[FAIL] FoodAtlas SHA256 mismatch:" >&2
      echo "  expected: $FOODATLAS_SHA256" >&2
      echo "  got:      $actual_sha" >&2
      rm -f "$tmp_zip"
      rmdir "$FOODATLAS_TGT" 2>/dev/null || true
      exit 1
    fi
    unzip -q "$tmp_zip" -d "$FOODATLAS_TGT.tmp"
    # zip extracts under foodatlas-v4.0/; flatten one level.
    mv "$FOODATLAS_TGT.tmp/foodatlas-v4.0/"* "$FOODATLAS_TGT/"
    rm -rf "$FOODATLAS_TGT.tmp"
    echo "[done] FoodAtlas v4.0 extracted to $FOODATLAS_TGT"
  else
    echo "[FAIL] FoodAtlas download from $FOODATLAS_URL" >&2
    rmdir "$FOODATLAS_TGT" 2>/dev/null || true
    exit 1
  fi
  trap - EXIT
  rm -f "$tmp_zip"
fi

# Note: the original paper experiments used FoodAtlas v3.2_20250211, which
# was retired upstream after our experiments. v4.0 produces statistically
# equivalent BT+Gemini NNLS pairwise accuracy (.6829 vs .6818 on v3.2; CIs
# overlap heavily). See supervised/train/compute_per_model_nnls.py for the
# adapter that handles the schema diff (parquet vs tsv, attestations vs
# metadata_contains, derived synonyms vs lookup_table_*).

cat <<'EOF'

[OK] Public datasets in place. Next steps:
  - Gated NECTAR data: see data/GATED.md
  - Derived caches:    see README.md (Caches section)
  - Run pipeline:      bash food_similarity/reproduce.sh
EOF
