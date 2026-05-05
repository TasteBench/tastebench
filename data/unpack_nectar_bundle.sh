#!/usr/bin/env bash
# Move the downloaded NECTAR Kaggle bundle into the on-disk paths the
# food-similarity pipeline expects. Run from the project root
# (the `tastebench/` directory after unzipping the supplementary).
#
# Usage:
#     bash data/unpack_nectar_bundle.sh <path-to-extracted-bundle>
#
# Where <path-to-extracted-bundle> is the directory you get after
# extracting the Kaggle Dataset zip
# (e.g., ~/Downloads/tastebench-nectar-bundle/).
#
# The bundle layout is the Kaggle-conventional flat form:
#     <bundle>/
#       nectar_consolidated_ingredients_nutrition.csv
#       nectar_consolidated_sensory_rating.csv
#       nectar_product_labels.csv
#       product_labels_manually_cleaned.csv
#       images.zip       (cropped/<year>/<category>/<code>/<view>.jpg)
#       taste_like.zip   (Taste Like CPG product directory CSVs)
#
# This script moves each artifact to its target path under the project root.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: bash data/unpack_nectar_bundle.sh <path-to-extracted-bundle>" >&2
    exit 1
fi

SRC="$(cd "$1" && pwd)"
NEURIPS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Sanity-check the expected files are all present in the source.
# The four NECTAR CSVs are always required as files.
EXPECTED_CSVS=(
    "nectar_consolidated_ingredients_nutrition.csv"
    "nectar_consolidated_sensory_rating.csv"
    "nectar_product_labels.csv"
    "product_labels_manually_cleaned.csv"
)
for f in "${EXPECTED_CSVS[@]}"; do
    if [ ! -f "${SRC}/${f}" ]; then
        echo "ERROR: missing ${f} in ${SRC}" >&2
        exit 1
    fi
done

# images and taste_like may arrive either as .zip files (Kaggle's
# native bundle layout) or as pre-extracted directories (some Kaggle
# download paths auto-extract inner zips). Accept either form.
if [ ! -f "${SRC}/images.zip" ] && [ ! -d "${SRC}/images" ]; then
    echo "ERROR: missing both images.zip and images/ in ${SRC}" >&2
    exit 1
fi
if [ ! -f "${SRC}/taste_like.zip" ] && [ ! -d "${SRC}/taste_like" ]; then
    echo "ERROR: missing both taste_like.zip and taste_like/ in ${SRC}" >&2
    exit 1
fi

# Make target directories (most exist already, but be safe).
mkdir -p "${NEURIPS_DIR}/data/consolidated_datasets"
mkdir -p "${NEURIPS_DIR}/shared/data"
mkdir -p "${NEURIPS_DIR}/food_similarity/zero_shot_baselines/data"
mkdir -p "${NEURIPS_DIR}/data/product_images"
mkdir -p "${NEURIPS_DIR}/data/taste_like"

echo "[1/6] consolidated_datasets/ CSVs..."
cp "${SRC}/nectar_consolidated_ingredients_nutrition.csv" "${NEURIPS_DIR}/data/consolidated_datasets/"
cp "${SRC}/nectar_consolidated_sensory_rating.csv"        "${NEURIPS_DIR}/data/consolidated_datasets/"

echo "[2/6] shared/data/nectar_product_labels.csv..."
cp "${SRC}/nectar_product_labels.csv" "${NEURIPS_DIR}/shared/data/"

echo "[3/6] food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv..."
cp "${SRC}/product_labels_manually_cleaned.csv" "${NEURIPS_DIR}/food_similarity/zero_shot_baselines/data/"

echo "[4/6] data/product_images/cropped/ from images..."
if [ -f "${SRC}/images.zip" ]; then
    unzip -o -q "${SRC}/images.zip" -d "${NEURIPS_DIR}/data/product_images/"
else
    cp -R "${SRC}/images/." "${NEURIPS_DIR}/data/product_images/"
fi

echo "[5/6] data/taste_like/ from taste_like..."
if [ -f "${SRC}/taste_like.zip" ]; then
    unzip -o -q "${SRC}/taste_like.zip" -d "${NEURIPS_DIR}/data/taste_like/"
else
    cp -R "${SRC}/taste_like/." "${NEURIPS_DIR}/data/taste_like/"
fi

echo "[6/6] verifying placements..."
for path in \
    "data/consolidated_datasets/nectar_consolidated_ingredients_nutrition.csv" \
    "data/consolidated_datasets/nectar_consolidated_sensory_rating.csv" \
    "shared/data/nectar_product_labels.csv" \
    "food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv"
do
    [ -f "${NEURIPS_DIR}/${path}" ] || { echo "ERROR: missing after copy: ${path}" >&2; exit 1; }
done
N_IMG=$(find "${NEURIPS_DIR}/data/product_images/cropped" -name '*.jpg' | wc -l | tr -d ' ')
[ "${N_IMG}" -gt 0 ] || { echo "ERROR: no JPGs found under data/product_images/cropped/" >&2; exit 1; }
N_TL=$(find "${NEURIPS_DIR}/data/taste_like" -name '*.csv' | wc -l | tr -d ' ')
[ "${N_TL}" -gt 0 ] || { echo "ERROR: no Taste Like CSVs found under data/taste_like/" >&2; exit 1; }

echo
echo "OK. Bundle unpacked into ${NEURIPS_DIR}/. Found ${N_IMG} cropped JPGs and ${N_TL} Taste Like CSVs."
echo "Tier 1 retraining is now runnable: see README.md."
