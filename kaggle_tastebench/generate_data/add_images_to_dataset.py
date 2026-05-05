"""
Add images to the Kaggle competition dataset.

Copies real NECTAR images (renamed to new product codes) and generates
augmented duplicates for Taste Like products to maintain anonymity.
"""

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# Maps Kaggle category -> (cropped_dir_name, year, primary_view_type, fallback_view_types)
CATEGORY_CONFIG = {
    "Bacon": ("Bacon", 2025, "cooked_top", []),
    "Barista_Milk": ("Barista_Milk", 2026, "top_view", []),
    "Bratwurst": ("Bratwurst", 2025, "cooked_front", []),
    "Breaded_Chicken_Filet": ("Breaded_Chicken_Filet", 2025, "cooked_front", []),
    "Breakfast_Sausages": ("Breakfast_Sausages", 2025, "cooked_front", []),
    "Burgers": ("Burgers", 2025, "cooked_front", []),
    "Butter": ("Salted_Butter", 2026, "butter_only", []),
    "Cheddar_Cheese": ("Cheddar_Cheese_Slices", 2026, "cross_section", []),
    "Chicken_Strips": ("Unbreaded_chicken_strips_and_chunks", 2025, "cooked_top", []),
    "Cream_Cheese": ("Cream_Cheese", 2026, "top_view", []),
    "Creamer": ("Creamer", 2026, "creamer_only", []),
    "Deli_Ham": ("Deli_Ham", 2025, "uncooked_top", []),
    "Deli_Turkey": ("Deli_Turkey", 2025, "uncooked_top", []),
    "Hot_Dogs": ("Hot_Dogs", 2025, "cooked_front", []),
    "Ice_Cream_Hard_Serve": ("Ice_Cream_Hard_Serve", 2026, "top_view", []),
    "Meatballs": ("Meatballs", 2025, "cross_section_top", []),
    "Milk": ("Milk", 2026, "top_view", []),
    "Mozzarella": ("Mozzarella_Cheese", 2026, "top_view", []),
    "Nuggets": ("Nuggets", 2025, "cross_section_front", []),
    "Pulled_Pork": ("Pulled_Pork", 2025, "cooked_front", []),
    "Sour_Cream": ("Sour_Cream", 2026, "top_view", []),
    "Steak": ("Steak", 2025, "cooked_front", []),
    "Unbreaded_Chicken_Breast": ("Unbreaded_Chicken_Breast", 2025, "cooked_top", ["cooked_front"]),
    "Yogurt": ("Plain_Greek_Yogurt", 2026, "top_view", []),
}

EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_image(cropped_base: Path, year: int, dir_name: str, product_code: int,
               view_type: str, fallbacks: list[str]) -> Path | None:
    """Find the image file for a given product, handling extension and suffix variants."""
    product_dir = cropped_base / str(year) / dir_name / str(product_code)
    if not product_dir.is_dir():
        return None

    for vt in [view_type] + fallbacks:
        # Try exact match first
        for ext in EXTENSIONS:
            candidate = product_dir / f"{vt}{ext}"
            if candidate.exists():
                return candidate

        # Try numbered suffix (_1, _2, etc.) but exclude _with_ variants
        matches = sorted(
            p for p in product_dir.iterdir()
            if p.stem.startswith(vt + "_")
            and "_with_" not in p.stem
            and p.suffix.lower() in EXTENSIONS
            and p.stem[len(vt) + 1:].isdigit()
        )
        if matches:
            return matches[0]

    return None


def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply subtle, natural-looking augmentations that mimic camera/photography variation."""
    import numpy as np

    # Horizontal flip — 50/50 chance (not always-on, to avoid flip as a classification signal)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Moderate random crop (8-18%) — changes framing meaningfully
    w, h = img.size
    crop_frac = rng.uniform(0.08, 0.18)
    new_w = int(w * (1 - crop_frac))
    new_h = int(h * (1 - crop_frac))
    left = rng.randint(0, w - new_w)
    top = rng.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BICUBIC)

    # Color temperature shift (warm/cool on red-blue axis)
    arr = np.array(img, dtype=np.float32)
    temp_shift = rng.uniform(-15, 15)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + temp_shift, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - temp_shift * 0.5, 0, 255)

    # White balance shift (green-magenta axis)
    green_shift = rng.uniform(-10, 10)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + green_shift, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # Gamma adjustment — simulates different camera exposure curves
    gamma = rng.uniform(0.85, 1.15)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.clip(np.power(arr, gamma), 0, 1) * 255.0
    img = Image.fromarray(arr.astype(np.uint8))

    # Brightness jitter
    factor = rng.uniform(0.82, 1.18)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast jitter
    factor = rng.uniform(0.82, 1.18)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation jitter
    factor = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Color(img).enhance(factor)

    # Subtle Gaussian noise — simulates different camera sensor characteristics
    arr = np.array(img, dtype=np.float32)
    noise_sigma = rng.uniform(2, 5)
    noise = np.array([rng.gauss(0, noise_sigma) for _ in range(arr.size)],
                     dtype=np.float32).reshape(arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # Slight blur to mask compression artifacts
    radius = rng.uniform(0.2, 0.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def main():
    parser = argparse.ArgumentParser(description="Add images to Kaggle competition dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    cropped_base = repo_root / "data" / "product_images" / "cropped"
    dataset_dir = script_dir / "dataset"
    output_dir = dataset_dir / "images"

    # Clear output directory to remove stale images from previous runs
    if not args.dry_run:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    # Load product code map
    code_map = pd.read_csv(dataset_dir / "product_code_map.csv")
    nectar = code_map[code_map["Source"] == "nectar"]
    taste_like = code_map[code_map["Source"] == "taste_like"]

    # Track results
    nectar_copied = 0
    nectar_missing = []
    taste_like_generated = 0
    taste_like_skipped = 0
    nectar_images_by_category: dict[str, list[Path]] = defaultdict(list)
    taste_like_generated_by_category: dict[str, int] = defaultdict(int)

    # Step 1: Process NECTAR products
    for _, row in nectar.iterrows():
        category = row["Category"]
        orig_code = row["Original_Product_Code"]
        new_code = row["New_Product_Code"]

        if category not in CATEGORY_CONFIG:
            continue

        dir_name, year, view_type, fallbacks = CATEGORY_CONFIG[category]
        src_path = find_image(cropped_base, year, dir_name, orig_code, view_type, fallbacks)

        out_category_dir = output_dir / category
        out_path = out_category_dir / f"{new_code}.jpg"

        if src_path is None:
            nectar_missing.append((category, orig_code, new_code))
            continue

        if args.dry_run:
            print(f"[COPY] {src_path} -> {out_path}")
        else:
            out_category_dir.mkdir(parents=True, exist_ok=True)
            # Copy JPEG files directly to preserve exact image data;
            # convert non-JPEG (e.g. PNG) to JPEG at high quality
            if str(src_path).lower().endswith((".jpg", ".jpeg")):
                shutil.copy2(src_path, out_path)
            else:
                img = Image.open(src_path).convert("RGB")
                img.save(out_path, "JPEG", quality=95)

        nectar_images_by_category[category].append(src_path)
        nectar_copied += 1

    # Step 2: Process Taste Like products (augmented duplicates)
    for _, row in taste_like.iterrows():
        category = row["Category"]
        new_code = row["New_Product_Code"]

        if category not in CATEGORY_CONFIG:
            continue

        available_sources = nectar_images_by_category.get(category, [])
        if not available_sources:
            taste_like_skipped += 1
            continue

        # Deterministic RNG per product (md5 avoids Python's per-session hash randomization)
        seed_str = f"{category}_{new_code}"
        det_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(args.seed + det_hash)
        source_img_path = rng.choice(available_sources)

        out_category_dir = output_dir / category
        out_path = out_category_dir / f"{new_code}.jpg"

        if args.dry_run:
            print(f"[AUGMENT] {source_img_path} -> {out_path}")
        else:
            out_category_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(source_img_path).convert("RGB")
            img = augment_image(img, rng)
            jpeg_quality = rng.randint(75, 92)
            img.save(out_path, "JPEG", quality=jpeg_quality)

        taste_like_generated += 1
        taste_like_generated_by_category[category] += 1

    # Step 3: Drop some Taste Like images to match NECTAR missing rate per category
    # (prevents "missing image = definitely NECTAR" as an anonymity leak)
    nectar_missing_by_cat: dict[str, int] = defaultdict(int)
    nectar_total_by_cat: dict[str, int] = defaultdict(int)
    for cat, _, _ in nectar_missing:
        nectar_missing_by_cat[cat] += 1
    for _, row in nectar.iterrows():
        if row["Category"] in CATEGORY_CONFIG:
            nectar_total_by_cat[row["Category"]] += 1

    taste_like_dropped = 0
    for cat, n_missing in nectar_missing_by_cat.items():
        n_nectar_total = nectar_total_by_cat[cat]
        missing_rate = n_missing / n_nectar_total
        n_taste_in_cat = len(taste_like[taste_like["Category"] == cat])
        n_to_drop = max(1, round(missing_rate * n_taste_in_cat))

        # Deterministically select which Taste Like products to drop
        taste_codes = sorted(
            taste_like[taste_like["Category"] == cat]["New_Product_Code"].tolist()
        )
        drop_seed = f"drop_{cat}"
        drop_hash = int(hashlib.md5(drop_seed.encode()).hexdigest(), 16) % (2**31)
        drop_rng = random.Random(args.seed + drop_hash)
        to_drop = drop_rng.sample(taste_codes, min(n_to_drop, len(taste_codes)))

        for code in to_drop:
            img_path = output_dir / cat / f"{code}.jpg"
            if img_path.exists():
                if args.dry_run:
                    print(f"[DROP] {img_path}")
                else:
                    img_path.unlink()
                taste_like_dropped += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"NECTAR images copied:       {nectar_copied}")
    print(f"Taste Like images generated: {taste_like_generated}")
    print(f"Taste Like images dropped:  {taste_like_dropped}")
    print(f"Taste Like skipped (no source): {taste_like_skipped}")
    print(f"NECTAR missing images:      {len(nectar_missing)}")

    if nectar_missing:
        print(f"\nMissing NECTAR images:")
        for cat, orig, new in nectar_missing:
            print(f"  {cat}: original={orig}, new={new}")

    # Per-category summary
    all_categories = sorted(set(code_map["Category"].unique()) & set(CATEGORY_CONFIG.keys()))
    print(f"\nPer-category breakdown:")
    for cat in all_categories:
        n_nectar = len(nectar_images_by_category.get(cat, []))
        n_taste = len(taste_like[(taste_like["Category"] == cat)])
        n_missing = len([m for m in nectar_missing if m[0] == cat])
        total_nectar_in_cat = len(nectar[nectar["Category"] == cat])
        n_taste_gen = taste_like_generated_by_category.get(cat, 0)
        print(f"  {cat}: {n_nectar}/{total_nectar_in_cat} NECTAR copied, "
              f"{n_taste_gen}/{n_taste} Taste Like generated, "
              f"{n_missing} missing")

    # Step 4: Update products.csv with image_path column
    products_path = dataset_dir / "products.csv"
    products = pd.read_csv(products_path)

    def get_image_path(row):
        img_file = output_dir / row["Category"] / f"{row['Product code']}.jpg"
        if img_file.exists():
            return f"images/{row['Category']}/{row['Product code']}.jpg"
        return ""

    if not args.dry_run:
        products["image_path"] = products.apply(get_image_path, axis=1)
        products.to_csv(products_path, index=False, na_rep="")
        n_with_images = (products["image_path"] != "").sum()
        print(f"\nUpdated products.csv: {n_with_images}/{len(products)} products have image_path")
    else:
        print(f"\n[DRY RUN] Would update products.csv with image_path column")


if __name__ == "__main__":
    main()
