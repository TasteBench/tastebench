"""FoodAtlas knowledge graph loader and ingredient-to-compound mapper.

Maps ingredient names from a NECTAR product label to a compound set
via the FoodAtlas knowledge graph. Two resolution paths per ingredient:

  Food path:     name → food entity → "contains" triplets → compounds
  Chemical path: name → chemical entity (i.e. the ingredient IS itself
                 a single compound, e.g. "salt" or "ascorbic acid")

The loader auto-detects FoodAtlas v3.2 (.tsv) or v4.0 (.parquet) from
the bundle directory; v4.0 is the canonical version for this paper.

References:
- FoodAtlas paper: https://pubmed.ncbi.nlm.nih.gov/39216404/
- FoodAtlas-KGv2:  https://github.com/IBPA/FoodAtlas-KGv2
"""

import json
import logging
import re
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Modifiers to strip during tier-2 normalization
INGREDIENT_MODIFIERS = [
    "Isolate", "Concentrate", "Extract", "Powder", "Flour", "Starch",
    "Modified", "Hydrolyzed", "Textured", "Vital", "Refined", "Enriched",
    "Expeller Pressed", "Expeller-Pressed", "Cold Pressed", "Virgin",
    "Extra Virgin", "High Oleic", "Low Erucic Acid", "Hydrogenated",
    "Defatted", "Dried", "Fermented", "Autolyzed", "Inactive",
    "Filtered", "Purified", "Distilled", "Rehydrated", "Toasted",
    "Roasted", "Ground", "Whole Grain", "Whole", "Raw",
]

# Manual synonym mapping: our ingredient name (lowercase) → FoodAtlas name (lowercase).
# Verified against lookup_table_food.tsv and lookup_table_chemical.tsv.
# fmt: off
INGREDIENT_SYNONYMS: Dict[str, str] = {
    # === Salts ===
    "sea salt": "salt",
    "himalayan salt": "salt",
    "himalayan pink salt": "salt",
    "himalyan pink salt": "salt",
    "sundried salt": "salt",
    "sundried sea salt": "salt",
    "iodized salt": "salt",
    "kosher salt": "salt",
    "pure salt": "salt",
    "french grey salt": "salt",
    "reduced sodium salt": "salt",
    "smoked salt": "salt",

    # === Sugars ===
    "sugar": "sucrose",
    "brown sugar": "sucrose",
    "cane sugar": "cane sugar",
    "raw cane sugar": "cane sugar",
    "organic cane sugar": "cane sugar",
    "turbinado sugar": "sucrose",
    "demerara sugar": "sucrose",
    "dark brown sugar": "sucrose",
    "beet sugar": "sucrose",
    "coconut sugar": "sucrose",
    "smoked sugar": "sucrose",
    "hardwood smoked sugar": "sucrose",
    "harwood smoked sugar": "sucrose",
    "caramelized sugar": "sucrose",
    "caramelised sugar": "sucrose",
    "vegan cane sugar": "cane sugar",
    "sugar cane": "cane sugar",

    # === Water ===
    "filtered water": "water",
    "purified water": "water",
    "spring water": "water",
    "cooked drinking water": "water",
    "blue ridge mountain well water": "water",

    # === Soy / Soybean ===
    # Protein isolates/concentrates → defatted flour (42 comp, all with conc)
    "soy protein concentrate": "soybean flour (defatted)",
    "soy protein isolate": "soybean flour (defatted)",
    "soy protein": "soybean flour (defatted)",
    "soya protein": "soybean flour (defatted)",
    "soya protein isolate": "soybean flour (defatted)",
    "soya protein concentrate": "soybean flour (defatted)",
    "soy protein-concentrate": "soybean flour (defatted)",
    "soy protein conc": "soybean flour (defatted)",
    "isolated soy protein": "soybean flour (defatted)",
    "textured soy protein concentrate": "soybean flour (defatted)",
    "textured soy protein": "soybean flour (defatted)",
    "textured soybean protein": "soybean flour (defatted)",
    "rehydrated soya proteins": "soybean flour (defatted)",
    "rehydrated soy protein": "soybean flour (defatted)",
    "hydrolyzed soy protein": "soybean flour (defatted)",
    # Generic soy → full soybean (390 compounds)
    "soy": "soybean",
    "soya": "soybean",
    "soy flour": "soybean flour",
    "soy lecithin": "soybean oil",
    "soya bean oil": "soybean oil",
    "soy bean oil": "soybean oil",
    "soy leghemoglobin": "soybean",

    # === Wheat — per-fraction routing (food-science-aware) ===
    # Wheat gluten / protein: lives in the ENDOSPERM, not bran or germ.
    # Route to a synthetic target whose ENTRY_PAIR uses kernel + refined
    # endosperm flours (where gluten resides) and excludes bran/germ.
    "wheat gluten": "wheat (gluten/protein fraction)",
    "vital wheat gluten": "wheat (gluten/protein fraction)",
    "textured wheat protein": "wheat (gluten/protein fraction)",
    "wheat protein concentrate": "wheat (gluten/protein fraction)",
    "wheat protein isolate": "wheat (gluten/protein fraction)",
    "hydrolyzed wheat protein": "wheat (gluten/protein fraction)",
    "1wheat protein": "wheat (gluten/protein fraction)",
    "wheat gluten and wheat starch": "wheat (gluten/protein fraction)",
    "gluten": "wheat (gluten/protein fraction)",
    "textured wheat gluten": "wheat (gluten/protein fraction)",
    # Wheat starch: pure starch from refined endosperm (~99% starch, no
    # bran/germ). Route to refined-endosperm-flour primaries.
    "wheat starch": "wheat (starch fraction)",
    "modified wheat starch": "wheat (starch fraction)",
    # Wheat flour (refined): mostly endosperm, may have minor bran.
    "wheat flour": "wheat (refined flour)",
    "enriched wheat flour": "wheat (refined flour)",

    # === Pea ===
    "pea protein isolate": "pea",
    "pea protein concentrate": "pea",
    "pea protein concentrates": "pea",
    "textured pea protein": "pea",
    "rehydrated pea protein": "pea",
    "yellow pea protein": "pea",
    "pea fiber": "pea",
    "pea fibers": "pea",
    "pea hull fiber": "pea",

    # === Rice ===
    "brown rice protein": "rice",
    "hydrolyzed rice protein": "rice",
    "rice fiber": "rice",

    # === Oat ===
    "oat fiber": "oat",

    # === Corn / Maize (maize (corn) food product has 259 compounds — best entry) ===
    # === Corn — per-fraction routing (food-science-aware) ===
    # Corn starch / modified cornstarch / maltodextrin: pure starch from
    # endosperm. Route to corn-flour-class primaries (closest in chemistry).
    "corn starch": "corn (starch fraction)",
    "cornstarch": "corn (starch fraction)",
    "modified corn starch": "corn (starch fraction)",
    "modified cornstarch": "corn (starch fraction)",
    "corn maltodextrin": "corn (starch fraction)",
    # Corn syrups (HFCS, glucose syrup): chemically just sugar, NOT corn-
    # flavor compounds. Route to dextrose chemical.
    "corn syrup": "dextrose",
    "corn syrup solids": "dextrose",
    "high fructose corn syrup": "dextrose",
    # Whole corn / flour / protein: route to whole-grain corn primaries.
    "corn": "maize (corn) food product",
    "corn flour": "corn (whole grain or flour)",
    "yellow corn flour": "corn (whole grain or flour)",
    "corn powder": "corn (whole grain or flour)",
    "soluble corn fiber": "corn (whole grain or flour)",
    # Corn oil: lipid extract — separate fraction.
    "corn oil": "corn (oil fraction)",

    # === Hemp (hemp oil has 20 compounds — best entry) ===
    "hemp protein": "hemp oil",
    "hemp protein powder": "hemp oil",
    "hemp seed": "hemp oil",
    "hemp seeds": "hemp oil",
    "hulled hemp seed": "hemp oil",

    # === Fava / Faba bean ===
    "fava bean protein": "faba bean",
    "faba bean protein isolate": "faba bean",
    "faba bean protein concentrate": "faba bean",
    "faba bean protein powder": "faba bean",
    "faba protein": "faba bean",

    # === Coconut ===
    "coconut fat": "coconut oil",
    "coconut water": "coconut",
    "coconut sugar": "sucrose",

    # === Sunflower (sunflower oil 126 comp, seed 55) ===
    "sunflower lecithin": "sunflower oil",
    "sunflower": "sunflower seed",
    "sunflower protein": "sunflower seed",

    # === Palm ===
    "palm": "palm oil",
    "palm kernel": "palm oil",
    "non-hydrogenated palm oil": "palm oil",
    "expeller pressed palm fruit oil": "palm oil",
    "palm fruit": "palm oil",

    # === Other oils (shea butter has no compounds in FoodAtlas) ===
    "shea oil": "cocoa butter",
    "shea": "cocoa butter",
    "shea fat": "cocoa butter",

    # === Vinegar (acetic acid is the key compound - use chemical path) ===
    "vinegar": "acetic acid",
    "apple cider vinegar": "acetic acid",
    "distilled vinegar": "acetic acid",
    "spirit vinegar": "acetic acid",
    "white vinegar": "acetic acid",
    "distilled white vinegar": "acetic acid",
    "red wine vinegar": "acetic acid",
    "balsamic vinegar": "acetic acid",
    "rice vinegar": "acetic acid",
    "apple vinegar": "acetic acid",
    "buffered vinegar": "acetic acid",
    "dried vinegar": "acetic acid",

    # === Spices & herbs ===
    "paprika": "pepper",  # pepper has 184 compounds — best for capsicum family
    "smoked paprika": "pepper",
    "apple wood smoked paprika": "pepper",
    "paprika extract": "pepper",
    "extractives of paprika": "pepper",
    "paprika extractives": "pepper",
    "oleoresin paprika": "pepper",
    "paprika oleoresin": "pepper",
    "paprika extract color": "pepper",
    "paprika color": "pepper",
    "mustard": "brown mustard",
    "dry mustard": "brown mustard",
    "fenugreek": "fenugreek seed",
    "seaweed": "kombu",  # kombu has 28 compounds
    "herbs": "thyme",  # thyme has 123 compounds
    "aromatic herbs": "thyme",
    "celery seed": "celery",  # celery has 143 compounds; celery seed has 0
    "celery seed oil": "celery",

    # === Gums & thickeners ===
    "xanthan gum": "xanthan",
    "xanthan & guar gums": "xanthan",
    "guar and xanthan gums": "xanthan",
    "locust bean gum": "locust bean",
    "carob bean gum": "locust bean",
    "guar bean gum": "guar gum",
    "konjac": "amorphophallus konjac",
    "konjac gum": "amorphophallus konjac",
    "konjac powder": "amorphophallus konjac",
    "konjac root": "amorphophallus konjac",
    "curdlan gum": "curdlan",
    "agar": "pectin",  # agar-agar has 0 compounds; pectin is similar polysaccharide
    "agar agar": "pectin",
    "tara gum": "locust bean",  # tara not in FoodAtlas; locust bean similar galactomannan
    "cellulose gum": "cellulose",
    "modified cellulose": "cellulose",
    "modified cellulose gum": "cellulose",
    "cellulose fiber": "cellulose",
    "powdered cellulose": "cellulose",

    # === Colors (chemical path for colorants) ===
    "caramel color": "caramel furanone",
    "annatto": "bixin",  # bixin is the primary pigment in annatto
    "annatto extract": "bixin",
    "anatto extract": "bixin",
    "annatto color": "bixin",
    "annatto for color": "bixin",
    "beetroot red": "betanin",
    "beta carotene": "beta-carotene",
    "betacarotene": "beta-carotene",
    "lycopene for color": "lycopene",
    "natural lycopene for color": "lycopene",

    # === Maple, tapioca ===
    "maple syrup": "sucrose",  # primarily sugar
    "tapioca syrup": "tapioca",
    "tapioca syrup solids": "tapioca",
    "tapioca maltodextrin": "tapioca",
    "tapioca dextrin": "tapioca",

    # === Buckwheat (buckwheat flour has 41 compounds) ===
    "buckwheat": "buckwheat flour",

    # === Bamboo ===
    "bamboo fiber": "bamboo shoot",
    "bamboo fibers": "bamboo shoot",
    "bamboo": "bamboo shoot",

    # === Chicory ===
    "chicory root fiber": "chicory",
    "chicory root extract": "chicory",
    "chicory fiber": "chicory",

    # === Mushroom / Mycelium (mushroom fruitbody has 272 compounds — best entry) ===
    "mycoprotein": "mushroom fruitbody",
    "mycelium": "mushroom fruitbody",
    "mushroom mycelium": "mushroom fruitbody",
    "mushroom root": "mushroom fruitbody",

    # === Dairy (milk has 450 compounds — best dairy entry) ===
    "cream": "milk",
    "cultured cream": "milk",
    "beef": "ground beef",
    "casein": "milk",  # casein is the primary milk protein
    "calcium caseinate": "milk",
    "micellar casein": "milk",
    "mozzarella": "cheddar cheese",  # mozzarella has 0 compounds, cheddar has 113
    "cheese culture": "cheddar cheese",
    "cheese cultures": "cheddar cheese",

    # === Vitamins & minerals (chemical path) ===
    "iron": "ferric oxide",
    "reduced iron": "ferric oxide",
    "b12": "vitamin b12",
    "b3": "niacin",
    "b6": "pyridoxine",
    "b1": "thiamine",
    "b2": "riboflavin",
    "b5": "pantothenic acid",
    "b9": "folic acid",
    "thiamine mononitrate": "thiamine",
    "mixed tocopherols": "tocopherol",
    "natural vitamin e": "tocopherol",
    "calcium lactate": "lactic acid",
    "calcium pantothenate": "pantothenic acid",
    "vitamin a acetate": "retinol",

    # === Starch variants ===
    "modified food starch": "starch",
    "food starch modified": "starch",
    "food starch-modified": "starch",
    "food starch – modified": "starch",
    "food starches- modified": "starch",
    "modified starch": "starch",
    "modified vegetable gum": "starch",
    "potato maltodextrin": "potato starch",

    # === Other chemical additives ===
    "sodium acid pyrophosphate": "pyrophosphoric acid",
    "potassium lactate": "lactic acid",
    "cultured dextrose": "dextrose",
    "cultured cane sugar": "cane sugar",
    "cultured sugar": "sucrose",
    "mono and diglycerides": "monoglycerides",
    "medium chain triglycerides": "triglycerides",
    "medium long chain triglycerides": "triglycerides",
    "glucono delta-lactone": "gluconolactone",
    "glucono-delta-lactone": "gluconolactone",

    # === Psyllium (not in FoodAtlas - map to cellulose as structural proxy) ===
    "psyllium husk": "cellulose",
    "psyllium": "cellulose",
    "psyllium husk fiber": "cellulose",
    "psyllium seed husk": "cellulose",
    "psyllium fiber": "cellulose",

    # === Fruit/juice (vegetable has 280 compounds, citrus fruit juice 14) ===
    "fruit juice": "citrus fruit juice",
    "fruit juice for color": "citrus fruit juice",
    "fruit juice color": "citrus fruit juice",
    "vegetable juice": "vegetable",
    "vegetable juice for color": "vegetable",
    "vegetable juice color": "vegetable",
    "fruit and vegetable juice": "vegetable",
    "fruit and vegetable juice concentrate": "vegetable",

    # === Additional high-frequency ingredients ===
    "coconut cream": "coconut",
    "lemon juice concentrate": "lemon",
    "beet powder": "common beet",
    "beet juice concentrate": "common beet",
    "beet juice": "common beet",
    "beet juice extract": "common beet",
    "beet juice powder": "common beet",
    "beet": "common beet",
    "beets": "common beet",
    "beetroot": "common beet",
    "beetroot powder": "common beet",
    "red beet concentrate": "common beet",
    "red beet juice concentrate": "common beet",
    "tomato paste": "tomato",
    "tomato powder": "tomato",
    "tomato puree": "tomato",
    "lemon juice": "lemon",
    "lime juice": "lime",
    "concentrated lime juice": "lime",
    "orange juice concentrate": "orange",
    "apple juice concentrate": "apple",
    "apple puree": "apple",
    "apple puree concentrate": "apple",
    "apple extract": "apple",
    "strawberry puree": "strawberry",
    "strawberry juice concentrate": "strawberry",
    "mango": "mango",
    "banana puree": "banana",
    "pear": "pear",
    "pear puree concentrate": "pear",
    "raspberry": "raspberry",
    "blueberry": "blueberry",
    "cranberry": "cranberry",
    "cherry powder": "cherry",
    "raisin juice concentrate": "raisin",
    "grape": "grape",
    "concentrated grape must": "grape",
    "concentrated grape juice": "grape",
    "tamarind paste": "tamarind",
    "ginger": "ginger food product",
    "turmeric": "turmeric",
    "black pepper": "pepper",
    "white pepper": "pepper",
    "cayenne": "pepper",
    "cayenne powder": "pepper",
    "chili": "pepper",
    "chili powder": "pepper",
    "chipotle chili powder": "pepper",
    "cinnamon": "cinnamon",
    "oregano": "oregano",
    "thyme": "thyme",
    "rosemary": "rosemary",
    "sage": "sage",
    "parsley": "parsley",
    "basil": "basil",
    "bay leaf": "bay leaf",
    "bay leaves": "bay leaf",
    "carrot": "carrot root",
    "carrots": "carrot root",
    "carrot concentrate for color": "carrot root",
    "caramelised carrot": "carrot root",
    "spinach": "spinach",
    "kale": "kale",
    "broccoli": "broccoli floret",
    "celery": "celery",
    "leek": "leek",
    "peanut": "peanut",
    "peanut butter": "peanut butter",
    "almond butter": "almond",
    "cashew": "cashew nut",
    "pistachio": "pistachio nut",
    "walnut": "walnut",
    "hazelnut": "hazelnut",
    "hazelnuts": "hazelnut",
    "macadamia nut butter": "macadamia nut",
    "macadamias": "macadamia nut",
    "cocoa butter": "cocoa butter",
    "cocoa powder": "cocoa",
    "cocoa": "cocoa",
    "dark chocolate": "chocolate",
    "chocolate": "chocolate",
    "coffee extract": "coffee",
    "cold brew coffee": "coffee",
    "vanilla extract": "vanilla",
    "vanilla": "vanilla",
    "honey": "honey",
    "molasses": "honey",  # closest sweet syrup with compounds
    "cane molasses": "honey",
    "barley": "barley seed (raw)",
    "barley malt": "barley malt",
    "malted barley": "barley malt",
    "malt extract": "barley malt",
    "quinoa": "quinoa seed (dried)",
    "flaxseed": "flaxseed",
    "flax": "flaxseed",
    "sesame": "sesame seed",
    "black sesame seeds": "sesame seed",
    "unhulled black sesame seeds": "sesame seed",

    # === Additional food mappings (post-cleaning round 2) ===
    # Fibers → map to parent food
    "citrus fiber": "citrus fruit food product",
    "potato fiber": "potato",
    "wheat fiber": "wheat (fiber fraction)",  # bran is the fiber fraction
    "apple fiber": "apple",
    "carrot fiber": "carrot root",
    "maple fiber": "sucrose",
    "sugar cane fiber": "cane sugar",
    "sugarcane fiber": "cane sugar",

    # Oil blends → map to first/dominant oil
    "canola and sunflower": "canola oil",
    "canola and sunflower oil": "canola oil",
    "canola and olive oil": "canola oil",
    "canola and olive oil blend": "canola oil",
    "canola and safflower oil": "canola oil",
    "hi-oleic sunflower": "sunflower oil",
    "coconut and sunflower oils": "coconut oil",
    "flax and olive oils": "flaxseed",
    "rapeseed and sunflower": "canola oil",
    "rapeseed in varying proportions": "canola oil",
    "canola lecithin": "canola oil",
    "safflower": "safflower oil",

    # Glycerin/glycerol → chemical
    "vegetable glycerin": "glycerol",
    "vegetable glycerine": "glycerol",
    "glycerin": "glycerol",

    # Whey protein (animal-free or not → similar compound profile)
    "animal-free whey protein": "mammalian milk whey",
    "non-animal whey protein": "mammalian milk whey",

    # Phosphates → chemical
    "dicalcium phosphate": "calcium phosphate",
    "calcium potassium phosphate citrate": "calcium phosphate",
    "tricalcium citrate": "citric acid",
    "tripotassium citrate": "citric acid",
    "disodium diphosphate": "phosphoric acid",
    "diphosphates": "phosphoric acid",

    # Purees/concentrates → parent food
    "garlic puree": "garlic",
    "garlic juice concentrate": "garlic",
    "carrot puree": "carrot root",
    "carrot concentrate for color": "carrot root",
    "concentrated radish juice": "radish",
    "radish concentrate for color": "radish",
    "purple carrot juice": "carrot root",
    "concentrated beet juice": "common beet",
    "beetroot juice concentrate": "common beet",
    "mushroom juice concentrate": "mushroom fruitbody",
    "caramelized carrot concentrate": "carrot root",
    "butternut squash puree": "butternut squash (raw)",
    "tomato concentrate": "tomato",
    "tomato lycopene extract for color": "lycopene",
    "sweet potato puree": "sweet potato",
    "sweet potato maltodextrin": "sweet potato",
    "onion juice concentrate": "onion",
    "onion and carrot juice concentrate": "onion",
    "lemon peel": "lemon",
    "dehydrated lemon peel": "lemon",
    "passionfruit juice concentrate": "passion fruit",
    "caramelized pear juice concentrate": "pear",
    "concentrated caramelised pear juice": "pear",

    # Peppers
    "jalapenos": "pepper",
    "pepperoncini": "pepper",
    "hanabero peppers": "pepper",

    # Starch blends → primary starch
    "potato dextrin": "potato starch",
    "modified potato and corn starch": "potato starch",
    "potato and corn starch": "potato starch",
    "modified tapioca and potato starch": "tapioca",
    "food starch": "starch",
    "vegetable root starch": "starch",
    "cornflour": "corn flour",
    "wheat dextrose": "dextrose",  # it's a sugar, not wheat-flavor

    # Protein variants
    "fava bean protein isolate": "faba bean",
    "potato protein isolate": "potato",
    "potato plant protein": "potato",
    "pea fiber and starch": "pea",
    "pea protein blend": "pea",
    "pea and rice protein concentrate": "pea",
    "pea protein fermented by shiitake mycelia": "pea",
    "pea and rice protein fermented by shiitake mycelia": "pea",
    "rehydrated pea and wheat protein": "pea",
    "rehydrated pea protein and rehydrated wheat protein": "pea",
    "rehydrated textured pea protein": "pea",
    "texturized protein": "pea",
    # "textured wheat gluten" handled in wheat-fraction section above
    "baker's yeast protein": "yeast",
    "field bean protein": "faba bean",
    "vegetable protein": "pea",

    # Soy sauce → soybean
    "soy sauce": "soybean",
    "tamari soy sauce": "soybean",

    # Lactic acid variants → chemical
    "non-dairy lactic acid": "lactic acid",
    "vegan lactic acid": "lactic acid",
    "veg lactic acid": "lactic acid",

    # Smoke variants → intentionally skip (no food compound)
    # "smoke", "smoked water", "natural applewood smoke flavor", etc.
    # These have no meaningful chemical compound mapping.

    # Vanilla variants
    "vanilla bean seeds": "vanilla",
    "vanilla bean specks": "vanilla",

    # Other specific foods
    "rowanberry fruit extract": "sweet rowanberry",
    "rowanberry extract": "sweet rowanberry",
    "ethical palm fruit": "palm oil",
    "palm and palm kernal oil": "palm oil",
    "refined coconut fat": "coconut oil",
    "refined non-hydrogenated coconut oil": "coconut oil",
    "coconut syrup": "coconut",
    "fermented corn sugar": "dextrose",  # fermentation product is sugar
    "fermented dextrose": "dextrose",
    "navy bean flour": "navy bean",
    "white distilled vinegar": "acetic acid",
    "wine vinegar": "acetic acid",
    "sweet cream": "milk",
    "pasteurized milk and cream": "milk",
    "hydrolyzed corn protein": "corn (whole grain or flour)",
    "hydrolyzed beef stock": "ground beef",
    "hydrolyzed sunflower lecithin": "sunflower oil",
    "rosemary antioxidant": "rosemary",
    "natural rosemary extract": "rosemary",
    "sage oil": "sage",
    "ginger oil": "ginger food product",
    "mustard powder": "brown mustard",
    "cellulose and xanthan gums": "xanthan",
    "xanthan and guar gums": "xanthan",
    "locust bean and cellulose gums": "locust bean",
    "gellan": "pectin",
    "mono- and diglycerides of fatty acids": "monoglycerides",
    "vegetable mono and diglycerides": "monoglycerides",
    "cocoa mass": "cocoa",
    "cocoa solids": "cocoa",
    "chocolate liquor": "chocolate",

    # Misc chemicals
    "sodium alginate casing": "alginic acid",
    "calcium alginate casing": "alginic acid",
    "sodium erythorbate": "ascorbic acid",
    "sodium diacetate": "acetic acid",
    "sorbic acid as a preservative": "sorbic acid",
    "sucrose esters of fatty acids": "sucrose",

    # === Coverage additions (post-audit on full NECTAR corpus) ===
    # Unmatched ingredients found by sweeping all 723 cleaned NECTAR
    # ingredients against the v3.2 lookup; targets verified to have
    # food-compounds or chemical entries in v3.2.
    "hibiscus": "roselle",                        # Hibiscus sabdariffa — 53 compounds
    "cassava root syrup": "cassava",              # 98 compounds
    "vegetarian lactose": "lactose",              # chemical
    "lactose": "lactose",                         # chemical (defensive)
    "dha algal oil": "algae",                     # 79 compounds via algae food path
    "dehydrated potato flakes": "potato",
    "cultured celery juice": "celery",
    "cultured celery powder": "celery",
    "lime juice solids": "lime",
    "bay powder": "bay leaf",
    "naturally extracted annatto": "bixin",
    "plain caramel": "caramel furanone",
    "beet juice concentrate color": "common beet",
    "mlct oil": "triglycerides",                  # medium-long chain triglycerides
    "caseed": "milk",                             # apparent typo of casein
    "cellulose gel": "cellulose",
    "red yeast rice powder": "rice",
    "rice flakes": "rice",
    "rusk": "common wheat kernel",                # rusk = dried wheat bread/cracker
    "processed eucheuma seaweed": "kombu",        # red-algae proxy in v3.2
    "barley yeast extract": "yeast",
    "burnt sugar": "sucrose",
    "caramelised sugar syrup": "sucrose",
    # NOTE: generic fiber labels intentionally NOT mapped to cellulose.
    # An ablation showed mapping all three of these to cellulose (single-
    # compound chemical) injects the same SMILES across ~30 products and
    # causes a small but consistent regression (0.6711 → 0.6572 on
    # BT+Gemini NNLS). Leave them unmatched — the inverse-rank weighted
    # ingredient aggregator handles a missing ingredient cleanly.
    # "plant fiber": "cellulose",
    # "vegetable fiber": "cellulose",
    # "dietary fiber": "cellulose",
    "lion's mane mushroom extract": "mushroom fruitbody",  # gets ENTRY_PAIRS merge
    "drinking water": "water",
    "grain alcohol": "alcohol",
    "dried glucose syrup": "dextrose",
    "dextrose monohydrate": "dextrose",
    "sodium tripolyphosphate": "phosphoric acid",
    "sodium carbonates": "sodium carbonate",
    "titanium white": "titanium dioxide",
    "polysorbate 65": "monoglycerides",           # fatty-acid emulsifier proxy

    # === Fermentation cultures (skip - not food compounds) ===
    # These are bacterial strains, not chemical compounds.
    # Mapped to None via absence from dict.

    # === Generic (skip - too vague for compound mapping) ===
    # "natural flavors", "natural flavor", "flavorings", "spice extractives"
    # "stabilizer", "thickener", "emulsifier", "color", "preservative"
    # "cultures", "live active cultures", "probiotic cultures"
    # These are intentionally NOT mapped.
}
# fmt: on


# v4.0-only overrides merged on top of INGREDIENT_SYNONYMS / ENTRY_PAIRS.
# These re-route names that v4.0 either renamed (e.g. "brown mustard" →
# "mustard seed"), stripped from synonym lists in favour of NCBI taxon
# URIs, or wiped to zero compounds (the *_food_product entities). Each
# entry's target is the closest-chemistry v4.0 entity that actually has
# compound data.
# fmt: off
INGREDIENT_SYNONYMS_V40: Dict[str, str] = {
    # Dehydrated forms (powders) → raw plant chemistry, where the dedicated
    # v4.0 powder entity is sparse and the raw entity is rich.
    "onion powder":  "onion (raw)",
    "garlic powder": "garlic",

    # Common-English aliases dropped from entities.synonyms in v4.0.
    "wheat":          "common wheat kernel",
    "canola":         "canola oil",
    "guar":           "guar gum",
    "konjac":         "konjacu tuber",
    "konjac gum":     "konjacu tuber",
    "konjac powder":  "konjacu tuber",
    "konjac root":    "konjacu tuber",
    "mustard":        "mustard seed",
    "dry mustard":    "mustard seed",
    "mustard powder": "mustard seed",

    # All yeast-class ingredients (active, inactive, deactivated,
    # nutritional, autolyzed, torula) collapse to "yeast extract", the
    # only yeast-bearing entity left in v4.0.
    "yeast":                    "yeast extract",
    "torula yeast":             "yeast extract",
    "deactivated yeast":        "yeast extract",
    "dried yeast":              "yeast extract",
    "inactivated yeast":        "yeast extract",
    "inactive dried yeast":     "yeast extract",
    "inactive dry yeast":       "yeast extract",
    "inactive yeast":           "yeast extract",
    "nutritional yeast":        "yeast extract",
    "autolyzed yeast extract":  "yeast extract",
}

# Reserved for entries whose primary/fallback ordering should differ
# between v3.2 and v4.0. Empty for now — the base ENTRY_PAIRS handles
# both versions gracefully via variadic fallback (entries missing in
# one version contribute 0 compounds without erroring).
ENTRY_PAIRS_V40: Dict[str, tuple] = {}
# fmt: on

# Primary (high-conc%) + Fallback (broad) entry pairs.
# Key: synonym target name (lowercase). Value: (primary, fallback).
# Primary is used for concentration-weighted compounds;
# Fallback adds extra compounds (with equal weight) not in primary.
# fmt: off
ENTRY_PAIRS: Dict[str, tuple] = {
    # Mushroom: button (80% conc) + fruitbody (272 compounds)
    "mushroom fruitbody": ("white button mushroom", "mushroom fruitbody"),
    # Soy: milk (82% conc) + soybean (390 compounds)
    "soybean": ("soybean milk", "soybean"),
    # Soy protein: defatted flour (100% conc) + soybean (390 compounds)
    "soybean flour (defatted)": ("soybean flour (defatted)", "soybean"),
    "soybean flour": ("soybean flour", "soybean"),
    # Milk/dairy: cow whole milk 3.5% (91% conc) + milk + cow milk.
    # v3.2 had the rich data on the "milk" entity (450 cmpds); v4.0 renamed
    # that to "cow milk (liquid)" (1728 cmpds) AND introduced a new
    # "cow milk" entity with 6819 cmpds. Listing all candidates lets each
    # version pick up whichever entity exists; merge dedupes by entity id.
    "milk": ("cow whole milk 3.5% fat", "milk", "cow milk", "cow milk (liquid)"),
    # v4.0 direct-lookup path: "Milk" / "Pasteurized Milk" / etc. resolve to
    # the entity literally named "cow milk (liquid)", so we need a separate
    # ENTRY_PAIRS key to attach the multi-fallback there too.
    "cow milk (liquid)": ("cow milk (liquid)", "cow milk", "cow whole milk 3.5% fat"),
    # Oat: oat flour (97% conc) + oat (170 compounds)
    "oat": ("oat flour", "oat"),
    # Sunflower seed: roasted (100% conc) + seed (55 compounds)
    "sunflower seed": ("sunflower seed (shell off, roasted)", "sunflower seed"),
    # Sunflower oil: already 84% conc, no change needed
    # Corn: keep maize, no good high-conc alternative with sufficient compounds
    # Wheat: keep common wheat kernel, durum is only slightly better
    # Rice: keep rice, no high-conc variant with sufficient compounds

    # Multi-target fallbacks for entities that have zero compounds in
    # v4.0 (most are *_food_product entities; some were wiped via the
    # 2026-04-21 KG release). Each tuple lists chemically-similar
    # entities; the resolver gathers their union.
    "locust bean":         ("locust bean", "guar gum"),
    "algae":               ("algae", "green algae"),
    # Wheat: ~7 NECTAR ingredient classes (gluten, flour, protein,
    # starch, etc.) map to common wheat kernel; multi-fallback gathers
    # the cumulative compound set across v4.0's wheat-specific entities.
    "common wheat kernel": (
        "common wheat kernel", "durum wheat kernel", "wheat germ",
        "wheat bran", "whole wheat flour", "white wheat flour (not heat treated)",
    ),
    "maize (corn) food product": (
        "maize (corn) food product", "yellow sweet corn kernel",
        "yellow corn flour", "corn flour", "cornmeal", "oil, corn",
    ),
    "yeast": ("yeast", "yeast extract"),
    "common beet": ("common beet", "beetroot", "beet", "sugar beet"),
    "rosemary": ("rosemary", "sage"),  # closest Lamiaceae cousin

    # Synthetic per-fraction wheat/corn keys (not real entity names):
    # used by INGREDIENT_SYNONYMS to route each wheat/corn ingredient
    # to a chemistry-appropriate target. Bran/germ are EXCLUDED from
    # gluten and starch fractions because vital wheat gluten and wheat
    # starch are extracted from the refined endosperm.
    "wheat (gluten/protein fraction)": (
        "common wheat kernel",                   # gluten lives in kernel endosperm
        "durum wheat kernel",
        "white wheat flour (not heat treated)",  # refined endosperm = where gluten extracts go
        "white wheat flour",
    ),
    "wheat (starch fraction)": (
        "white wheat flour (not heat treated)",  # refined endosperm starch
        "white wheat flour",
        "common wheat kernel",
        "durum wheat kernel",
    ),
    "wheat (fiber fraction)": (
        "wheat bran",                            # bran IS the fiber fraction
        "whole wheat flour",                     # contains some bran
        "common wheat kernel",
    ),
    "wheat (refined flour)": (
        "white wheat flour (not heat treated)",  # refined endosperm flour
        "white wheat flour",
        "whole wheat flour",
        "common wheat kernel",
    ),
    "corn (starch fraction)": (
        "corn flour",                            # closest to refined corn starch
        "yellow corn flour",
        "yellow sweet corn kernel",
        "cornmeal",
    ),
    "corn (oil fraction)": (
        "oil, corn",                             # exact match in v4.0 (38 cmpds)
        "yellow sweet corn kernel",
    ),
    "corn (whole grain or flour)": (
        "yellow sweet corn kernel",              # whole-grain corn
        "yellow corn flour",
        "corn flour",
        "cornmeal",
    ),
}
# fmt: on


@dataclass
class CompoundMatch:
    """A chemical compound matched to an ingredient."""

    foodatlas_id: str
    name: str
    pubchem_cid: Optional[int] = None
    chebi_id: Optional[int] = None
    concentration: Optional[float] = None
    match_path: str = "food"  # "food" or "chemical"


@dataclass
class IngredientMapping:
    """Result of mapping a single ingredient to compounds."""

    ingredient_name: str
    matched_entity_id: Optional[str] = None
    matched_entity_name: Optional[str] = None
    match_tier: Optional[int] = None  # 1=exact, 2=normalized, 3=chemical
    compounds: List[CompoundMatch] = field(default_factory=list)


class FoodAtlasMapper:
    """Maps ingredient names to chemical compounds via FoodAtlas.

    Three-tier matching strategy:
    1. Exact match (case-insensitive) on food lookup table
    2. Normalized match: strip modifiers and retry
    3. Chemical lookup: ingredient is itself a chemical compound
    """

    def __init__(
        self,
        food_atlas_dir: str | Path,
        food_atlas_version: Optional[str] = None,
        synonym_hydration_dir: Optional[str | Path] = None,
        conc_unit_allowlist: Optional[Set[str]] = None,
        disable_v40_native_dict: bool = False,
        flavor_descriptor_filter: bool = False,
        include_ambiguous_attestations: bool = False,
        attestation_source_blacklist: Optional[Set[str]] = None,
        lit2kg_filter_score_min: Optional[float] = None,
        soft_quality_weight: bool = False,
    ) -> None:
        """
        Args:
            food_atlas_dir: Directory containing the FoodAtlas bundle.
            food_atlas_version: "v3.2" or "v4.0". If None, auto-detect.
            synonym_hydration_dir: Optional path to a v3.2-style bundle whose
                lookup_table_food.tsv / lookup_table_chemical.tsv supply
                additional name → entity_id aliases. v4.0 dropped most plural /
                English synonyms from entities.synonyms (replaced with NCBI
                taxon URIs); since v4.0 entity IDs are mostly stable from v3.2
                we recover lost matches by joining v3.2's lookup tables on
                entity_id, dropping any IDs that no longer exist in v4.0.
                Only the alias dictionary is borrowed — never compounds or
                concentrations. Ignored when food_atlas_version == "v3.2".
            conc_unit_allowlist: Optional set of conc_unit strings to keep
                when building the evidence → concentration index. v4.0
                introduced "ions" and "molecules" units (from chebi
                attestations) that aren't commensurate with v3.2's
                mg/100g-dominated values; restricting to the v3.2 unit
                family avoids skewing the conc-weighted aggregation.
                None = keep all (faithful v3.2 behavior).
        """
        self.food_atlas_dir = Path(food_atlas_dir)
        if food_atlas_version is None:
            food_atlas_version = self._detect_version(self.food_atlas_dir)
        if food_atlas_version not in ("v3.2", "v4.0"):
            raise ValueError(
                f"Unsupported food_atlas_version: {food_atlas_version!r} "
                "(expected 'v3.2' or 'v4.0')"
            )
        self.food_atlas_version = food_atlas_version
        self.synonym_hydration_dir = (
            Path(synonym_hydration_dir) if synonym_hydration_dir else None
        )
        self.conc_unit_allowlist = conc_unit_allowlist
        self.flavor_descriptor_filter = flavor_descriptor_filter
        self.include_ambiguous_attestations = include_ambiguous_attestations
        self.attestation_source_blacklist = attestation_source_blacklist
        self.lit2kg_filter_score_min = lit2kg_filter_score_min
        self.soft_quality_weight = soft_quality_weight
        # Build version-aware mapping dicts. For v4.0 (with V40 overrides
        # enabled), V40 entries take precedence over base entries; for v3.2
        # (or v4.0 with V40 disabled), only base entries apply.
        self._use_v40_overrides = (
            self.food_atlas_version == "v4.0" and not disable_v40_native_dict
        )
        if self._use_v40_overrides:
            self._ingredient_synonyms = {**INGREDIENT_SYNONYMS, **INGREDIENT_SYNONYMS_V40}
            self._entry_pairs = {**ENTRY_PAIRS, **ENTRY_PAIRS_V40}
        else:
            self._ingredient_synonyms = INGREDIENT_SYNONYMS
            self._entry_pairs = ENTRY_PAIRS
        self._load_data()
        self._build_indices()

    @staticmethod
    def _detect_version(d: Path) -> str:
        if (d / "entities.parquet").exists():
            return "v4.0"
        if (d / "entities.tsv").exists():
            return "v3.2"
        raise FileNotFoundError(
            f"Could not auto-detect FoodAtlas version in {d}: "
            "neither entities.parquet (v4.0) nor entities.tsv (v3.2) present"
        )

    def _load_data(self) -> None:
        if self.food_atlas_version == "v3.2":
            self._load_data_v32()
        else:
            self._load_data_v40()

    def _load_data_v32(self) -> None:
        """Load FoodAtlas v3.2 TSV files."""
        d = self.food_atlas_dir
        logger.info(f"Loading FoodAtlas v3.2 from {d}")

        # Food name lookup
        self._food_lookup = pd.read_csv(
            d / "lookup_table_food.tsv", sep="\t",
            converters={"foodatlas_id": literal_eval},
        )

        # Chemical name lookup
        self._chem_lookup = pd.read_csv(
            d / "lookup_table_chemical.tsv", sep="\t",
            converters={"foodatlas_id": literal_eval},
        )

        # Entities (for external_ids → PubChem CIDs)
        self._entities = pd.read_csv(
            d / "entities.tsv", sep="\t",
            converters={"external_ids": literal_eval, "synonyms": literal_eval},
        )

        # Triplets (food "contains" chemical)
        self._triplets = pd.read_csv(
            d / "triplets.tsv", sep="\t",
            converters={"metadata_ids": literal_eval},
        )
        # Filter to "contains" relationships only
        self._contains = self._triplets[self._triplets["relationship_id"] == "r1"]
        self._contains = self._contains.rename(columns={"metadata_ids": "_evidence_ids"})

        # Metadata for concentrations: foodatlas_id → conc_value
        self._metadata = pd.read_csv(
            d / "metadata_contains.tsv", sep="\t",
            converters={"reference": literal_eval},
        )
        self._evidence_id_col = "foodatlas_id"

        logger.info(
            f"FoodAtlas v3.2 loaded: {len(self._food_lookup)} food names, "
            f"{len(self._chem_lookup)} chemical names, "
            f"{len(self._entities)} entities, "
            f"{len(self._contains)} contains-relationships"
        )

    @staticmethod
    def _hydrate_synonyms_from(
        v32_dir: Path,
        food_rows: Dict[str, List[str]],
        chem_rows: Dict[str, List[str]],
        v40_entity_ids: Set[str],
    ) -> tuple:
        """Add (name → entity_id) aliases from a v3.2 lookup table.

        Only adds entity_ids that still exist in v4.0.
        """
        n_food_added = 0
        n_chem_added = 0
        food_table = v32_dir / "lookup_table_food.tsv"
        chem_table = v32_dir / "lookup_table_chemical.tsv"
        if food_table.exists():
            df = pd.read_csv(
                food_table, sep="\t",
                converters={"foodatlas_id": literal_eval},
            )
            for _, row in df.iterrows():
                name = str(row["name"]).strip().lower()
                if not name:
                    continue
                ids_kept = [eid for eid in row["foodatlas_id"] if eid in v40_entity_ids]
                if not ids_kept:
                    continue
                if name not in food_rows:
                    n_food_added += 1
                food_rows.setdefault(name, []).extend(ids_kept)
        if chem_table.exists():
            df = pd.read_csv(
                chem_table, sep="\t",
                converters={"foodatlas_id": literal_eval},
            )
            for _, row in df.iterrows():
                name = str(row["name"]).strip().lower()
                if not name:
                    continue
                ids_kept = [eid for eid in row["foodatlas_id"] if eid in v40_entity_ids]
                if not ids_kept:
                    continue
                if name not in chem_rows:
                    n_chem_added += 1
                chem_rows.setdefault(name, []).extend(ids_kept)
        return n_food_added, n_chem_added

    def _expand_aliases_algorithmic(
        self,
        food_rows: Dict[str, List[str]],
        chem_rows: Dict[str, List[str]],
    ) -> int:
        """Expand the lookup with algorithmic name variants — pure v4.0,
        no external data. For each existing (name, entity_id) pair we add
        morphological variants:
          - plural → singular (almonds → almond)
          - singular → plural (almond → almonds)
          - parenthetical-stripped ("beet (raw)" → "beet")
          - "<X> food product" → "<X>" (so "yeast food product" also
            answers to "yeast")
        Returns the count of NEW (name → entity_id) entries added.
        """
        added = 0

        # Common processing modifiers we strip: NECTAR ingredient names
        # often have these prefixes/suffixes that v4.0 entities don't.
        _STRIP_PREFIXES = (
            "ground ", "raw ", "dried ", "dehydrated ", "fresh ", "whole ",
            "organic ", "natural ", "refined ", "crude ", "powdered ",
            "frozen ", "cooked ", "roasted ", "toasted ", "fermented ",
        )
        _STRIP_SUFFIXES = (
            " powder", " extract", " concentrate", " isolate", " puree",
            " paste", " juice", " oil", " flour", " starch", " fiber",
            " seed", " seeds", " kernel", " kernels", " flakes",
        )

        def variants(s: str) -> Set[str]:
            v: Set[str] = set()
            s = s.strip().lower()
            if not s:
                return v
            # Drop URI-style synonyms ("<http://...>", "http://...", etc.)
            if s.startswith("<") and s.endswith(">"):
                return v
            if s.startswith("http://") or s.startswith("https://"):
                return v
            v.add(s)
            # Strip parentheticals: "beet (raw)" → "beet"
            stripped = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
            if stripped and stripped != s:
                v.add(stripped)
            # Drop "food product" suffix: "yeast food product" → "yeast"
            if s.endswith(" food product"):
                v.add(s[: -len(" food product")].strip())
            # Strip common processing modifiers: "ground almond" → "almond",
            # "almond powder" → "almond". Conservative: only generate the
            # stripped form if the result is still a reasonable food name.
            for prefix in _STRIP_PREFIXES:
                if s.startswith(prefix):
                    rest = s[len(prefix):].strip()
                    if rest and len(rest) > 2:
                        v.add(rest)
            for suffix in _STRIP_SUFFIXES:
                if s.endswith(suffix):
                    head = s[: -len(suffix)].strip()
                    if head and len(head) > 2:
                        v.add(head)
            # Hyphen handling: replace hyphens with spaces ("garlic-powder"
            # → "garlic powder") and vice versa. Many NECTAR labels use
            # one form; v4.0 uses the other.
            if "-" in s:
                v.add(s.replace("-", " "))
            if " " in s:
                v.add(s.replace(" ", "-"))
            # Plural / singular morphology (English heuristics)
            if s.endswith("ies"):
                v.add(s[:-3] + "y")
            elif s.endswith("es") and len(s) > 3:
                v.add(s[:-2])
            elif s.endswith("s") and len(s) > 2 and not s.endswith("ss"):
                v.add(s[:-1])
            else:
                if s.endswith("y") and len(s) > 1 and s[-2] not in "aeiou":
                    v.add(s[:-1] + "ies")
                elif s.endswith(("s", "x", "ch", "sh", "z")):
                    v.add(s + "es")
                else:
                    v.add(s + "s")
            return v

        # Walk each entity's synonyms + common_name; expand and merge into
        # food_rows / chem_rows depending on entity_type.
        for _, row in self._entities.iterrows():
            etype = row["entity_type"]
            if etype == "food":
                target = food_rows
            elif etype == "chemical":
                target = chem_rows
            else:
                continue
            eid = row["foodatlas_id"]
            base_names: List[str] = []
            common = row.get("common_name", "")
            if isinstance(common, str) and common:
                base_names.append(common)
            syns = row.get("synonyms")
            if isinstance(syns, list):
                base_names.extend([s for s in syns if isinstance(s, str)])
            for nm in base_names:
                for v in variants(nm):
                    if not v:
                        continue
                    if v not in target or eid not in target[v]:
                        target.setdefault(v, []).append(eid)
                        added += 1
        return added

    @staticmethod
    def _hydrate_synonyms_from_static_file(
        path: Path,
        food_rows: Dict[str, List[str]],
        chem_rows: Dict[str, List[str]],
        v40_entity_ids: Set[str],
    ) -> tuple:
        """Add (name → entity_id) aliases from a static TSV shipped alongside
        v4.0 (schema: name<TAB>foodatlas_id<TAB>entity_type). Pre-filtered to
        v4.0 entity IDs at export time, so this method is reproducible from
        v4.0 + the alias file alone — no live v3.2 bundle needed.
        """
        n_food_added = 0
        n_chem_added = 0
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            name = str(row["name"]).strip().lower()
            eid = str(row["foodatlas_id"]).strip()
            etype = str(row["entity_type"]).strip()
            if not name or not eid:
                continue
            if eid not in v40_entity_ids:
                continue
            if etype == "food":
                if name not in food_rows:
                    n_food_added += 1
                food_rows.setdefault(name, []).append(eid)
            elif etype == "chemical":
                if name not in chem_rows:
                    n_chem_added += 1
                chem_rows.setdefault(name, []).append(eid)
        return n_food_added, n_chem_added

    def _load_data_v40(self) -> None:
        """Load FoodAtlas v4.0 parquet files.

        v4.0 differences vs v3.2:
          - Lookup tables (food/chemical) are derived from entities.synonyms
            (no separate lookup_table_*.tsv files).
          - Triplets reference attestations via attestation_ids (was metadata_ids).
          - Concentrations live in attestations.parquet keyed by attestation_id
            (was metadata_contains.tsv keyed by foodatlas_id).
          - JSON-encoded list/dict columns (synonyms, external_ids,
            attestation_ids) are stored as strings; parsed with json.loads.
        """
        d = self.food_atlas_dir
        logger.info(f"Loading FoodAtlas v4.0 from {d}")

        def _parse_json(s, default):
            if not isinstance(s, str) or not s:
                return default
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return default

        # Entities
        self._entities = pd.read_parquet(d / "entities.parquet")
        self._entities["synonyms"] = self._entities["synonyms"].apply(
            lambda s: _parse_json(s, [])
        )
        self._entities["external_ids"] = self._entities["external_ids"].apply(
            lambda s: _parse_json(s, {})
        )

        # Derive food/chemical lookup tables from synonyms + common_name.
        food_rows: Dict[str, List[str]] = {}
        chem_rows: Dict[str, List[str]] = {}
        for _, row in self._entities.iterrows():
            etype = row["entity_type"]
            if etype == "food":
                target = food_rows
            elif etype == "chemical":
                target = chem_rows
            else:
                continue
            eid = row["foodatlas_id"]
            names = list(row["synonyms"]) if isinstance(row["synonyms"], list) else []
            common = row.get("common_name", "")
            if isinstance(common, str) and common:
                names.append(common)
            for nm in names:
                if not isinstance(nm, str):
                    continue
                key = nm.strip().lower()
                if not key:
                    continue
                target.setdefault(key, []).append(eid)

        # Algorithmic alias expansion (pure v4.0; no v3.2 dependency).
        # v4.0's entities.synonyms field is stripped (mean 1.6 aliases per
        # entity vs 2.9 in v3.2; English plurals and common variants were
        # replaced by NCBI taxon URIs). We expand each base synonym with
        # plural/singular morphological variants and parenthetical-stripped
        # forms — pure linguistic transformations, no external data.
        n_alg_added = self._expand_aliases_algorithmic(food_rows, chem_rows)
        if n_alg_added:
            logger.info(
                f"Algorithmic alias expansion: +{n_alg_added} (name → entity_id) entries"
            )
        # Optional / opt-in: live v3.2 bundle hydration. DEPRECATED — leave
        # synonym_hydration_dir as None for v4.0-only reproducibility.
        if self.synonym_hydration_dir is not None:
            v40_eids = set(self._entities["foodatlas_id"])
            n_food_added, n_chem_added = self._hydrate_synonyms_from(
                self.synonym_hydration_dir, food_rows, chem_rows, v40_eids,
            )
            logger.info(
                f"[DEPRECATED] Live v3.2 hydration from {self.synonym_hydration_dir}: "
                f"+{n_food_added} food, +{n_chem_added} chemical aliases. "
                "Pass synonym_hydration_dir=None for pure v4.0 reproducibility."
            )

        self._food_lookup = pd.DataFrame(
            [{"name": n, "foodatlas_id": ids} for n, ids in food_rows.items()]
        )
        self._chem_lookup = pd.DataFrame(
            [{"name": n, "foodatlas_id": ids} for n, ids in chem_rows.items()]
        )

        # Triplets
        triplets = pd.read_parquet(d / "triplets.parquet")
        triplets["attestation_ids"] = triplets["attestation_ids"].apply(
            lambda s: _parse_json(s, [])
        )
        self._triplets = triplets
        self._contains = triplets[triplets["relationship_id"] == "r1"].rename(
            columns={"attestation_ids": "_evidence_ids"}
        )

        # Attestations: only the unambiguous table is keyed for concentration.
        meta_main = pd.read_parquet(d / "attestations.parquet")
        if self.include_ambiguous_attestations and (d / "attestations_ambiguous.parquet").exists():
            meta_amb = pd.read_parquet(d / "attestations_ambiguous.parquet")
            # Same schema; concat. Some triplets reference attestation_ids
            # that landed in the ambiguous table — including them lets us
            # recover their conc_value where present.
            self._metadata = pd.concat([meta_main, meta_amb], ignore_index=True)
            logger.info(
                f"Including attestations_ambiguous: +{len(meta_amb)} rows "
                f"({len(self._metadata)} total)"
            )
        else:
            self._metadata = meta_main
        self._evidence_id_col = "attestation_id"

        logger.info(
            f"FoodAtlas v4.0 loaded: {len(self._food_lookup)} food names "
            f"(derived), {len(self._chem_lookup)} chemical names (derived), "
            f"{len(self._entities)} entities, "
            f"{len(self._contains)} contains-relationships, "
            f"{len(self._metadata)} attestations"
        )

    def _build_indices(self) -> None:
        """Build fast lookup indices."""
        # Food name → list of entity IDs (lowercase)
        self._food_name_to_ids: Dict[str, List[str]] = {}
        for _, row in self._food_lookup.iterrows():
            name = str(row["name"]).lower().strip()
            entity_ids = row["foodatlas_id"]  # already parsed as list
            self._food_name_to_ids.setdefault(name, []).extend(entity_ids)

        # Chemical name → list of entity IDs (lowercase)
        self._chem_name_to_ids: Dict[str, List[str]] = {}
        for _, row in self._chem_lookup.iterrows():
            name = str(row["name"]).lower().strip()
            entity_ids = row["foodatlas_id"]
            self._chem_name_to_ids.setdefault(name, []).extend(entity_ids)

        # Entity ID → external_ids dict
        self._entity_external: Dict[str, dict] = {}
        self._entity_names: Dict[str, str] = {}
        for _, row in self._entities.iterrows():
            eid = row["foodatlas_id"]
            ext = row.get("external_ids", {})
            if isinstance(ext, dict):
                self._entity_external[eid] = ext
            self._entity_names[eid] = str(row.get("common_name", ""))

        # Optional flavor-relevance filter: keep only chem_ids whose entity
        # has non-empty attributes.flavor_descriptors. This is a chemistry-
        # aware filter that aligns the compound feature with the prediction
        # target (taste/sensory). v4.0-only — relies on attributes column.
        flavor_chem_ids: Optional[Set[str]] = None
        if self.flavor_descriptor_filter and "attributes" in self._entities.columns:
            flavor_chem_ids = set()
            for _, row in self._entities.iterrows():
                if row["entity_type"] != "chemical":
                    continue
                attrs = row["attributes"]
                if isinstance(attrs, str) and attrs:
                    try:
                        d_attrs = json.loads(attrs)
                    except (ValueError, TypeError):
                        continue
                    fd = d_attrs.get("flavor_descriptors", [])
                    if isinstance(fd, list) and len(fd) > 0:
                        flavor_chem_ids.add(row["foodatlas_id"])
            logger.info(
                f"Flavor-descriptor filter: keeping {len(flavor_chem_ids)} "
                "chemicals with non-empty flavor_descriptors"
            )

        # Optional attestation-source blacklist. Drop a contains-triplet if
        # ALL of its attestation_ids are in the blacklisted source set.
        # Use case: drop edges only supported by lit2kg:gpt-* (LLM-extracted
        # from literature, lower precision than chebi/dmd/ctd/foodon/fdc).
        blacklisted_evid_ids: Optional[Set[str]] = None
        if self.attestation_source_blacklist and "source" in self._metadata.columns:
            mask = self._metadata["source"].isin(self.attestation_source_blacklist)
            blacklisted_evid_ids = set(self._metadata.loc[mask, self._evidence_id_col])
            logger.info(
                f"Source blacklist {self.attestation_source_blacklist}: "
                f"{len(blacklisted_evid_ids)} attestation_ids will be ignored"
            )
        # Optional: blacklist lit2kg attestations whose filter_score is below
        # threshold (filter_score is only populated for lit2kg sources). This
        # is a softer alternative to a full source blacklist — it keeps the
        # quality-flagged lit2kg edges while dropping the rest.
        if self.lit2kg_filter_score_min is not None and "filter_score" in self._metadata.columns:
            is_lit2kg = self._metadata["source"].astype(str).str.startswith("lit2kg:", na=False)
            low_score = self._metadata["filter_score"].fillna(-1) < self.lit2kg_filter_score_min
            mask = is_lit2kg & low_score
            extra = set(self._metadata.loc[mask, self._evidence_id_col])
            if blacklisted_evid_ids is None:
                blacklisted_evid_ids = extra
            else:
                blacklisted_evid_ids = blacklisted_evid_ids | extra
            logger.info(
                f"lit2kg filter_score<{self.lit2kg_filter_score_min}: "
                f"+{len(extra)} attestation_ids blacklisted"
            )

        # Food entity → list of (chemical_entity_id, evidence_ids)
        # evidence_ids = metadata_ids in v3.2, attestation_ids in v4.0
        self._food_to_compounds: Dict[str, List[tuple]] = {}
        n_filtered = 0
        n_blacklisted = 0
        for _, row in self._contains.iterrows():
            food_id = row["head_id"]
            chem_id = row["tail_id"]
            if flavor_chem_ids is not None and chem_id not in flavor_chem_ids:
                n_filtered += 1
                continue
            ev_ids_raw = row.get("_evidence_ids", [])
            ev_ids = ev_ids_raw if isinstance(ev_ids_raw, list) else (
                list(ev_ids_raw) if ev_ids_raw is not None else []
            )
            if blacklisted_evid_ids is not None and ev_ids:
                # Keep only if at least one ev_id is NOT in the blacklist.
                if all(eid in blacklisted_evid_ids for eid in ev_ids):
                    n_blacklisted += 1
                    continue
            self._food_to_compounds.setdefault(food_id, []).append((chem_id, ev_ids))
        if blacklisted_evid_ids is not None:
            logger.info(
                f"Source blacklist dropped {n_blacklisted} contains-edges "
                f"(only attested by blacklisted sources)"
            )
        if flavor_chem_ids is not None:
            logger.info(
                f"Flavor-descriptor filter dropped {n_filtered} non-flavor "
                f"contains-edges; kept {sum(len(v) for v in self._food_to_compounds.values())}"
            )

        # Evidence ID → concentration (v3.2: metadata_contains.foodatlas_id;
        # v4.0: attestations.attestation_id). When conc_unit_allowlist is set,
        # entries outside the allowed units are stored as None so the conc-
        # weighted aggregator falls back to the constant default for them
        # (v4.0 introduced "ions" / "molecules" units that are not
        # commensurate with v3.2's mg/100g-dominated values).
        self._meta_conc: Dict[str, Optional[float]] = {}
        # When soft_quality_weight is on, also store a per-attestation
        # quality multiplier in [0, 1]: 1.0 for curated sources (chebi /
        # dmd / ctd / foodon / fdc / cdno / foodatlas), filter_score for
        # lit2kg sources (which is the only family v4.0 populates this
        # field for). NaN/None → 0 (effectively dropped).
        self._meta_quality: Dict[str, float] = {}
        allowed = self.conc_unit_allowlist
        has_unit_col = "conc_unit" in self._metadata.columns
        has_source_col = "source" in self._metadata.columns
        has_fs_col = "filter_score" in self._metadata.columns
        for _, row in self._metadata.iterrows():
            mid = row[self._evidence_id_col]
            val = row.get("conc_value")
            if allowed is not None and has_unit_col:
                unit = row.get("conc_unit")
                unit = str(unit) if unit is not None and pd.notna(unit) else ""
                if unit not in allowed:
                    self._meta_conc[mid] = None
                    if self.soft_quality_weight:
                        self._meta_quality[mid] = 0.0
                    continue
            try:
                self._meta_conc[mid] = float(val) if pd.notna(val) else None
            except (ValueError, TypeError):
                self._meta_conc[mid] = None
            if self.soft_quality_weight:
                src = str(row.get("source")) if has_source_col else ""
                if src.startswith("lit2kg:"):
                    fs = row.get("filter_score") if has_fs_col else None
                    try:
                        q = float(fs) if pd.notna(fs) else 0.0
                    except (ValueError, TypeError):
                        q = 0.0
                    self._meta_quality[mid] = max(0.0, q)
                else:
                    self._meta_quality[mid] = 1.0

        logger.info(
            f"Indices built: {len(self._food_name_to_ids)} food names, "
            f"{len(self._chem_name_to_ids)} chemical names, "
            f"{len(self._food_to_compounds)} foods with compounds"
        )

    def _get_pubchem_cid(self, entity_id: str) -> Optional[int]:
        """Extract PubChem CID from an entity's external_ids."""
        ext = self._entity_external.get(entity_id, {})
        cids = ext.get("pubchem_compound", [])
        return cids[0] if cids else None

    def _get_chebi_id(self, entity_id: str) -> Optional[int]:
        """Extract ChEBI ID from an entity's external_ids."""
        ext = self._entity_external.get(entity_id, {})
        ids = ext.get("chebi", [])
        return ids[0] if ids else None

    def _get_concentration(self, metadata_ids: List[str]) -> Optional[float]:
        """Get (quality-weighted) average concentration from metadata entries.

        When soft_quality_weight is on, weight each attestation's conc_value
        by its quality score: 1.0 for curated sources, filter_score for
        lit2kg attestations. Lit2kg edges with filter_score=0 / NaN are
        effectively dropped without losing the edge entirely (so the FART
        embedding still enters the per-ingredient compound feature).
        """
        if self.soft_quality_weight and self._meta_quality:
            num = 0.0
            den = 0.0
            for mid in metadata_ids:
                if mid not in self._meta_conc or self._meta_conc[mid] is None:
                    continue
                q = self._meta_quality.get(mid, 1.0)
                if q <= 0:
                    continue
                num += self._meta_conc[mid] * q
                den += q
            return num / den if den > 0 else None
        values = [
            self._meta_conc[mid]
            for mid in metadata_ids
            if mid in self._meta_conc and self._meta_conc[mid] is not None
        ]
        if values:
            return sum(values) / len(values)
        return None

    def _normalize_name(self, name: str) -> List[str]:
        """Generate normalized variants of an ingredient name."""
        candidates = []
        n = name
        for mod in INGREDIENT_MODIFIERS:
            n = re.sub(r'\b' + re.escape(mod) + r'\b', '', n, flags=re.IGNORECASE)
        n = " ".join(n.split()).strip()
        if n:
            candidates.append(n.lower())

        # Try removing "protein" specifically
        n2 = re.sub(r'\bprotein\b', '', name, flags=re.IGNORECASE).strip()
        n2 = " ".join(n2.split()).strip()
        if n2:
            candidates.append(n2.lower())

        # Try just the last significant word
        words = name.split()
        if len(words) > 1:
            candidates.append(words[-1].lower())

        return candidates

    def _map_via_food(self, ingredient_name: str) -> Optional[IngredientMapping]:
        """Try to map ingredient via food lookup (tiers 1-2)."""
        name_lower = ingredient_name.lower().strip()

        # Tier 1: exact match
        food_ids = self._food_name_to_ids.get(name_lower)
        tier = 1

        # Tier 2: normalized match
        if not food_ids:
            tier = 2
            for candidate in self._normalize_name(ingredient_name):
                food_ids = self._food_name_to_ids.get(candidate)
                if food_ids:
                    break

        if not food_ids:
            return None

        # Use first matched food entity
        food_id = food_ids[0]
        compounds = self._food_to_compounds.get(food_id, [])

        mapping = IngredientMapping(
            ingredient_name=ingredient_name,
            matched_entity_id=food_id,
            matched_entity_name=self._entity_names.get(food_id, ""),
            match_tier=tier,
        )

        for chem_id, meta_ids in compounds:
            mapping.compounds.append(CompoundMatch(
                foodatlas_id=chem_id,
                name=self._entity_names.get(chem_id, ""),
                pubchem_cid=self._get_pubchem_cid(chem_id),
                chebi_id=self._get_chebi_id(chem_id),
                concentration=self._get_concentration(meta_ids),
                match_path="food",
            ))

        return mapping

    def _map_via_chemical(self, ingredient_name: str) -> Optional[IngredientMapping]:
        """Try to map ingredient as a direct chemical (tier 3)."""
        name_lower = ingredient_name.lower().strip()
        chem_ids = self._chem_name_to_ids.get(name_lower)

        if not chem_ids:
            return None

        chem_id = chem_ids[0]
        mapping = IngredientMapping(
            ingredient_name=ingredient_name,
            matched_entity_id=chem_id,
            matched_entity_name=self._entity_names.get(chem_id, ""),
            match_tier=3,
            compounds=[CompoundMatch(
                foodatlas_id=chem_id,
                name=self._entity_names.get(chem_id, ""),
                pubchem_cid=self._get_pubchem_cid(chem_id),
                chebi_id=self._get_chebi_id(chem_id),
                concentration=None,
                match_path="chemical",
            )],
        )
        return mapping

    def _get_food_compounds(self, food_name: str) -> List[CompoundMatch]:
        """Get all compounds for a food name, or empty list if not found."""
        fids = self._food_name_to_ids.get(food_name.lower().strip(), [])
        if not fids:
            return []
        compounds = self._food_to_compounds.get(fids[0], [])
        result = []
        for chem_id, meta_ids in compounds:
            result.append(CompoundMatch(
                foodatlas_id=chem_id,
                name=self._entity_names.get(chem_id, ""),
                pubchem_cid=self._get_pubchem_cid(chem_id),
                chebi_id=self._get_chebi_id(chem_id),
                concentration=self._get_concentration(meta_ids),
                match_path="food",
            ))
        return result

    def _merge_primary_fallback(
        self, primary_name: str, *fallback_names: str
    ) -> List[CompoundMatch]:
        """Merge compounds from a primary (high-conc) entry and one-or-more
        fallback (broad-coverage) entries.

        Primary compounds keep their concentrations. Each subsequent fallback
        contributes only its non-overlapping compounds with concentration=None.
        Variadic to support version-aware fallbacks: e.g. for the milk family
        v3.2's rich fallback "milk" entity is renamed to "cow milk (liquid)"
        in v4.0, while v4.0 has a NEW richer "cow milk" entity. Listing both
        as fallbacks lets each version pick up whichever exists, without
        regressing the other.
        """
        primary = self._get_food_compounds(primary_name)
        seen_ids = {c.foodatlas_id for c in primary}
        for fallback_name in fallback_names:
            for compound in self._get_food_compounds(fallback_name):
                if compound.foodatlas_id not in seen_ids:
                    compound.concentration = None  # no conc for fallback-only
                    primary.append(compound)
                    seen_ids.add(compound.foodatlas_id)
        return primary

    def map_ingredient(self, ingredient_name: str) -> Optional[IngredientMapping]:
        """Map a single ingredient to its chemical compounds (4-tier strategy).

        Tier 0: Manual synonym lookup (highest priority, curated)
          - If synonym target has an ENTRY_PAIRS entry, uses primary+fallback
        Tier 1: Exact match on food lookup table
        Tier 2: Normalized match (strip modifiers) on food lookup
        Tier 3: Direct chemical lookup
        """
        # Tier 0: manual synonym (uses version-aware merged dict)
        synonym = self._ingredient_synonyms.get(ingredient_name.lower().strip())
        if synonym:
            # Check for primary+fallback pair (also version-aware)
            pair = self._entry_pairs.get(synonym)
            if pair:
                primary_name, *fallback_names = pair
                compounds = self._merge_primary_fallback(primary_name, *fallback_names)
                if compounds:
                    fb_label = "+".join(fallback_names)
                    return IngredientMapping(
                        ingredient_name=ingredient_name,
                        matched_entity_id=f"{primary_name}+{fb_label}",
                        matched_entity_name=f"{primary_name} (+{fb_label})",
                        match_tier=0,
                        compounds=compounds,
                    )

            # Standard synonym: try food first, then chemical
            result = self._map_via_food(synonym)
            if result:
                result.ingredient_name = ingredient_name
                result.match_tier = 0
                return result
            result = self._map_via_chemical(synonym)
            if result:
                result.ingredient_name = ingredient_name
                result.match_tier = 0
                return result

        # Tier 1-2: food lookup (also check for entry pairs on the resolved name)
        result = self._map_via_food(ingredient_name)
        if result:
            matched_name = result.matched_entity_name.lower()
            pair = self._entry_pairs.get(matched_name)
            if pair:
                primary_name, *fallback_names = pair
                compounds = self._merge_primary_fallback(primary_name, *fallback_names)
                if compounds:
                    result.compounds = compounds
                    result.matched_entity_name = f"{primary_name} (+{'+'.join(fallback_names)})"
            return result

        # Tier 3: chemical lookup
        return self._map_via_chemical(ingredient_name)

    def map_product(
        self, ingredient_list: str
    ) -> Dict[str, Optional[IngredientMapping]]:
        """Map all ingredients in a pipe-delimited list."""
        ingredients = [i.strip() for i in ingredient_list.split("|") if i.strip()]
        return {ing: self.map_ingredient(ing) for ing in ingredients}

    def get_all_pubchem_cids(self) -> Set[int]:
        """Get all unique PubChem CIDs in the FoodAtlas dataset."""
        cids = set()
        for eid, ext in self._entity_external.items():
            for cid in ext.get("pubchem_compound", []):
                cids.add(int(cid))
        return cids
