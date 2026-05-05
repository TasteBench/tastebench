# Nutrition feature selection

The four `category_group → columns` lists in
[`configs/cosine_dist/N*.yaml`](../configs/cosine_dist/) and
[`configs/l2_dist/N*.yaml`](../configs/l2_dist/) are not "all available
nutrients" — they are deliberately pruned. Standard US Nutrition Facts
panels can appear nearly identical between plant-based and animal
products while metabolite profiles diverge dramatically (Van Vliet et
al. 2021), so we include only features where label content predicts
sensory perception in *both* product types and exclude features whose
sensory mapping is broken by plant-based reformulation.

## Feature sets per category group

| Category group | Categories | Features |
|---|---|---|
| **Meat** (14) | Bacon, Bratwurst, Breakfast Sausages, Burgers, Chicken Strips, Breaded Chicken Filet, Unbreaded Chicken Breast, Deli Ham, Deli Turkey, Hot Dogs, Meatballs, Nuggets, Pulled Pork, Steak | Total Fat, Sodium, Protein, Dietary Fiber |
| **Non-Sweet Dairy** (6) | Butter, Cream Cheese, Sour Cream, Creamer, Milk, Barista Milk | Total Fat, Sodium |
| **Cheese** (2) | Cheddar Cheese, Mozzarella | Total Fat, Sodium, Total Carbohydrate |
| **Sweet Dairy** (2) | Ice Cream Hard Serve, Yogurt | Total Fat, Total Sugars |

## Why each included feature transfers across plant/animal products

- **Total Fat** — Fat perception is driven by texture (lubrication,
  viscosity), not chemoreception of specific fatty acids: orbitofrontal
  fat-responsive neurons fire equally to silicone and paraffin oils
  (Rolls et al. 2003), so label fat predicts mouthfeel regardless of
  source. Magwere et al. (2025) confirm this as the primary creaminess
  predictor across cow and plant-based milks (PLSR β=0.11).

- **Sodium** — Direct taste contributor (saltiness, flavor enhancement,
  bitterness suppression). Sensory role is source-agnostic even though
  the structural role differs (myofibrillar protein extraction in meat
  vs. pure flavoring in plant-based) (Liem et al. 2011).

- **Protein** (Meat only) — Protein content determines fibrous-network
  formation in high-moisture extrusion: Chiang et al. (2019) showed
  texturisation degree and sensory fibrousness scaling with wheat-gluten
  ratio; Godschalk-Broers et al. (2022) corroborate across 27 commercial
  meat analogs. Limitation: total protein alone misses protein-source
  effects.

- **Dietary Fiber** (Meat only) — Reflects methylcellulose
  concentration, used in essentially all major plant-based meats; its
  thermal-gelation property simulates animal-protein denaturation during
  cooking (Bakhsh et al. 2021).

- **Total Carbohydrate** (Cheese only) — Proxies the modified-starch
  matrix (tapioca, potato) that replaces casein for melt and stretch in
  plant-based cheese; starch-based formulations outperform protein-based
  ones sensorially (Short et al. 2021; Saraco & Blaxland 2020).

- **Total Sugars** (Sweet Dairy only) — Controls freezing-point
  depression, ice-crystal size, and melt rate in ice cream; primary
  liking driver in yogurt where it also masks plant-protein off-flavors.

## Why each excluded feature is excluded

- **Saturated / Mono / Poly Fat** — 74% of plant-based cheeses use
  coconut oil (mostly C12:0, m.p. 24°C) vs. animal saturated fat (C16:0
  / C18:0, m.p. 63–70°C); identical "Saturated Fat: 7g" can mean
  completely different melting behavior (Saraco & Blaxland 2020). MUFA
  is confounded with Total Fat (r=0.91); PUFA correlates with off-flavor
  oxidation.

- **Calories** — Linear combination of macros via Atwater factors
  (R²≈0.99). Menichetti et al. (2023) excluded for the same reason.

- **Cholesterol** — Exactly 0 mg in all plant-based products (zero
  variance, no discriminative information).

- **Trans Fat** — Near-zero variance after FDA's 2018 GRAS revocation.

- **Vitamin D** — Cashman (2024) reviewed fortification across food
  vehicles: "vitamin D addition...does not alter their sensory
  characteristics or overall acceptability." Pure fortification artifact.

- **Calcium** — Fortification artifact (tricalcium phosphate / calcium
  carbonate). At high levels causes grittiness — a defect engineered
  *against*, not encoded as similarity.

- **Iron** — Conflates two functionally different substances:
  soy-leghemoglobin (sensory-relevant, used in Impossible) and non-heme
  supplemental iron (sensory-inert).

- **Potassium** — No consistent sensory role; elevation in plant-based
  products is incidental to legume/vegetable ingredients.

- **Protein** (Non-Sweet Dairy, Sweet Dairy, Cheese) — Higher protein
  *worsens* sensory similarity in plant dairy: Alsado et al. (2023)
  showed oat milk fortified with oat protein scored "grainy, chalky,
  sandy"; commercial plant-based cheeses deliberately keep protein low
  (0.11–3% vs. 25% in dairy Cheddar; Saraco & Blaxland 2020).

- **Total Carbohydrate** (Non-Sweet Dairy) — In dairy contexts variation
  reflects raw-material composition (lactose vs. starch hydrolysis), not
  intentional sensory ingredients; Magwere et al. (2025) PLSR did not
  identify it as a significant sensory predictor.

- **Sodium** (Sweet Dairy) — Negligible sensory role at typical levels
  (50–100 mg/100g) in sweet products.

## Citations

1. Van Vliet et al. (2021), *Sci. Rep.* 11(1):13828. https://doi.org/10.1038/s41598-021-93100-3
2. Rolls, Verhagen & Kadohisa (2003), *J. Neurophysiol.* 90(6):3711-3724. https://doi.org/10.1152/jn.00515.2003
3. Chiang, Loveday, Hardacre & Parker (2019), *Food Structure* 19:100102. https://doi.org/10.1016/j.foostr.2018.11.002
4. Godschalk-Broers, Sala & Scholten (2022), *Foods* 11(15):2227. https://doi.org/10.3390/foods11152227
5. Bakhsh et al. (2021), *Foods* 10(3):560. https://doi.org/10.3390/foods10030560
6. Magwere et al. (2025), *J. Food Sci.* https://doi.org/10.1111/1750-3841.70370
7. Alsado, Lopez-Aldana, Chen & Wismer (2023), *Foods* 12(22):4097. https://doi.org/10.3390/foods12224097
8. Short, Kinchla & Nolden (2021), *Foods* 10(4):725. https://doi.org/10.3390/foods10040725
9. Saraco & Blaxland (2020), *Br. Food J.* 122(12):3727-3740. https://doi.org/10.1108/BFJ-11-2019-0825
10. Menichetti, Ravandi, Mozaffarian & Barabási (2023), *Nat. Commun.* 14:2312. https://doi.org/10.1038/s41467-023-37457-1
11. Cashman (2024), *J. Steroid Biochem. Mol. Biol.* https://doi.org/10.1016/j.jsbmb.2024.106494
12. Liem, Miremadi & Keast (2011), *Nutrients* 3(6):694-711. https://doi.org/10.3390/nu3060694
