# Thyroid Disease: Medical Grounding for Experiment 2

> **Audience:** A software/ML engineer with no medical background.
> **Purpose:** Give you enough rigorous, correct physiology to understand *why* the thyroid-disease classifier makes the predictions it makes — so that when you later read SHAP explanations, you can sanity-check them against real medicine instead of taking them on faith.
> **Dataset:** Garavan Institute "thyroid0387" archive (Sydney, Australia), 9,172 patient records collected roughly 1984–1987.
> **Task:** Classify each patient into exactly one of 7 grouped clinical classes.

This document contains **no code**. It is the medical reference layer for the whole experiment.

---

## Table of Contents

1. [The thyroid gland and what its hormones do](#1-the-thyroid-gland-and-what-its-hormones-do)
2. [The HPT axis and the TSH negative-feedback loop](#2-the-hpt-axis-and-the-tsh-negative-feedback-loop)
3. [The core diagnostic-logic table](#3-the-core-diagnostic-logic-table)
4. [The lab features, one by one](#4-the-lab-features-one-by-one)
5. [The 7 target classes and their pathophysiology](#5-the-7-target-classes-and-their-pathophysiology)
6. [Etiology: why each condition happens](#6-etiology-why-each-condition-happens)
7. [The clinical-history binary flags and why they matter](#7-the-clinical-history-binary-flags-and-why-they-matter)
8. [Connecting the medicine to the ML model](#8-connecting-the-medicine-to-the-ml-model)
9. [Limitations and caveats of this dataset](#9-limitations-and-caveats-of-this-dataset)
10. [Glossary](#10-glossary)

---

## 1. The thyroid gland and what its hormones do

### 1.1 The gland

The **thyroid** is a small, butterfly-shaped endocrine gland in the front of the neck, wrapped around the trachea (windpipe) just below the larynx (voice box). "Endocrine" means it secretes hormones directly into the bloodstream rather than through a duct.

Despite weighing only ~20–25 grams, the thyroid is the body's **master metabolic thermostat**. It does one big job: it produces thyroid hormone, which sets the basal metabolic rate — the speed at which essentially every cell in the body burns energy.

### 1.2 The two hormones: T4 and T3

The thyroid makes two iodine-containing hormones:

- **T4 (thyroxine)** — contains **four** iodine atoms. This is the *main* product of the gland (~90–95% of output). T4 is essentially a **prohormone**: relatively inactive on its own, it acts as a stable circulating reservoir.
- **T3 (triiodothyronine)** — contains **three** iodine atoms. T3 is the **biologically active** hormone — roughly 3–4× more potent than T4 at the receptor. The thyroid secretes a small amount of T3 directly, but **~80% of the body's active T3 is made outside the thyroid**, by enzymes (deiodinases) that strip one iodine atom off T4 in peripheral tissues (liver, kidney, etc.).

This T4 → T3 conversion step is important for the ML model: it is the point at which **non-thyroidal illness** disrupts the system (see §5.3). When a person is seriously ill, the body down-regulates T4→T3 conversion, so T3 falls even when the thyroid itself is fine.

> **Iodine note.** Both hormones are built from the amino acid tyrosine plus dietary **iodine**. No iodine → no thyroid hormone. This is why iodine deficiency is a classic global cause of hypothyroidism and goitre (§6.2).

### 1.3 What thyroid hormone actually does

Thyroid hormone receptors sit inside the nucleus of nearly every cell. When T3 binds, it changes gene expression and ramps up metabolic machinery. Practical effects of having **enough** thyroid hormone:

- **Basal metabolic rate**: sets resting energy/heat production. More hormone → faster metabolism, more heat.
- **Heart**: increases heart rate and contractility.
- **Nervous system**: needed for alertness, normal reflexes, and (critically in infancy) brain development.
- **Growth & development**: essential for normal childhood growth and skeletal maturation.
- **GI tract**: sets gut motility (how fast food moves through).
- **Thermoregulation**: governs tolerance to heat and cold.

You can predict almost every symptom of thyroid disease by asking "what happens if the body's metabolism runs **too slow** or **too fast**?"

| System | **Too little** hormone (hypothyroid) | **Too much** hormone (hyperthyroid) |
|---|---|---|
| Metabolism / weight | Slow; weight gain | Fast; weight loss despite eating |
| Temperature | Cold intolerance | Heat intolerance, sweating |
| Heart | Slow heart rate (bradycardia) | Fast heart rate (tachycardia), palpitations |
| Energy / mood | Fatigue, sluggishness, depression | Anxiety, restlessness, insomnia |
| GI | Constipation | Diarrhea / frequent stools |
| Skin / hair | Dry skin, hair loss | Warm, moist skin; fine tremor |
| Reflexes | Slow | Brisk |

Hold onto this mental model: **hypothyroid = everything slows down; hyperthyroid = everything speeds up.** The lab numbers below are just the quantitative fingerprint of that same idea.

---

## 2. The HPT axis and the TSH negative-feedback loop

This section is **the single most important idea** in the document. The inverse relationship between TSH and thyroid hormone is the physics the ML model exploits, and almost every SHAP explanation you'll read later traces back to it.

### 2.1 Three glands, one control loop

Thyroid hormone is not controlled by the thyroid alone. It is governed by a three-level command chain called the **Hypothalamic–Pituitary–Thyroid (HPT) axis**:

```
   HYPOTHALAMUS  (in the brain)
        │  releases TRH (thyrotropin-releasing hormone)
        ▼
   PITUITARY GLAND  (base of the brain)
        │  releases TSH (thyroid-stimulating hormone)
        ▼
   THYROID GLAND  (neck)
        │  releases T4 and T3
        ▼
   BODY TISSUES  (metabolism)
        │
        └────── negative feedback ──────► back UP to hypothalamus & pituitary
```

- The **hypothalamus** senses the body's need and secretes **TRH**.
- TRH tells the **pituitary** to secrete **TSH**.
- **TSH** is the gas pedal for the thyroid: it tells the thyroid to make and release T4/T3, and to grow.
- Circulating **T4/T3** then feed **back** to the pituitary and hypothalamus and **suppress** further TRH/TSH release.

### 2.2 The negative-feedback loop (the inverse relationship)

This is a thermostat. Think of TSH as a furnace controller watching the "temperature" (blood thyroid hormone):

- **Thyroid hormone too LOW** → pituitary senses the deficit → **pumps out MORE TSH** to flog the thyroid into producing more. → **TSH rises.**
- **Thyroid hormone too HIGH** → pituitary senses the excess → **shuts off TSH** to let levels fall. → **TSH drops (suppressed).**

Therefore, **when the problem is in the thyroid gland itself**, TSH and thyroid hormone move in **opposite directions**:

- **High TSH ⇄ low thyroid hormone** (gland is failing; pituitary screaming for more)
- **Low TSH ⇄ high thyroid hormone** (gland overproducing; pituitary backing off)

> **Why TSH is the single best screening test.** Because of the feedback loop, TSH is exquisitely sensitive: a small change in thyroid hormone causes a large, logarithmic change in TSH. The pituitary often detects a problem before T4/T3 even leave the normal range. That is exactly why a tree-based ML model will lean so heavily on TSH — it carries the most information per measurement. This sensitivity also makes the typical reference range of TSH **right-skewed**, which is why TSH is frequently log-transformed before modeling.

### 2.3 The crucial exception: central (secondary) disease

The inverse relationship only holds when the **thyroid gland** is the broken part and the pituitary is healthy. If the **pituitary or hypothalamus** is broken (so it can't send the TSH signal), the logic flips:

- **Secondary / central hypothyroidism**: the pituitary fails → **TSH is low (or inappropriately normal)** *and* thyroid hormone is **low**. Both are low together. This is the opposite of the usual "high TSH = hypothyroid" rule, and it's why a model must consider T4/FTI alongside TSH, not TSH alone. The `hypopituitary` flag (§7) is the dataset's marker for this.

So the full reasoning needs **two axes** — TSH *and* a measure of actual thyroid hormone (FTI / TT4 / T3). That two-dimensional logic is captured in the table below.

---

## 3. The core diagnostic-logic table

This is the clinical reasoning the model is implicitly learning. Read each row as "if labs look like this, the patient is probably in this state." (Arrows: ↑ high, ↓ low, → normal.)

| TSH | FTI / Free T4 | T3 | Interpretation | Maps to class |
|---|---|---|---|---|
| ↑ high | ↓ low | ↓/→ | **Primary hypothyroidism** — thyroid is failing, pituitary compensating loudly | `hypothyroid` |
| ↑ high | → normal | → | **Subclinical / compensated hypothyroidism** — gland struggling but still keeping hormone normal at the cost of high TSH | `hypothyroid` |
| ↓ low | ↑ high | ↑/→ | **Primary hyperthyroidism** (e.g. Graves', toxic nodule) — gland overproducing, pituitary suppressed | `hyperthyroid` |
| ↓ low | → normal | ↑ | **T3-toxicosis / subclinical hyperthyroidism** — excess is mostly T3, or early/mild overactivity | `hyperthyroid` |
| ↓ low | ↓ low | ↓ | **Central (secondary) hypothyroidism** OR **non-thyroidal illness** — pituitary not signalling, or systemic illness suppressing the axis | `hypothyroid` (central) or `nonthyroidal_illness` |
| → normal | ↓ or → | ↓ | **Non-thyroidal illness ("sick euthyroid")** — sick patient, low T3 with otherwise unremarkable TSH | `nonthyroidal_illness` |
| → normal | → normal | → | **Euthyroid (normal)** — no thyroid problem requiring comment | `negative` |
| → / variable | ↑ high TT4 but → free | → | **Binding-protein anomaly** — *total* hormone shifted by carrier protein, *free* (active) hormone normal | `binding_protein` |

### 3.1 How to read this table

- **Two-dimensional rule of thumb:** look at TSH first (the most sensitive dial), then confirm direction with FTI/T4/T3.
- The dangerous-but-important rows are the ones where TSH and hormone are **both low** — those break the simple inverse rule and require the model to use *context* (the `sick`, `hypopituitary`, `psych` flags; age; etc.).
- The **binding-protein** row is the trap: **TT4 (total T4) can be wildly abnormal while the patient is perfectly healthy**, because most T4 in blood is bound to carrier proteins and only the tiny *free* fraction is active. That's the whole reason FTI and T4U exist (§4).

---

## 4. The lab features, one by one

For each lab the model uses, here is what it physically measures, a rough adult reference range, and how it shifts in hypo- vs hyper-thyroidism.

> **Caution on ranges.** Reference ranges depend on the assay, the lab, the era, and the population. The numbers below are typical *modern* adult ranges for intuition only. This 1980s dataset used older assays, so the *absolute* numbers may differ — but the *direction* of every shift is timeless and is what matters for interpretation.

### 4.1 TSH — Thyroid-Stimulating Hormone

- **What it is:** The pituitary's command signal to the thyroid (§2). Measured in mIU/L.
- **Typical range:** ~0.4–4.0 mIU/L.
- **Hypothyroid (primary):** **↑ HIGH** — pituitary shouting at a failing gland. Often the *first* and most dramatic abnormality.
- **Hyperthyroid:** **↓ LOW / suppressed** — pituitary shuts off because hormone is excessive.
- **Why it's #1:** Most sensitive single test; logarithmic response; usually the strongest predictor a model will find. Skewed distribution → often log-transformed.

### 4.2 TT4 — Total Thyroxine (total T4)

- **What it is:** The **total** amount of T4 in blood — both the protein-**bound** (~99.95%) and the **free** (~0.05%) fractions added together. Measured in µg/dL (or nmol/L).
- **Typical range:** ~5–12 µg/dL.
- **Hypothyroid:** **↓ LOW.**
- **Hyperthyroid:** **↑ HIGH.**
- **The catch:** Because TT4 includes the bound fraction, it is **distorted by binding-protein levels**. High binding protein (e.g. pregnancy, estrogen) raises TT4 even though the patient is euthyroid; low binding protein lowers it. So TT4 *alone* can mislead — which is the entire reason for T4U and FTI.

### 4.3 T3 — Triiodothyronine

- **What it is:** The active hormone (§1.2). Usually measured as total T3 here. ng/dL (or nmol/L).
- **Typical range:** ~80–200 ng/dL.
- **Hypothyroid:** ↓ low (but T3 is often *preserved* until late — the body prioritizes keeping T3 up, so T3 is a **less sensitive** marker of mild hypothyroidism than TSH/T4).
- **Hyperthyroid:** **↑ HIGH** — and in **"T3-toxicosis"** T3 rises while T4 is still normal, so T3 can be the *only* abnormal hormone. This is exactly the `hyperthyroid` sub-case the model must catch from T3.
- **Non-thyroidal illness:** **↓ LOW** — the hallmark of "sick euthyroid": peripheral T4→T3 conversion is throttled, so T3 drops first and most. **Low T3 with normalish TSH in a `sick` patient is the signature of the `nonthyroidal_illness` class.**

### 4.4 T4U — Thyroxine Uptake (T3/T4 resin uptake)

- **What it is:** An **indirect measure of the available binding-protein capacity** in the blood — essentially "how many empty seats are on the carrier proteins." It does **not** measure hormone directly; it's a correction factor. (Historically reported as a "resin uptake" ratio.)
- **Why it exists:** To let you back out the binding-protein distortion from TT4. T4U moves in the opposite direction to the *amount of unoccupied binding protein*: when binding protein is high (more empty seats), the uptake reading is **low**; when binding protein is low, uptake is **high**.
- **Use:** Combined with TT4 to compute FTI (below). On its own it tells the model "is this a binding-protein situation?" — which is precisely the `binding_protein` class signal.

### 4.5 FTI — Free Thyroxine Index

- **What it is:** A **computed** estimate of free (active) T4, classically **FTI = TT4 × T4U**. It corrects total T4 for the binding-protein status, approximating what a direct free-T4 measurement would show. (Modern labs measure free T4 directly; in this era FTI was the standard workaround.)
- **Typical range:** unit-less index, lab-specific.
- **Hypothyroid:** **↓ LOW.**
- **Hyperthyroid:** **↑ HIGH.**
- **Binding-protein anomaly:** **→ NORMAL** even when TT4 is abnormal. **This is the killer discriminator:** if TT4 is high but FTI is normal, the patient is euthyroid with a binding-protein quirk (`binding_protein`), *not* hyperthyroid. A good model should weight FTI over raw TT4 to avoid that mistake.

### 4.6 Age

- **What it is:** Patient age in years.
- **Why it matters clinically:** Thyroid disease incidence rises with age (especially autoimmune hypothyroidism in older women). Reference ranges (notably TSH) drift upward with age. Subclinical disease and nodular goitre are more common in the elderly. Age also helps the model contextualize ambiguous labs.
- **Data note:** Watch for impossible values (e.g. ages > 120 or negative) — this dataset is known to contain a few data-entry errors in age, which should be cleaned.

### 4.7 Sex

- **What it is:** Biological sex (M/F).
- **Why it matters:** Autoimmune thyroid disease (both Hashimoto's hypothyroidism and Graves' hyperthyroidism) is **far more common in women** (roughly 5–8× for many conditions). Female sex also ties into pregnancy and estrogen effects on binding protein (§5.4, §7). So sex is a genuine epidemiological prior, not noise.

---

## 5. The 7 target classes and their pathophysiology

The model assigns each patient to **exactly one** of these 7 grouped classes. The original Garavan archive used dozens of single-letter diagnosis codes; the experiment folds them into these 7 buckets. Class names are used **verbatim** below.

### 5.0 `negative` — no thyroid condition requiring comment (~74%)

- **What it means:** Euthyroid — the thyroid is functioning normally, or at least nothing in the workup warranted a diagnostic comment.
- **Lab signature:** TSH, FTI, TT4, T3 all in range.
- **ML reality:** This is the **majority class (~74%)**. It dominates accuracy and makes the problem **heavily imbalanced**. A model that predicts "negative" for everyone scores ~74% accuracy while being clinically useless — so you must judge the model on **per-class recall / F1 / macro-averaged metrics**, not raw accuracy.

### 5.1 `hypothyroid` — underactive thyroid

- **Groups:** primary, compensated (subclinical), and secondary (central) hypothyroidism.
- **What it means:** Not enough thyroid hormone → metabolism slows down (cold, fatigue, weight gain, constipation, bradycardia).
- **Lab signature:**
  - **Primary:** **TSH ↑ high, FTI/TT4 ↓ low.** The classic inverse pattern.
  - **Compensated / subclinical:** **TSH ↑ high, FTI/TT4 → normal** — gland is failing but still keeping hormone normal at the cost of elevated TSH.
  - **Secondary / central:** **TSH ↓ low or normal, FTI/TT4 ↓ low** — pituitary not signalling (see `hypopituitary` flag).
- **Strongest features:** TSH (very high), then low FTI/TT4.

### 5.2 `hyperthyroid` — overactive thyroid

- **Groups:** hyperthyroid, T3 toxicosis, toxic (nodular) goitre — **plus** the antithyroid-treatment codes **O/P/Q** folded in (patients being treated *for* hyperthyroidism).
- **What it means:** Too much thyroid hormone → metabolism speeds up (heat intolerance, weight loss, palpitations, tremor, anxiety, diarrhea).
- **Lab signature:** **TSH ↓ suppressed, FTI/TT4 ↑ high and/or T3 ↑ high.** In **T3-toxicosis**, T3 may be the only elevated hormone.
- **Note on folding in O/P/Q:** Patients on **antithyroid medication** (e.g. carbimazole/methimazole) are, by definition, hyperthyroid patients under treatment — so the `on_antithyroid_medication` flag is a strong pointer to this class even if labs are currently being normalized by the drug.
- **Strongest features:** suppressed TSH, high T3, high FTI; `on_antithyroid_medication`, `query_hyperthyroid`.

### 5.3 `nonthyroidal_illness` — sick euthyroid syndrome (code K)

- **What it means:** This is **NOT a thyroid disease at all.** It is the thyroid axis's *response* to a serious **non-thyroid** illness (sepsis, major surgery, trauma, starvation, MI, cancer, ICU-level illness). The thyroid gland is intrinsically fine; the *system* down-regulates to conserve energy during severe stress.
- **Why it happens (pathophysiology):**
  - Severe illness suppresses peripheral **T4 → T3 conversion** (deiodinase activity drops), so **T3 falls first and most** ("low-T3 syndrome").
  - Inflammatory cytokines and stress hormones suppress the hypothalamus/pituitary, so **TSH is often normal or low** instead of rising the way it would in true hypothyroidism.
  - In the sickest patients, T4 and FTI fall too — and low T4 in critical illness is a **poor-prognosis** marker.
- **Lab signature:** **T3 ↓ low**, **TSH → normal or ↓ low**, T4/FTI normal-to-low. The tell-tale combination is **low T3 + non-elevated TSH in a sick patient** — which would be contradictory in true thyroid disease (low hormone *should* drive TSH up).
- **Strongest features:** **low T3**, the **`sick` flag**, normal/low TSH. The `sick` flag is the key contextual signal that separates this from real hypothyroidism.
- **Why it's its own class:** Because treating these patients with thyroid hormone is usually wrong — the labs are a *consequence* of illness, not a thyroid disorder. The model must learn to **not** call them hypothyroid.

### 5.4 `binding_protein` — thyroid-hormone binding-protein anomalies (codes I/J)

- **What it means:** The **carrier proteins** that ferry thyroid hormone in blood (chiefly **TBG — thyroxine-binding globulin**, plus transthyretin and albumin) are increased or decreased. This shifts the **total** hormone (TT4) up or down, but the body keeps the **free (active)** hormone normal, so the patient is **clinically euthyroid**.
- **Why it happens:**
  - **Increased TBG → high TT4:** **pregnancy**, **estrogen** (oral contraceptives, HRT), certain hereditary conditions, some liver disease.
  - **Decreased TBG → low TT4:** androgens, steroids, nephrotic syndrome (protein loss), severe illness, hereditary TBG deficiency.
- **Lab signature:** **TT4 abnormal (high or low), but FTI → normal and TSH → normal.** T4U is the giveaway — it shifts to reflect the altered binding capacity.
- **Strongest features:** the **TT4-vs-FTI mismatch** (abnormal total, normal index), T4U, and `pregnant`.
- **Teaching point:** This class is the concrete reason FTI/T4U exist. It is also the class most likely to be *misclassified as hyper- or hypo-thyroid* by a naive model that trusts TT4 — so it's a great test of whether the model learned to prefer FTI.

### 5.5 `replacement_therapy` — on thyroid hormone replacement (codes L/M/N)

- **What it means:** The patient is **taking thyroid hormone medication** (levothyroxine = synthetic T4, i.e. "thyroxine"), usually because they had hypothyroidism or thyroid removal. The sub-codes distinguish **consistent** (well-replaced), **under-replaced**, and **over-replaced**.
- **Why it's its own class:** Their lab values reflect the **dose**, not natural gland function — and crucially, the **`on_thyroxine` history flag is an almost-deterministic giveaway.** A patient on thyroxine is, by definition, on replacement therapy.
- **Lab signature (depends on dosing quality):**
  - Well-replaced: labs roughly normal.
  - **Under-replaced:** looks **hypothyroid** (TSH ↑, FTI ↓) — not enough dose.
  - **Over-replaced:** looks **hyperthyroid** (TSH ↓, FTI ↑) — too much dose.
- **Strongest features:** **`on_thyroxine` flag (dominant)**, supported by `query_on_thyroxine`, plus labs to judge over/under-replacement.
- **ML note:** Because `on_thyroxine` is so predictive, expect SHAP to show it dominating this class. That's medically correct — but watch for **leakage-like behavior**: the flag essentially *defines* the class, so the model isn't really "diagnosing" anything subtle here.

### 5.6 `discordant_results` — discordant assays / elevated TBG / elevated thyroid hormones (codes R/S/T)

- **What it means:** A grab-bag class for records where the **assay results don't agree with each other** or with the clinical picture, including elevated TBG and "elevated thyroid hormones" findings that don't cleanly fit hyper/hypo. "Discordant" literally means the numbers tell conflicting stories (e.g. a TSH that doesn't match the T4, or results flagged as inconsistent).
- **Why it happens:**
  - Assay **interference** (antibodies in the patient's blood fooling the test, e.g. heterophile antibodies).
  - **Binding-protein** extremes (elevated TBG) skewing totals.
  - Timing artifacts (recent dose, recovering illness, transient states).
  - Genuine biological in-between states the simple model can't reconcile.
- **Lab signature:** **internally inconsistent** — that's the defining feature, not any single value.
- **ML note:** This is typically a **small, noisy class** and is one of the hardest to predict. Don't be alarmed if recall here is poor; the class is defined by *inconsistency*, which is intrinsically hard to learn.

### 5.7 Summary of which codes fold into which class

| Class | Original code(s) folded in | One-line essence |
|---|---|---|
| `negative` | (no diagnosis comment) | Euthyroid / normal |
| `hypothyroid` | primary / compensated / secondary hypothyroid | Too little hormone |
| `nonthyroidal_illness` | K | Sick patient, not a thyroid disease |
| `binding_protein` | I, J | Carrier-protein shift; free hormone normal |
| `replacement_therapy` | L, M, N | On thyroxine; consistent/under/over |
| `discordant_results` | R, S, T | Inconsistent assays / elevated TBG / elevated hormones |
| `hyperthyroid` | hyperthyroid, T3-toxic, toxic goitre + O, P, Q | Too much hormone (incl. antithyroid-treated) |

---

## 6. Etiology: why each condition happens

Understanding *cause* helps you reason about which history flags should matter.

### 6.1 Causes of hypothyroidism

- **Hashimoto's thyroiditis (autoimmune):** The most common cause in iodine-replete countries. The immune system makes antibodies that attack and gradually destroy the thyroid. Strongly female-predominant; rises with age. Often shows up first as **subclinical/compensated** hypothyroidism (high TSH, normal T4).
- **Iodine deficiency:** Historically the #1 global cause. No iodine → can't build hormone → low T4 → high TSH → the gland enlarges trying to compensate (**goitre**, see `goitre` flag).
- **Iatrogenic (treatment-caused):** thyroid surgery (`thyroid_surgery`), radioactive iodine **I-131** ablation (`I131_treatment`), or certain drugs — notably **lithium** (`lithium`) which blocks hormone release, and amiodarone.
- **Central (secondary):** pituitary or hypothalamic failure (`hypopituitary`) → low TSH signal → low hormone. Rare but breaks the usual TSH logic.

### 6.2 Causes of hyperthyroidism

- **Graves' disease (autoimmune):** The most common cause. Antibodies *stimulate* the TSH receptor, constantly flogging the thyroid to overproduce — like a stuck-on gas pedal. Female-predominant; classic extra signs are bulging eyes (exophthalmos) and goitre.
- **Toxic nodular goitre / toxic adenoma:** One or more thyroid nodules become autonomous and pump out hormone regardless of TSH. More common in older patients and in iodine-deficient regions.
- **Thyroiditis (transient):** inflammation can dump stored hormone into the blood, causing temporary hyperthyroidism.
- **Iatrogenic / drug-induced:** too much thyroxine (over-replacement → looks like `replacement_therapy` over-dosed), iodine load, amiodarone.

### 6.3 Why non-thyroidal illness ("sick euthyroid") happens

Covered in §5.3. The key idea for etiology: it is an **adaptive (or maladaptive) systemic response to severe illness**, not thyroid pathology. The body throttles metabolism during crisis by reducing T4→T3 conversion and dampening the HPT axis. The cause is *elsewhere* (infection, surgery, trauma); the thyroid labs are collateral.

### 6.4 Why binding-protein anomalies happen

Covered in §5.4. Causes that **raise** TBG: pregnancy, estrogen/oral contraceptives, hereditary excess, hepatitis. Causes that **lower** TBG: androgens, glucocorticoids, nephrotic syndrome, severe illness, hereditary deficiency. The hormone-*producing* machinery is normal; only the *transport* layer changed.

### 6.5 Why replacement-therapy patients form their own class

They are a distinct population: their thyroid function is being *driven by a pill*, not by their gland. Their labs reflect dosing accuracy, and they are flagged by `on_thyroxine`. Clinically, the question for them isn't "do they have thyroid disease?" (they're being treated for one) but "is the dose right?" — hence the consistent/under/over sub-codes. Mixing them with naturally-euthyroid or naturally-hypothyroid patients would confuse the model, so they get their own bucket.

### 6.6 What causes discordant assay results

Covered in §5.6: assay interference (antibodies), binding-protein extremes, timing/transient states, and genuine biological ambiguity. The unifying theme is that the **numbers contradict each other**.

---

## 7. The clinical-history binary flags and why they matter

These are 0/1 indicators recorded from the patient's history/referral. Several are **near-deterministic shortcuts** to a class — the model will (correctly) lean on them hard. Knowing which flags *cause* or *imply* which state lets you validate SHAP.

| Flag | What it means | Why it matters / which class it points to |
|---|---|---|
| `on_thyroxine` | Patient is taking thyroxine (levothyroxine) | **Strongest pointer to `replacement_therapy`.** Taking thyroxine ≈ being on replacement. |
| `query_on_thyroxine` | Uncertainty/query about whether on thyroxine | Soft version of above; supports `replacement_therapy`. |
| `on_antithyroid_medication` | Taking antithyroid drugs (e.g. carbimazole) | Strong pointer to **`hyperthyroid`** — you only take these if you're hyperthyroid. (Codes O/P/Q.) |
| `sick` | Patient is currently ill | **Key context for `nonthyroidal_illness`.** Low T3 + sick ⇒ sick-euthyroid, not hypothyroid. |
| `pregnant` | Patient is pregnant | Raises TBG → high TT4 with normal FTI ⇒ pushes toward **`binding_protein`**; also alters reference ranges. Female-only. |
| `thyroid_surgery` | History of thyroid surgery | Removed/reduced gland ⇒ tends toward **hypothyroid / replacement**. |
| `I131_treatment` | Received radioactive iodine (I-131) | Ablates thyroid tissue ⇒ commonly causes **hypothyroidism** afterward; also used to treat hyperthyroidism. |
| `query_hypothyroid` | Clinician suspects hypothyroidism | Referral suspicion → prior probability toward **`hypothyroid`**. |
| `query_hyperthyroid` | Clinician suspects hyperthyroidism | Referral suspicion → prior probability toward **`hyperthyroid`**. |
| `lithium` | On lithium (psychiatric drug) | Lithium blocks thyroid hormone release ⇒ can **cause hypothyroidism / goitre**. |
| `goitre` | Enlarged thyroid present | Non-specific (occurs in iodine deficiency, Hashimoto's, Graves', nodular disease) but signals thyroid pathology of *some* kind. |
| `tumor` | Thyroid tumor/nodule | Nodules can be hot (overproducing → hyper) or non-functional; flags structural disease. |
| `hypopituitary` | Pituitary underactivity | Marker of **central/secondary** disease — the case where low TSH accompanies low hormone (breaks the inverse rule). |
| `psych` | Psychiatric illness/symptoms | Thyroid disease mimics psychiatric symptoms (hypo→depression, hyper→anxiety); also flags patients on psychotropics (e.g. lithium) and overlaps with `sick`-type confounding. |

### 7.1 Important modeling caveat about flags

Many of these flags are **descriptors of treatment or referral suspicion**, not independent diagnostic evidence. `on_thyroxine` essentially *defines* `replacement_therapy`; `on_antithyroid_medication` essentially *defines* the treated-`hyperthyroid` subgroup. That makes them **extremely predictive but also a little circular** — the model is partly "reading the answer off the chart" rather than reasoning from physiology. This is fine for accuracy, but be honest about it when interpreting feature importance: high SHAP on `on_thyroxine` for `replacement_therapy` is *expected and trivially correct*, not a deep insight.

---

## 8. Connecting the medicine to the ML model

This section is the bridge: **which features should dominate which class, and why** — your checklist for sanity-checking SHAP outputs.

### 8.1 Expected top features per class

| Class | Features that *should* be most predictive | Physiological reason |
|---|---|---|
| `negative` | All labs in-range; absence of flags | Nothing abnormal to flag. |
| `hypothyroid` | **TSH ↑** (dominant), **FTI/TT4 ↓**, `query_hypothyroid`, `thyroid_surgery`, `I131_treatment`, `lithium` | Failing gland → pituitary drives TSH up; treatments/drugs that destroy gland. |
| `hyperthyroid` | **TSH ↓ suppressed**, **T3 ↑**, **FTI ↑**, `on_antithyroid_medication`, `query_hyperthyroid` | Overproduction suppresses TSH; antithyroid meds imply the diagnosis. |
| `nonthyroidal_illness` | **T3 ↓** + **`sick`** + TSH **not** elevated | Illness throttles T4→T3 conversion and dampens TSH; context flag is essential. |
| `binding_protein` | **TT4 abnormal but FTI normal** (the mismatch), **T4U**, `pregnant` | Carrier-protein shift moves total but not free hormone. |
| `replacement_therapy` | **`on_thyroxine`** (dominant), `query_on_thyroxine`, labs for over/under-dose | Being on thyroxine ≈ being on replacement. |
| `discordant_results` | No clean single feature; conflicting lab patterns | Class defined by inconsistency — hard to learn. |

### 8.2 How to read SHAP against physiology

When you later inspect SHAP values, ask: **does the feature the model leaned on make physiological sense for that class?**

- ✅ *"SHAP says **high TSH** pushed this patient toward `hypothyroid`."* — **Correct.** A failing thyroid makes the pituitary pump out more TSH (§2.2). This is the textbook signal.
- ✅ *"SHAP says **low T3 + `sick`=1** pushed toward `nonthyroidal_illness`."* — **Correct.** Exactly the sick-euthyroid signature (§5.3).
- ✅ *"SHAP says **high TT4 but normal FTI** pushed toward `binding_protein`."* — **Correct and sophisticated.** It means the model learned to trust *free* over *total* hormone (§4.5).
- ✅ *"SHAP says **`on_thyroxine`=1** pushed toward `replacement_therapy`."* — **Correct but trivial** — the flag nearly defines the class (§7.1).
- 🚩 *"SHAP says **high TT4** alone pushed toward `hyperthyroid`** despite a normal FTI and normal TSH."* — **Suspicious.** That could be a binding-protein case the model misread; investigate. A high *total* T4 with normal *free* index and normal TSH is **not** hyperthyroidism.
- 🚩 *"SHAP says **low TSH** pushed toward `hyperthyroid`** in a patient flagged `hypopituitary`/`sick`."* — **Suspicious.** Low TSH there may mean central hypothyroidism or sick-euthyroid, not overactivity (§2.3, §5.3).

### 8.3 Modeling consequences of the physiology

- **Class imbalance:** `negative` ~74%; several classes (`discordant_results`, `binding_protein`, central hypothyroid) are tiny. Use **macro-F1 / per-class recall**, stratified splits, and consider resampling or class weights. Raw accuracy is misleading.
- **TSH skew:** TSH spans orders of magnitude and is right-skewed (logarithmic feedback). A **log transform** usually helps linear models; tree models are scale-invariant but still benefit from sane handling of outliers.
- **Engineered features carry the physiology:** **FTI = TT4 × T4U** already encodes the binding-protein correction. Keeping FTI (and the TT4-vs-FTI relationship) lets the model express §4.5 reasoning directly.
- **Missing data is informative:** In this dataset, whether a test (e.g. T3, TBG) was even *ordered* correlates with the clinical question. Missingness is **not** random — handle it thoughtfully (it can leak the label).
- **Flag leakage:** `on_thyroxine` / `on_antithyroid_medication` are near-labels for their classes. Expect them to dominate; don't over-interpret that as the model "understanding" thyroid disease.

---

## 9. Limitations and caveats of this dataset

Be explicit about these when reporting results — they bound how far any conclusion generalizes.

- **Age and single institution.** Collected ~1984–1987 at one center (Garavan Institute, Sydney). **Assay methods, reference ranges, and reporting conventions have changed substantially** since then. Absolute lab values are **not** directly comparable to modern labs — only the *directions* of shifts are timeless.
- **Population bias.** A single Australian referral population from the 1980s. Iodine status, demographics, ethnicity mix, and disease prevalence differ from today's global populations. Models trained here may **not transfer** to other settings.
- **Label provenance.** Diagnoses are single-letter codes assigned at the time, not adjudicated against modern criteria. The 7-class grouping is an **analyst's simplification**; boundaries (especially `discordant_results`) are fuzzy.
- **Severe class imbalance.** ~74% `negative`; some classes have very few examples → unstable per-class metrics, and a strong temptation for the model to ignore rare classes.
- **Missing values everywhere.** Many records lack one or more labs (tests weren't ordered). **Missingness is non-random and correlated with the clinical question** — both a hazard (potential leakage) and a signal.
- **Data-entry errors.** Known issues include implausible ages (e.g. > 120) and occasional out-of-range values. Clean before training.
- **Flag circularity / leakage.** Several history flags effectively encode the answer (`on_thyroxine` ⇒ `replacement_therapy`; `on_antithyroid_medication` ⇒ treated `hyperthyroid`). High predictive power from these is *bookkeeping*, not clinical insight.
- **Not a substitute for clinical diagnosis.** This is an educational/benchmark dataset. Real thyroid diagnosis uses current free-T4/free-T3 assays, antibody panels (anti-TPO, TRAb), imaging, and clinical exam — none fully captured here.

---

## 10. Glossary

- **Euthyroid** — normal thyroid function.
- **Hypothyroid** — underactive thyroid (too little hormone).
- **Hyperthyroid / thyrotoxicosis** — overactive thyroid (too much hormone).
- **Primary disease** — the problem is in the **thyroid gland** itself (TSH moves opposite to hormone).
- **Secondary / central disease** — the problem is in the **pituitary/hypothalamus** (TSH fails to respond correctly).
- **Subclinical / compensated** — abnormal TSH but still-normal hormone levels (gland compensating).
- **HPT axis** — Hypothalamic–Pituitary–Thyroid control loop.
- **TRH** — thyrotropin-releasing hormone (from hypothalamus).
- **TSH** — thyroid-stimulating hormone (from pituitary; the gas pedal).
- **T4 (thyroxine)** — main thyroid hormone, prohormone reservoir (4 iodines).
- **T3 (triiodothyronine)** — active thyroid hormone (3 iodines).
- **TT4** — total T4 (bound + free).
- **Free T4 / FTI** — the active, unbound fraction (or its computed index).
- **T4U** — thyroxine uptake; indirect measure of binding-protein capacity, used to compute FTI.
- **TBG** — thyroxine-binding globulin, the main carrier protein.
- **Deiodinase** — enzyme that converts T4 → T3 in peripheral tissues.
- **Goitre** — enlarged thyroid gland.
- **Hashimoto's thyroiditis** — autoimmune destruction of thyroid → hypothyroidism.
- **Graves' disease** — autoimmune stimulation of thyroid → hyperthyroidism.
- **Sick euthyroid / non-thyroidal illness** — abnormal thyroid labs caused by severe non-thyroid illness, not by thyroid disease.
- **I-131** — radioactive iodine used to ablate thyroid tissue.
- **Levothyroxine** — synthetic T4 used as replacement therapy.

---

*End of medical grounding document for Experiment 2 (thyroid-disease classification).*
