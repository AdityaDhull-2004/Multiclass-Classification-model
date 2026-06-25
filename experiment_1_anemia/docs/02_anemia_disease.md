# Experiment 1 — The Disease Behind the Data: Multi-Class Anemia from CBC

> **Audience:** A software / machine-learning engineer with no medical background.
> **Goal:** Give you enough rigorous, accurate clinical grounding that when the model produces SHAP explanations ("low HGB + low MCV pushed this patient toward *iron deficiency anemia*"), you can tell whether the model learned **real physiology** or a **spurious shortcut**.
>
> This document lays out the medical reasoning behind multi-class anemia classification from CBC data. The dataset is **~1,281 patients**, each described by **14 CBC features**, labeled into **9 diagnostic classes**.
>
> No Python here — this is the *why*. The code lives elsewhere.

---

## Table of Contents

1. [What blood is made of, and what a CBC measures](#1-what-blood-is-made-of-and-what-a-cbc-measures)
2. [What anemia is](#2-what-anemia-is)
3. [The key diagnostic framework: morphological classification by MCV](#3-the-key-diagnostic-framework-morphological-classification-by-mcv)
4. [The 14 CBC features, one by one](#4-the-14-cbc-features-one-by-one)
5. [The 9 diagnostic classes: etiology, pathophysiology, and CBC signature](#5-the-9-diagnostic-classes)
6. [Mapping each feature to clinical reasoning](#6-mapping-each-feature-to-clinical-reasoning)
7. [Platelets and white cells: which classes they discriminate](#7-platelets-and-white-cells-which-classes-they-discriminate)
8. [From medicine to ML: what SHOULD be predictive, and how to sanity-check SHAP](#8-from-medicine-to-ml-what-should-be-predictive)
9. [Quick-reference cheat sheet](#9-quick-reference-cheat-sheet)
10. [Caveats and limitations](#10-caveats-and-limitations)

---

## 1. What blood is made of, and what a CBC measures

Blood is a tissue. By volume it is roughly **55% plasma** (water carrying proteins, electrolytes, hormones, clotting factors) and **45% cells**. The **Complete Blood Count (CBC)** is the single most-ordered blood test in medicine. It counts and characterizes the three cellular lineages, all of which are produced in the **bone marrow** from a common hematopoietic stem cell.

### The three cell lineages

| Lineage | Cell | Job | Roughly how many |
|---|---|---|---|
| **Erythroid** | Red blood cells (RBC, erythrocytes) | Carry oxygen via **hemoglobin** | ~4.5–5.5 million per microliter (µL) |
| **Myeloid / Lymphoid** | White blood cells (WBC, leukocytes) | Immune defense | ~4,000–11,000 per µL |
| **Megakaryocytic** | Platelets (PLT, thrombocytes) | Clotting (stop bleeding) | ~150,000–450,000 per µL |

Key mental model: **all three lines share one factory (the marrow).** That is why a single disease can perturb several lines at once. Leukemia, for example, is a cancer of white cells that crowds the marrow and *secondarily* drops red cells and platelets — which is exactly why one of our 9 classes is *Leukemia with thrombocytopenia*.

### What the CBC actually reports

A modern automated analyzer (e.g. a Coulter/impedance or flow-cytometry instrument) physically draws each cell through a sensor and measures its size and optical/electrical properties. From the raw counts it computes **derived indices** (MCV, MCH, MCHC, etc.). Understanding *which numbers are measured directly vs. derived* matters for ML, because derived features are **deterministic functions** of others — they encode redundancy the model can exploit (and which can mislead naïve feature-importance reading).

Directly measured: WBC, RBC, HGB, PLT, and the differential percentages.
Derived: HCT, MCV, MCH, MCHC, the absolute white-cell counts (LYMn, NEUTn), PCT.

We will flag each one in Section 4.

---

## 2. What anemia is

### Definition

**Anemia is a reduction in the oxygen-carrying capacity of blood**, defined operationally by a low **hemoglobin (HGB)** concentration (and/or low hematocrit / red cell count). The World Health Organization thresholds are approximately:

- Adult men: HGB **< 13.0 g/dL**
- Non-pregnant adult women: HGB **< 12.0 g/dL**
- Pregnant women: HGB **< 11.0 g/dL**

Hemoglobin is the iron-containing protein inside red cells that binds oxygen. When HGB falls, tissues get less oxygen, producing the classic symptoms: fatigue, pallor, shortness of breath on exertion, palpitations, headache.

> **HGB is the gatekeeper.** Anemia is *defined* by HGB. So in any well-behaved model, **HGB should be a dominant feature** for separating "Healthy" from every anemic class. If SHAP shows HGB barely matters, be suspicious.

### Why subtyping matters (and why this is a *multi-class* problem, not binary)

Saying "the patient is anemic" is like saying "the server is slow" — true but useless for action. **Anemia is a *sign*, not a diagnosis.** Dozens of distinct diseases lower hemoglobin, and the treatments are mutually exclusive and sometimes dangerous if mismatched:

- **Iron deficiency** → give iron. (Giving iron to someone with iron *overload* is harmful.)
- **B12/folate deficiency (macrocytic)** → give B12 or folate. (Giving folate alone to a B12-deficient patient can mask and worsen neurological damage.)
- **Anemia of chronic disease / renal** → treat the underlying disease; iron won't help.
- **Leukemia** → urgent oncology referral; this is a malignancy, not a nutrient problem.

The whole point of Experiment 1 is to **automate the subtyping** — to go from raw CBC numbers straight to the *actionable* category. The morphological framework in Section 3 is how clinicians do this by hand, and it is precisely why the red-cell indices dominate the model's feature importances.

---

## 3. The key diagnostic framework: morphological classification by MCV

**This is the single most important concept in the entire experiment.** If you read only one section, read this one. It directly explains why **MCV, MCH, MCHC, and HGB are the top SHAP features** in this experiment.

### The core idea

When red cells are abnormal, they are abnormal in **size** and/or **color (hemoglobin content)**. Two axes capture almost all of it:

1. **Size axis — driven by MCV (Mean Corpuscular Volume).** How big is the average red cell?
   - **Microcytic** = too small (MCV **< 80 fL**)
   - **Normocytic** = normal size (MCV **80–100 fL**)
   - **Macrocytic** = too big (MCV **> 100 fL**)

2. **Color axis — driven by MCH / MCHC.** How much hemoglobin is packed into each cell (i.e., how red it is under the microscope)?
   - **Hypochromic** = pale, under-filled with hemoglobin (low MCH / low MCHC)
   - **Normochromic** = normally colored (normal MCH / MCHC)

(fL = femtoliter, 10⁻¹⁵ L; pg = picogram.)

### Why size encodes the cause

Red cells are made in the marrow by repeated cell divisions while filling with hemoglobin. The **final cell size is a tug-of-war between division and hemoglobin synthesis**:

- If the cell **can't make enough hemoglobin** (not enough iron, or a globin-chain defect), it keeps dividing, getting *smaller and paler* → **microcytic, hypochromic**. This is the signature of **iron deficiency** and **thalassemia**.
- If the cell **can't divide properly** (DNA-synthesis problem from B12/folate deficiency), it grows large before dividing → **macrocytic**. This is **megaloblastic anemia**.
- If the problem is **outside the production machinery** — bleeding, destruction, suppressed marrow signaling, kidney disease — cells are made normally but there just *aren't enough* → **normocytic, normochromic**.

So MCV is not just a number; it is a **pointer to a mechanism.** That mechanistic link is why an ML model that has learned real signal will lean heavily on MCV/MCH/MCHC.

### The diagnostic table (memorize the shape of this)

| Morphology | MCV | MCH / MCHC | Typical color | Classic causes | Class(es) in our dataset |
|---|---|---|---|---|---|
| **Microcytic hypochromic** | Low (<80) | Low | Pale, small | Iron deficiency, thalassemia, chronic disease (some) | *Iron deficiency anemia*, *Other microcytic anemia* |
| **Normocytic normochromic** | Normal (80–100) | Normal | Normal | Acute blood loss, anemia of chronic disease, chronic kidney disease, early marrow failure | *Normocytic normochromic anemia* |
| **Normocytic hypochromic** | Normal | Low | Pale, normal-size | Mixed / transitional states, evolving iron deficiency, chronic disease | *Normocytic hypochromic anemia* |
| **Macrocytic** | High (>100) | High MCH (MCHC usually normal) | Large | B12 deficiency, folate deficiency, alcohol, liver disease, hypothyroidism, some drugs | *Macrocytic anemia* |

### The two-axis decision sketch

```
                         Is HGB low?  ── No ──►  consider Healthy (check WBC, PLT too)
                              │
                             Yes  → ANEMIA. Now look at MCV:
                              │
          ┌───────────────────┼─────────────────────┐
          ▼                    ▼                      ▼
     MCV < 80 fL         MCV 80–100 fL           MCV > 100 fL
   (MICROCYTIC)          (NORMOCYTIC)            (MACROCYTIC)
          │                    │                      │
   check MCH/MCHC        check MCH/MCHC          → Macrocytic anemia
          │                    │                   (B12 / folate)
   low → hypochromic    normal → normochromic
   → Iron deficiency    → chronic disease /
     or Other            blood loss / renal
     microcytic         low MCH → normocytic
                         hypochromic
```

> **For ML:** MCV cleanly *splits the anemia classes into three buckets*, and MCH/MCHC subdivide the microcytic and normocytic buckets. This is a near-textbook case of features that are individually informative **and** interact. Expect tree-based models (and their SHAP values) to reflect exactly this structure.

---

## 4. The 14 CBC features, one by one

For each feature: what it measures, an approximate adult normal range (ranges vary by lab, sex, age, and instrument — treat as orientation, not law), whether it is measured or derived, and how it behaves across anemia types.

### Red-cell core

**RBC — Red Blood Cell count**
- *Measures:* number of red cells per volume (millions/µL). Directly measured.
- *Normal:* ~4.7–6.1 (men), ~4.2–5.4 (women) million/µL.
- *Behavior:* Low in most anemias. **Important exception:** in **thalassemia trait / Other microcytic anemia**, RBC count is often **normal or even high** despite low hemoglobin — the marrow makes *many small* cells. This RBC-vs-MCV mismatch is a key discriminator between iron deficiency (RBC low) and thalassemia (RBC preserved).

**HGB — Hemoglobin**
- *Measures:* concentration of oxygen-carrying protein (g/dL). Directly measured.
- *Normal:* ~13.5–17.5 (men), ~12.0–15.5 (women) g/dL.
- *Behavior:* **The definition of anemia.** Low in every anemic class; normal in Healthy. The master severity variable.

**HCT — Hematocrit**
- *Measures:* fraction of blood volume occupied by red cells (%). Largely **derived** (≈ RBC × MCV).
- *Normal:* ~41–53% (men), ~36–46% (women).
- *Behavior:* Tracks HGB closely (rule of thumb HCT ≈ 3 × HGB). Low in anemia. Because it is derived from RBC and MCV, it is **highly collinear** with them — note this when interpreting feature importance.

### The red-cell indices (the diagnostic engine)

**MCV — Mean Corpuscular Volume**
- *Measures:* average **size** of a red cell (fL). Directly measured by the analyzer (or derived as HCT/RBC).
- *Normal:* ~80–100 fL.
- *Behavior:* **The master classifier of anemia morphology** (Section 3). Low → microcytic; high → macrocytic; normal → normocytic.

**MCH — Mean Corpuscular Hemoglobin**
- *Measures:* average **mass of hemoglobin per red cell** (pg). Derived (= HGB/RBC).
- *Normal:* ~27–33 pg.
- *Behavior:* Tracks MCV closely. Low in iron deficiency and thalassemia (hypochromia); high in macrocytic anemia (big cells hold more hemoglobin).

**MCHC — Mean Corpuscular Hemoglobin Concentration**
- *Measures:* hemoglobin **concentration within** a red cell (g/dL) — hemoglobin *density*, independent of cell size. Derived (= HGB/HCT).
- *Normal:* ~32–36 g/dL.
- *Behavior:* The purest "**hypochromic vs normochromic**" signal. Low → hypochromic (iron deficiency). Distinguishes *Normocytic normochromic* (MCHC normal) from *Normocytic hypochromic* (MCHC low) — two classes that share MCV but differ on this axis. Notably, MCHC stays roughly normal even in macrocytic anemia (the big cell holds proportionally more hemoglobin, keeping concentration steady).

### White-cell features

**WBC — White Blood Cell count**
- *Measures:* total leukocytes per volume (thousands/µL). Directly measured.
- *Normal:* ~4.0–11.0 ×10³/µL.
- *Behavior:* Usually normal in pure anemias. **Markedly abnormal in leukemia** — often very high (cancerous cells flooding the blood) but sometimes low. The key feature for flagging the leukemia classes.

**LYMp — Lymphocyte percentage**
- *Measures:* % of WBC that are lymphocytes. Derived (part of the differential).
- *Normal:* ~20–40%.
- *Behavior:* In many leukemias (especially lymphoid), the differential is grossly skewed — lymphocyte fraction can be very high. Helps characterize *which* white-cell line is abnormal.

**NEUTp — Neutrophil percentage**
- *Measures:* % of WBC that are neutrophils. Derived.
- *Normal:* ~40–70%.
- *Behavior:* LYMp + NEUTp roughly partition the differential. Skewed in leukemia. Neutrophils are the front-line bacterial defenders; a low absolute count (see NEUTn) means infection risk.

**LYMn — Lymphocyte absolute count**
- *Measures:* lymphocytes per volume (×10³/µL). Derived (= WBC × LYMp).
- *Normal:* ~1.0–4.0 ×10³/µL.
- *Behavior:* Absolute counts are clinically more meaningful than percentages. Elevated in lymphoid leukemias. Because LYMn = WBC × LYMp, it is collinear with both.

**NEUTn — Neutrophil absolute count**
- *Measures:* neutrophils per volume (×10³/µL). Derived (= WBC × NEUTp).
- *Normal:* ~2.0–7.0 ×10³/µL.
- *Behavior:* Low (**neutropenia**) when marrow is crowded out (leukemia) or suppressed; high in infection. Together with LYMn, characterizes the white-cell disturbance in the leukemia classes.

### Platelet features

**PLT — Platelet count**
- *Measures:* platelets per volume (thousands/µL). Directly measured.
- *Normal:* ~150–450 ×10³/µL.
- *Behavior:* **The defining feature of thrombocytopenia** (PLT low). Drops in marrow infiltration (leukemia), destruction, or sequestration. The single most important discriminator for *Thrombocytopenia* and *Leukemia with thrombocytopenia*.

**PDW — Platelet Distribution Width**
- *Measures:* variability in platelet size (how uneven the platelet population is). Directly measured.
- *Normal:* ~9–17 fL (lab-dependent).
- *Behavior:* Often elevated when the marrow is pumping out young, variably-sized platelets (active destruction/regeneration). Helps characterize *why* platelets are low — a supporting feature for the thrombocytopenia classes.

**PCT — Plateletcrit**
- *Measures:* fraction of blood volume occupied by platelets (%). Derived (≈ PLT × mean platelet volume). Analogous to HCT but for platelets.
- *Normal:* ~0.17–0.35%.
- *Behavior:* Tracks PLT closely; low in thrombocytopenia. Collinear with PLT.

---

## 5. The 9 diagnostic classes

For each class: **why it happens** (etiology/pathophysiology) and its **characteristic CBC signature**. The class names below are the exact labels used in the dataset's confusion matrix.

### 5.0 Healthy

- **Why:** No hematologic disease. All three cell lines normal.
- **CBC signature:** HGB, RBC, HCT within reference range; MCV 80–100 (normocytic); MCH/MCHC normal; WBC, differential, and PLT all normal.
- **Discriminators:** Normal HGB is the headline — Healthy is the only class where the oxygen-carrying gatekeeper is intact. Also normal PLT and WBC (separating it from the thrombocytopenia and leukemia classes).

### 5.1 Iron deficiency anemia (IDA)

- **Why:** Not enough iron to build hemoglobin. Causes: **chronic blood loss** (menstrual, gastrointestinal bleeding such as ulcers or colon cancer — the most common cause in adults), **inadequate dietary intake**, **malabsorption** (celiac disease, gastric surgery), or **increased demand** (pregnancy, growth).
- **Pathophysiology:** Iron is the core of heme. Without it, developing red cells under-produce hemoglobin and keep dividing → **small, pale cells**.
- **CBC signature:** **Microcytic, hypochromic.** Low HGB, **low MCV**, **low MCH**, **low MCHC**. RBC count usually **low**. Classically **high RDW** (red cells vary a lot in size as iron runs out gradually) — note: RDW is *not* in this dataset's 14 features, so the model must lean on MCV/MCH/MCHC/RBC instead.
- **Key discriminators vs. Other microcytic:** RBC is *low* here (vs. preserved/high in thalassemia), and MCHC is clearly low (hypochromic).

### 5.2 Other microcytic anemia

- **Why:** Microcytic anemias that are **not** classic iron deficiency. The main members: **thalassemia trait/disease** (inherited defect in producing one of hemoglobin's globin chains) and the microcytic end of **anemia of chronic disease** (inflammation traps iron away from red-cell production).
- **Pathophysiology:** In thalassemia, the globin imbalance yields **many small cells**; the marrow compensates by overproducing, so the *count* is preserved even though each cell is small and hemoglobin is low.
- **CBC signature:** **Microcytic** (low MCV, low MCH) but with a **different pattern from IDA**: RBC count **normal or high**, and the degree of microcytosis is often *disproportionate* to the mild anemia. RDW is often *normal* in thalassemia (cells uniformly small) — again, RDW isn't in our 14 features, so the **RBC-vs-MCV mismatch** is the model's best available proxy.
- **Key discriminators:** Same low MCV as IDA, but **RBC preserved** is the tell. This is the hardest pair for both clinicians and models to separate, so expect IDA ↔ Other microcytic to be the most confusable pair in the confusion matrix.

### 5.3 Macrocytic anemia

- **Why:** Red cells too large. Most important cause is **megaloblastic anemia** from **vitamin B12 or folate deficiency**, which impairs DNA synthesis so cells grow without dividing. Other causes: alcohol excess, liver disease, hypothyroidism, certain drugs (e.g. chemotherapy, some anticonvulsants), and reticulocytosis.
- **Pathophysiology:** B12 and folate are cofactors for DNA replication. Deficiency stalls nuclear division while the cytoplasm keeps maturing → **large (macrocytic) cells**. Severe cases also drop white cells and platelets (marrow-wide effect).
- **CBC signature:** **Macrocytic.** Low HGB, **high MCV (>100 fL)**, **high MCH** (big cells carry more hemoglobin), **MCHC usually normal** (concentration preserved). RBC low. In severe megaloblastic cases, mild drops in WBC and PLT may appear.
- **Key discriminators:** MCV is the standout — it is the *only* anemia class with high MCV, so MCV alone nearly defines this class.

### 5.4 Normocytic normochromic anemia

- **Why:** Anemia where red cells are **made normally but are too few**. Causes: **acute blood loss** (before the marrow adapts), **anemia of chronic disease/inflammation**, **chronic kidney disease** (kidneys make erythropoietin, the hormone that signals red-cell production — failing kidneys → fewer red cells), early **bone marrow failure**, and hemolysis.
- **Pathophysiology:** The production machinery and iron supply are fine, so cells have normal size and color — there are simply not enough of them (underproduction or loss).
- **CBC signature:** **Normocytic, normochromic.** Low HGB, **MCV normal (80–100)**, **MCH and MCHC normal**, RBC low. WBC and PLT usually normal (in pure forms).
- **Key discriminators:** The combination **low HGB + completely normal indices** is the signature. The model should rely on HGB being low while MCV/MCH/MCHC sit in-range — essentially "anemia with nothing else flagged."

### 5.5 Normocytic hypochromic anemia

- **Why:** Red cells are **normal in size but pale** (under-hemoglobinized). This is a transitional / mixed picture — e.g. **early or evolving iron deficiency** before cells shrink, **anemia of chronic disease**, or mixed deficiencies.
- **Pathophysiology:** Hemoglobin content per cell drops (low MCH/MCHC) before the cell size falls — the hypochromia precedes the microcytosis.
- **CBC signature:** **Normocytic but hypochromic.** Low HGB, **MCV normal**, but **MCH and/or MCHC low**. RBC low/normal.
- **Key discriminators:** Distinguished from *Normocytic normochromic* purely on the **color axis (MCHC/MCH low)** while sharing the same MCV. This is the cleanest illustration of why MCHC is its own informative feature: it separates two classes that MCV cannot.

### 5.6 Thrombocytopenia

- **Why:** Low platelet count. Three mechanism families:
  - **Decreased production** — marrow suppressed or replaced (drugs, infection, marrow disease).
  - **Increased destruction** — immune (ITP), drug-induced, or consumption (DIC, TTP).
  - **Sequestration** — an enlarged spleen pools platelets out of circulation.
- **Pathophysiology:** Platelets stop bleeding; too few → bruising, petechiae, and bleeding risk. Note this is fundamentally a **platelet disorder, not an anemia** — HGB may be normal — which is why it sits as its own class alongside the anemias.
- **CBC signature:** **Low PLT** (and low PCT, which tracks PLT). PDW may be elevated if destruction drives young-platelet release. Red-cell indices and WBC typically normal (in isolated thrombocytopenia).
- **Key discriminators:** **PLT is the whole story.** A model should put nearly all the weight for this class on PLT/PCT, with red-cell features near-irrelevant.

### 5.7 Leukemia

- **Why:** A **cancer of white blood cells**. Malignant immature white cells (**blasts**) proliferate uncontrollably in the bone marrow and spill into the blood. Acute leukemias (AML, ALL) and chronic leukemias (CML, CLL) differ in tempo and cell type.
- **Pathophysiology:** Leukemic cells **crowd out the normal marrow**, impairing production of normal red cells, normal white cells, and platelets — a *marrow-wide* takeover. The blood fills with abnormal cells.
- **CBC signature:** **Abnormal WBC** — often very **high** (sometimes strikingly so), occasionally low; the differential is **grossly skewed** (abnormal LYMp/NEUTp and absolute counts). Because normal production is crowded out, there is often a **secondary anemia** (low HGB) and frequently low platelets. A blood film would show blasts (the analyzer flags abnormal populations).
- **Key discriminators:** **WBC and the differential (LYMp, NEUTp, LYMn, NEUTn)** carry this class. Abnormal white-cell counts are the headline that no pure anemia class shows.

### 5.8 Leukemia with thrombocytopenia

- **Why:** Leukemia (as above) where the marrow takeover has **also suppressed platelet production** — a common and clinically important combination, because low platelets in a leukemia patient signal bleeding risk and advanced marrow involvement.
- **Pathophysiology:** Same leukemic infiltration, but here the megakaryocytic (platelet) line is sufficiently displaced to drop PLT.
- **CBC signature:** **Leukemia's white-cell abnormality (abnormal WBC, skewed differential) PLUS low PLT/PCT.** Often anemia (low HGB) as well.
- **Key discriminators:** It is the **intersection**: leukemic white-cell features *and* thrombocytopenia features both fire. Separating it from plain *Leukemia* hinges on PLT; separating it from plain *Thrombocytopenia* hinges on WBC/differential. A correct model should show SHAP contributions from **both** the WBC group and the PLT group for this class.

---

## 6. Mapping each feature to clinical reasoning

This section is the bridge to SHAP interpretation. For each feature, "what a high/low value *means* clinically" so you can read a SHAP attribution and check it against physiology.

| Feature | Low value points toward… | High value points toward… | Primary classes it informs |
|---|---|---|---|
| **HGB** | Anemia (any subtype) — severity marker | Healthy (or, if extreme, polycythemia — out of scope) | All anemias vs. Healthy |
| **RBC** | IDA, most anemias | **Thalassemia / Other microcytic** (preserved count); polycythemia | IDA vs. Other microcytic split |
| **HCT** | Anemia (tracks HGB) | Healthy | Same as HGB (collinear) |
| **MCV** | **Microcytic** (IDA, Other microcytic) | **Macrocytic** (B12/folate) | The 3-way morphology split |
| **MCH** | Hypochromic/microcytic (IDA, thalassemia) | Macrocytic | Microcytic + macrocytic classes |
| **MCHC** | **Hypochromic** (IDA, Normocytic hypochromic) | (rarely high; spherocytosis) | Hypochromic vs normochromic axis |
| **WBC** | Some leukemias, marrow failure | **Leukemia** (high counts) | Leukemia classes |
| **LYMp** | — | Lymphoid leukemia | Leukemia classes |
| **NEUTp** | (neutropenia component) | Infection / myeloid shift | Leukemia classes |
| **LYMn** | — | Lymphoid leukemia | Leukemia classes |
| **NEUTn** | Neutropenia (marrow crowding) | Infection | Leukemia classes |
| **PLT** | **Thrombocytopenia**, leukemia-with-thrombocytopenia | (thrombocytosis — out of scope) | Thrombocytopenia classes |
| **PDW** | — | Active platelet destruction/regeneration | Supports thrombocytopenia classes |
| **PCT** | Thrombocytopenia (tracks PLT) | — | Thrombocytopenia classes (collinear with PLT) |

### Worked examples (the kind of reasoning you'll apply to SHAP)

- **"SHAP says low HGB + low MCV + low MCHC pushed this patient toward *Iron deficiency anemia*."** Physiologically correct: low HGB = anemia; low MCV = microcytic; low MCHC = hypochromic. That triad *is* the textbook IDA signature. ✅
- **"SHAP says low MCV but *normal/high RBC* pushed toward *Other microcytic anemia* (away from IDA)."** Correct: the RBC-vs-MCV mismatch is the classic thalassemia discriminator. ✅
- **"SHAP says high MCV pushed toward *Macrocytic anemia*."** Correct: high MCV is essentially unique to this class. ✅
- **"SHAP says low PLT (and low PCT) pushed toward *Thrombocytopenia*, with red-cell features near zero."** Correct and reassuring — the model isn't leaning on irrelevant red-cell noise. ✅
- **"SHAP says high WBC + skewed LYMp/NEUTp + low PLT pushed toward *Leukemia with thrombocytopenia*."** Correct: white-cell abnormality (leukemia) plus low platelets (thrombocytopenia) is exactly the intersection. ✅
- **Red flag:** "SHAP says *PDW* is the top driver separating *Macrocytic anemia* from *Healthy*." Suspicious — PDW is a platelet feature with no mechanistic link to macrocytosis. Likely a dataset artifact or spurious correlation worth investigating.

---

## 7. Platelets and white cells: which classes they discriminate

The red-cell indices (Section 3) handle the **anemia morphology** classes. But four of our nine classes are defined by the **other two cell lines**. Here is how the non-red features earn their place.

### Platelet features → the thrombocytopenia axis

- **PLT** is the decisive feature for **Thrombocytopenia** and **Leukemia with thrombocytopenia**. Its threshold (~150 ×10³/µL) cleanly flags "platelets are low."
- **PCT** (plateletcrit) is essentially PLT expressed as a volume fraction — strongly collinear, reinforcing the same signal.
- **PDW** adds *mechanism flavor*: a high PDW alongside low PLT suggests active destruction/regeneration rather than pure underproduction. It is a **supporting** feature, not a primary classifier.

> In ML terms: PLT and PCT are near-duplicates (one platelet "factor"); PDW is a weaker, partially independent signal. Don't be surprised if the model concentrates importance on PLT and treats PCT as redundant.

### White-cell features → the leukemia axis

- **WBC** is the gateway for both **Leukemia** classes — abnormal (usually high) total count is the flag that "something is wrong with the white line."
- **LYMp / NEUTp** (percentages) and **LYMn / NEUTn** (absolute counts) describe *how* the differential is skewed, helping confirm a leukemic pattern and (in principle) hint at lineage. The percentages and absolutes are linked by `absolute = WBC × percentage`, so they carry overlapping information.

### The two-axis structure of the non-red classes

|  | Normal WBC | Abnormal WBC (leukemic) |
|---|---|---|
| **Normal PLT** | (anemia classes / Healthy) | **Leukemia** |
| **Low PLT** | **Thrombocytopenia** | **Leukemia with thrombocytopenia** |

This 2×2 makes the relationship explicit: **PLT and WBC are roughly orthogonal axes**, and three of the four corners are named classes. A model that has learned this will show SHAP contributions from the **PLT group** and the **WBC group** lighting up the appropriate corner.

---

## 8. From medicine to ML: what SHOULD be predictive

Use this section as your **SHAP sanity-check rubric**. For each class, the features that *should* dominate (per physiology) and what would count as a red flag.

| Class | Features that SHOULD dominate | What would be a red flag |
|---|---|---|
| **Healthy** | HGB (normal), PLT (normal), WBC (normal) | Heavy reliance on PDW or an absolute-count feature |
| **Iron deficiency anemia** | HGB↓, MCV↓, MCH↓, MCHC↓, RBC↓ | Platelet/WBC features driving it |
| **Other microcytic anemia** | MCV↓ with RBC normal/high; MCH↓ | MCHC strongly low (that leans IDA, not this) |
| **Macrocytic anemia** | **MCV↑** (dominant), MCH↑, HGB↓ | MCV not appearing in the top features |
| **Normocytic normochromic** | HGB↓ with MCV/MCH/MCHC all normal | MCV or MCHC being extreme drivers |
| **Normocytic hypochromic** | HGB↓, MCV normal, **MCHC↓/MCH↓** | MCV being a strong driver (would make it microcytic) |
| **Thrombocytopenia** | **PLT↓**, PCT↓, (PDW) | Red-cell indices dominating |
| **Leukemia** | **WBC** abnormal, LYMp/NEUTp/LYMn/NEUTn skewed | Red-cell-only explanation |
| **Leukemia with thrombocytopenia** | WBC abnormal **AND** PLT↓ | Only one of the two axes firing |

### General ML notes grounded in the biology

- **Collinearity is built into the CBC.** HCT≈f(RBC,MCV); MCH=HGB/RBC; MCHC=HGB/HCT; LYMn=WBC×LYMp; NEUTn=WBC×NEUTp; PCT≈f(PLT). These deterministic relationships mean importance can be *split or shifted* among collinear features. SHAP on tree ensembles distributes credit among correlated features somewhat arbitrarily — so judge a **feature *group*** (all red-cell indices together) rather than a single feature when sanity-checking.
- **MCV is the spine of the problem.** Because it three-way-splits the anemia classes, expect it among the very top global importances — consistent with our finding of MCV/MCH/MCHC/HGB as top SHAP features.
- **Expect specific confusions.** IDA ↔ Other microcytic share low MCV (separable mainly by RBC); Normocytic normochromic ↔ Normocytic hypochromic share MCV (separable by MCHC); Leukemia ↔ Leukemia-with-thrombocytopenia differ only on PLT. These are *physiologically* hard boundaries, so errors concentrated there are reassuring, not alarming — the model is confused exactly where medicine is subtle.
- **Threshold-shaped decisions.** Anemia (HGB cutoff), microcytosis (MCV 80), macrocytosis (MCV 100), thrombocytopenia (PLT 150) are all near-step-functions clinically. Tree models capture this naturally; if a linear model underperforms, the thresholding is likely why.
- **The model rediscovers the flowchart.** A well-trained classifier should essentially relearn the Section 3 decision sketch plus the Section 7 platelet/WBC axes. If SHEP/SHAP shows that, the model learned medicine. If it shows weight on physiologically irrelevant features, investigate for leakage or dataset artifacts.

---

## 9. Quick-reference cheat sheet

**The three-way MCV split (memorize):**
- MCV **< 80** → microcytic → *Iron deficiency* / *Other microcytic*
- MCV **80–100** → normocytic → *Normocytic normochromic* / *Normocytic hypochromic*
- MCV **> 100** → macrocytic → *Macrocytic anemia*

**The color axis (MCHC):**
- MCHC low → hypochromic (IDA, normocytic hypochromic)
- MCHC normal → normochromic

**The non-red axes:**
- PLT low → *Thrombocytopenia*
- WBC abnormal → *Leukemia*
- Both → *Leukemia with thrombocytopenia*

**The gatekeeper:**
- HGB normal → *Healthy*; HGB low → an anemia (then use MCV/MCHC to subtype)

**One-line signatures:**

| Class | One-line CBC signature |
|---|---|
| Healthy | HGB, MCV, PLT, WBC all normal |
| Iron deficiency anemia | HGB↓, MCV↓, MCH↓, MCHC↓, RBC↓ |
| Other microcytic anemia | MCV↓, MCH↓, **RBC normal/high** |
| Macrocytic anemia | HGB↓, **MCV↑**, MCH↑ |
| Normocytic normochromic | HGB↓, MCV/MCH/MCHC **normal** |
| Normocytic hypochromic | HGB↓, MCV normal, **MCHC↓** |
| Thrombocytopenia | **PLT↓**, PCT↓; red cells normal |
| Leukemia | **WBC abnormal**, differential skewed |
| Leukemia with thrombocytopenia | **WBC abnormal + PLT↓** |

---

## 10. Caveats and limitations

- **Reference ranges are approximate.** They vary by laboratory, instrument, sex, age, altitude, and pregnancy. The numbers here orient your intuition; they are not diagnostic thresholds for patient care.
- **CBC alone is not a final diagnosis.** Real workups add ferritin/iron studies, B12/folate levels, reticulocyte count, RDW, peripheral blood smear review, and bone marrow biopsy. This dataset deliberately restricts to 14 CBC features — which is part of what makes the ML task both interesting and inherently limited.
- **RDW is absent.** Red cell Distribution Width is one of the *most* useful real-world discriminators (especially IDA vs. thalassemia, where it is high vs. normal). It is **not** among the 14 features here, so the model substitutes the RBC-vs-MCV relationship — a weaker proxy. Keep this in mind when IDA and Other microcytic confuse.
- **Class definitions are simplifications.** "Other microcytic anemia," "Normocytic hypochromic anemia," etc. are umbrella categories assigned by the dataset's labeling process; real hematology is more granular and context-dependent.
- **This document is educational, not clinical guidance.** It exists to ground the ML experiment, not to diagnose anyone.

---

*End of Experiment 1 disease grounding. Next: the data pipeline and the model that learns to walk this flowchart automatically.*
