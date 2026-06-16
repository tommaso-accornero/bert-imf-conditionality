# R/06_bert_pipeline_diagram.R
# Schematic diagram of the BERT classification pipeline for Section 3.2.
# Pure ggplot2 — boxes via geom_rect, arrows via geom_segment, labels via geom_text.
# Box text below is DRAFT — wording/content to be finalised separately.

library(tidyverse)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

FIGURES_DIR <- "results/figures"
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Box definitions ───────────────────────────────────────────────────────────
# Main spine: x = 0 to 6 (top to bottom). Validation + baseline boxes: x = 7.5 to 13.5,
# shifted down 2 units relative to the previous version for better spacing.

boxes <- tribble(
  ~id,       ~xmin, ~xmax, ~ymin, ~ymax, ~label,
  "data",     0, 6, 14.0, 15.4, "IMF Monitor (1980-2019)\n35,401 conditions, 13 categories\nhand-coded (Kentikelenis & Stubbs 2023)",
  "prep",     0, 6, 12.0, 13.4, "Preprocessing & deduplication\nlowercase, whitespace; dedup on (country, year, text)\n35,401 -> 34,325 conditions",
  "split",    0, 6, 10.0, 11.4, "Train / Validation / Test split\n24,027 / 5,149 / 5,149\nstratified by 13-cat label, seed = 42",
  "train",    0, 6,  8.0,  9.4, "Fine-tune bert-base-uncased\n13-category & 7-category schemes\nMAX_LEN=128, batch=32, epochs=4, lr=3e-5\nbalanced class weights (ENV=13.3, FIN=0.32)",
  "models",   0, 6,  6.0,  7.4, "Trained classifiers\nmacro-F1 = 0.898 (13-cat)\nmacro-F1 = 0.918 (7-cat)",
  "mona",     0, 6,  4.0,  5.4, "Apply to MONA 2020-2024\n9,863 conditions classified\n(13-cat and 7-cat labels)",
  "final",    0, 6,  2.0,  3.4, "Combined dataset, 1980-2024\nfor mission creep regression (Section 3.4)",
  "val",    7.5, 13.5, 4,  7.4, "Three-check validation chain\n\n1. Seam validation (IMF Monitor 2015-2019)\n     99.0% within 1 area, bias = +0.077\n\n2. Lookup agreement (MONA)\n     76.9% (7-cat)\n\n3. Manual validation (100 MONA conditions)\n     86% / 89%, 0 errors at GOVERNANCE/FISCAL",
  "baseline", 7.5, 13.5, 8.6,  10.0, "Baseline comparison (13-cat macro-F1)\nTF-IDF + Logistic Regression: 0.774\nTF-IDF + Random Forest: 0.733\nBERT: 0.898 (+12.4 points)"
)

boxes <- boxes |>
  mutate(
    xcenter = (xmin + xmax) / 2,
    ycenter = (ymin + ymax) / 2
  )

# ── Arrow definitions ─────────────────────────────────────────────────────────
# Main spine: vertical arrows between consecutive boxes (unchanged).
# Branches: recalculated to connect within the new overlap zones —
# models<->validation and validation<->mona are now horizontal.

arrows <- tribble(
  ~x,   ~y,   ~xend, ~yend,
  3,    14.0, 3,     13.4,   # data -> prep
  3,    12.0, 3,     11.4,   # prep -> split
  3,    10.0, 3,      9.4,   # split -> train
  3,     8.0, 3,      7.4,   # train -> models
  3,     6.0, 3,      5.4,   # models -> mona
  3,     4.0, 3,      3.4,   # mona -> final
  6,     6.4, 7.5,    6.4,   # models -> validation (horizontal)
  7.5,   4.7, 6,      4.7,   # validation -> mona (horizontal)
  7.5,   9, 6,      7.3    # baseline -> models (short diagonal)
)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig_pipeline <- ggplot() +
  geom_rect(
    data = boxes,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "white", colour = "grey30", linewidth = 0.5
  ) +
  geom_text(
    data = boxes,
    aes(x = xcenter, y = ycenter, label = label),
    size = 3, lineheight = 0.95
  ) +
  geom_segment(
    data = arrows,
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(length = unit(2.5, "mm"), type = "closed"),
    colour = "grey30", linewidth = 0.5
  ) +
  labs(title = "BERT Classification Pipeline") +
  coord_fixed(ratio = 1, xlim = c(-0.5, 14), ylim = c(1.5, 16), expand = FALSE) +
  theme_void() +
  theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))

ggsave(
  file.path(FIGURES_DIR, "fig_12_bert_pipeline.png"),
  fig_pipeline, width = 10, height = 10, dpi = 300
)