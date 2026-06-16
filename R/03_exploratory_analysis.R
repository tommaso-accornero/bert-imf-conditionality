# R/03_exploratory_analysis.R
# Phase 2: Exploratory analysis and baseline visualisation — IMF Monitor 1980-2019.
# Output: results/figures/fig_01 through fig_09, fig_11
#
# Column name note: the actual CSV columns differ from the task spec.
#   spec: condition_year, condition_text, condition_policy_area, country_name
#   actual: year, text, category, country
# collapsed_7cat is not present in imf_monitor_clean.csv (it lives in mona_with_labels.csv).
# It is constructed here from `category` using COLLAPSE_MAP.
# ggridges is not installed; fig_07 uses violin instead of ridgeline.

library(tidyverse)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROCESSED_DIR <- "data/processed"
FIGURES_DIR   <- "results/figures"
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# Two reform waves — used selectively per figure (see comments on each figure).
# 2002: 2002 Conditionality Guidelines (first streamlining wave; volume & scope story)
# 2009: Lending framework overhaul, SPC abolition (condition type story)
REFORM_2002 <- 2002
REFORM_2009 <- 2009

# ── Load data ─────────────────────────────────────────────────────────────────

imf <- read_csv(
  file.path(PROCESSED_DIR, "imf_monitor_clean.csv"),
  show_col_types = FALSE
)

# ── Verify required columns ───────────────────────────────────────────────────

required_cols <- c("year", "text", "category", "arrangement_id", "condition_type", "country")
missing_cols  <- setdiff(required_cols, names(imf))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

# ── Build collapsed_7cat ──────────────────────────────────────────────────────

COLLAPSE_MAP <- c(
  FP  = "FISCAL",   RTP = "FISCAL",   DEB = "FISCAL",
  FIN = "FINANCIAL",
  EXT = "EXTERNAL",
  SOE = "PRIVATISATION", PRI = "PRIVATISATION",
  LAB = "LABOUR",
  SP  = "SOCIAL",   POV = "SOCIAL",
  INS = "GOVERNANCE", ENV = "GOVERNANCE", OTH = "GOVERNANCE"
)

imf <- imf |>
  mutate(
    cat7   = COLLAPSE_MAP[category],
    decade = paste0(floor(year / 10) * 10, "s")
  )

# ── Colour palette (7-cat, defined once) ─────────────────────────────────────
# Order: FISCAL first (most common), GOVERNANCE last (mission creep signal).

CAT7_ORDER <- c("FISCAL", "FINANCIAL", "PRIVATISATION", "EXTERNAL",
                "LABOUR", "SOCIAL", "GOVERNANCE")

CAT7_COLOURS <- c(
  FISCAL        = "#2166ac",
  FINANCIAL     = "#4dac26",
  PRIVATISATION = "#d01c8b",
  EXTERNAL      = "#f1a340",
  LABOUR        = "#998ec3",
  SOCIAL        = "#d7191c",
  GOVERNANCE    = "#636363"
)

imf <- imf |>
  mutate(cat7 = factor(cat7, levels = CAT7_ORDER))

# ── Shared theme ──────────────────────────────────────────────────────────────

theme_thesis <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(colour = "grey40", size = 11),
      axis.title    = element_text(size = 11),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}

# 2002 line: dashed. 2009 line: dotted. Different linetypes so both are readable together.
vline_2002 <- geom_vline(xintercept = REFORM_2002, linetype = "dashed", colour = "grey30", linewidth = 0.6)
vline_2009 <- geom_vline(xintercept = REFORM_2009, linetype = "dotted", colour = "grey30", linewidth = 0.6)

label_2002 <- annotate("text", x = REFORM_2002 + 0.4, y = Inf,
                       label = "2002 Conditionality reform", hjust = 0, vjust = 1.5, size = 2.65, colour = "grey30")
label_2009 <- annotate("text", x = REFORM_2009 + 0.4, y = Inf,
                       label = "2009 Lending reform", hjust = 0, vjust = 1.5, size = 2.65, colour = "grey30")

# ── fig_01: Conditions per year ───────────────────────────────────────────────
# Reform reference: 2002 CG — the volume decline starts here.

cat("Producing fig_01...\n")

fig_01_data <- imf |> count(year)

fig_01 <- ggplot(fig_01_data, aes(x = year, y = n)) +
  geom_line(linewidth = 0.8, colour = "#2166ac") +
  geom_point(size = 1.5, colour = "#2166ac") +
  vline_2002 +
  label_2002 +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  labs(
    title = "IMF Conditions per Year, 1980-2019",
    x     = NULL,
    y     = "Number of conditions"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_01_conditions_per_year.png"),
  fig_01, width = 10, height = 6, dpi = 300
)

# ── fig_02: 7-category share by decade ───────────────────────────────────────

cat("Producing fig_02...\n")

fig_02_data <- imf |>
  count(decade, cat7) |>
  group_by(decade) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  mutate(decade = factor(decade, levels = c("1980s", "1990s", "2000s", "2010s")))

fig_02 <- ggplot(fig_02_data, aes(x = decade, y = prop, fill = cat7)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = CAT7_COLOURS, breaks = CAT7_ORDER, name = NULL) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "7-Category Distribution by Decade",
    x     = NULL,
    y     = "Share of conditions"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_02_category_share_by_decade.png"),
  fig_02, width = 10, height = 6, dpi = 300
)

# ── fig_03: Condition type composition over time ──────────────────────────────

cat("Producing fig_03...\n")

CTYPE_ORDER <- c("QPC", "SB", "PA", "IB", "SPC", "PC")

CTYPE_COLOURS <- c(
  QPC = "#2166ac", SB = "#4dac26", PA = "#d01c8b",
  IB  = "#f1a340", SPC = "#998ec3", PC = "#d7191c"
)

fig_03_data <- imf |>
  filter(condition_type %in% CTYPE_ORDER) |>
  count(year, condition_type) |>
  group_by(year) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  mutate(condition_type = factor(condition_type, levels = CTYPE_ORDER))

# Reform reference: 2009 only — the SPC abolition is what's visible here.
fig_03 <- ggplot(fig_03_data, aes(x = year, y = prop, fill = condition_type)) +
  geom_area(alpha = 0.9) +
  vline_2009 +
  label_2009 +
  scale_fill_manual(values = CTYPE_COLOURS, name = NULL) +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Condition Type Composition, 1980-2019",
    subtitle = "2009 overhaul abolished Structural Performance Criteria (SPC); Structural Benchmarks (SB) became dominant.",
    x        = NULL,
    y        = "Share of conditions"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_03_condition_type_over_time.png"),
  fig_03, width = 10, height = 6, dpi = 300
)

# ── fig_04: Text length by category ──────────────────────────────────────────

cat("Producing fig_04...\n")

fig_04_data <- imf |>
  mutate(word_count = str_count(text, "\\S+")) |>
  group_by(cat7) |>
  mutate(median_wc = median(word_count, na.rm = TRUE)) |>
  ungroup() |>
  mutate(cat7 = fct_reorder(cat7, median_wc))

x_cap    <- 200
n_trimmed <- sum(fig_04_data$word_count > x_cap, na.rm = TRUE)
cat(sprintf("  fig_04: %d observations trimmed at x = %d words\n", n_trimmed, x_cap))

fig_04 <- ggplot(fig_04_data, aes(x = word_count, y = cat7)) +
  geom_boxplot(fill = "grey80", outlier.size = 0.6, outlier.alpha = 0.4) +
  coord_cartesian(xlim = c(0, x_cap)) +
  labs(
    title    = "Condition Text Length by Category (Word Count)",
    subtitle = sprintf("x-axis capped at %d words; %d observations not shown", x_cap, n_trimmed),
    x        = "Word count",
    y        = NULL
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_04_text_length_by_category.png"),
  fig_04, width = 10, height = 6, dpi = 300
)

# ── fig_05: Baseline policy scope per year ────────────────────────────────────
# Unit: arrangement. Scope = distinct categories per arrangement.
# A multi-year arrangement contributes its scope to each year it has conditions.

cat("Producing fig_05...\n")

arrangement_scope <- imf |>
  group_by(arrangement_id) |>
  summarise(
    scope_13   = n_distinct(category),
    scope_7    = n_distinct(cat7),
    years_active = list(unique(year)),
    .groups = "drop"
  ) |>
  unnest(years_active) |>
  rename(year = years_active)

fig_05_data <- arrangement_scope |>
  group_by(year) |>
  summarise(
    mean_scope_13 = mean(scope_13),
    mean_scope_7  = mean(scope_7),
    n_arrangements = n(),
    .groups = "drop"
  )

# Reform references: both — scope plateau starts at 2002, dip visible at 2009.
fig_05 <- ggplot(fig_05_data, aes(x = year)) +
  geom_line(aes(y = mean_scope_13, colour = "13-category"), linewidth = 0.8) +
  geom_line(aes(y = mean_scope_7, colour = "7-category"), linewidth = 0.8) +
  vline_2002 + label_2002 +
  vline_2009 + label_2009 +
  scale_colour_manual(
    values = c("13-category" = "#2166ac", "7-category" = "#d01c8b"),
    name   = "Scheme"
  ) +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  labs(
    title    = "Mean Policy Scope per Program per Year, 1980-2019 (Hand-Coded Data)",
    subtitle = "Scope = distinct policy categories per arrangement. Dashed lines: loess smoother.",
    x        = NULL,
    y        = "Mean number of distinct categories"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_05_baseline_scope_per_year.png"),
  fig_05, width = 10, height = 6, dpi = 300
)

# ── fig_06: Non-core share over time ─────────────────────────────────────────
# Non-core = GOVERNANCE + SOCIAL + LABOUR

cat("Producing fig_06...\n")

NON_CORE <- c("GOVERNANCE", "SOCIAL", "LABOUR")

fig_06_data <- imf |>
  mutate(is_noncore = cat7 %in% NON_CORE) |>
  group_by(year) |>
  summarise(
    noncore_share = mean(is_noncore),
    .groups = "drop"
  )

# Reform references: both — non-core growth plateaus near 2002, dips at 2009 then recovers.
fig_06 <- ggplot(fig_06_data, aes(x = year, y = noncore_share)) +
  geom_line(linewidth = 0.8, colour = "#636363") +
  geom_smooth(method = "loess", se = TRUE, colour = "#d01c8b",
              fill = "#d01c8b", alpha = 0.15, linewidth = 0.7) +
  vline_2002 + label_2002 +
  vline_2009 + label_2009 +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Non-Core Category Share Over Time, 1980-2019",
    subtitle = "Non-core = GOVERNANCE + SOCIAL + LABOUR. Shaded band: loess 95% CI.",
    x        = NULL,
    y        = "Share of conditions"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_06_noncore_share_over_time.png"),
  fig_06, width = 10, height = 6, dpi = 300
)

# ── fig_07: Program scope distribution by decade (violin) ────────────────────
# ggridges not installed; using violin plot.
# Unit: arrangement-level scope_7, one row per arrangement.

cat("Producing fig_07...\n")

arrangement_scope_decade <- imf |>
  group_by(arrangement_id) |>
  summarise(
    scope_7 = n_distinct(cat7),
    decade  = paste0(floor(min(year) / 10) * 10, "s"),
    .groups = "drop"
  ) |>
  mutate(decade = factor(decade, levels = c("1980s", "1990s", "2000s", "2010s")))

fig_07 <- ggplot(arrangement_scope_decade, aes(x = decade, y = scope_7)) +
  geom_violin(fill = "grey80", colour = "grey40", trim = TRUE) +
  geom_boxplot(width = 0.08, fill = "white", outlier.size = 0.8) +
  scale_y_continuous(breaks = 1:7) +
  labs(
    title    = "Distribution of Program Policy Scope by Decade (7-Category Scheme)",
    subtitle = "Each point = one arrangement. Scope = number of distinct 7-cat domains.",
    x        = NULL,
    y        = "Policy scope (distinct domains)"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_07_scope_distribution_by_decade.png"),
  fig_07, width = 10, height = 8, dpi = 300
)

# ── fig_08: Share of broad programs over time ─────────────────────────────────
# Broad = scope_7 >= 5 (out of max 7).
# Threshold of >= 5 on the 7-cat scheme is equivalent to >= 7 on the 13-cat
# scheme per thesis design, capturing programs that span a majority of policy domains.

cat("Producing fig_08...\n")

fig_08_data <- arrangement_scope |>
  mutate(is_broad = scope_7 >= 5) |>
  group_by(year) |>
  summarise(
    broad_share    = mean(is_broad),
    n_arrangements = n(),
    .groups = "drop"
  )

# Reform references: both — plateau near 2002, dip at 2009 then recovery.
fig_08 <- ggplot(fig_08_data, aes(x = year, y = broad_share)) +
  geom_line(linewidth = 0.8, colour = "#2166ac") +
  vline_2002 + label_2002 +
  vline_2009 + label_2009 +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title    = "Share of Broad Programs (>=5 Domains) per Year, 1980-2019",
    subtitle = "Broad = scope_7 >= 5 of 7 possible domains. Shaded band: loess 95% CI.",
    x        = NULL,
    y        = "Share of arrangements"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_08_high_scope_programs_over_time.png"),
  fig_08, width = 10, height = 6, dpi = 300
)

# ── fig_09: Non-core absolute counts over time ────────────────────────────────
# Companion to fig_06. fig_06 shows non-core SHARE — this shows absolute COUNTS.
# Distinguishes whether non-core growth is a real expansion or just a share gain
# driven by core conditions falling faster. Both can be true simultaneously.

cat("Producing fig_09...\n")

fig_09_data <- imf |>
  filter(cat7 %in% NON_CORE) |>
  count(year, cat7) |>
  mutate(cat7 = factor(cat7, levels = c("GOVERNANCE", "SOCIAL", "LABOUR")))

# Console summary: total non-core conditions by decade
cat("\n  Non-core absolute counts by decade:\n")
imf |>
  filter(cat7 %in% NON_CORE) |>
  mutate(decade = paste0(floor(year / 10) * 10, "s")) |>
  count(decade, cat7) |>
  pivot_wider(names_from = cat7, values_from = n, values_fill = 0) |>
  mutate(total_noncore = GOVERNANCE + SOCIAL + LABOUR) |>
  print()

NONCORE_COLOURS <- c(
  GOVERNANCE = "#636363",
  SOCIAL     = "#d7191c",
  LABOUR     = "#998ec3"
)

fig_09 <- ggplot(fig_09_data, aes(x = year, y = n, colour = cat7)) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.9) +
  vline_2002 + label_2002 +
  vline_2009 + label_2009 +
  scale_colour_manual(values = NONCORE_COLOURS, name = NULL) +
  scale_x_continuous(breaks = seq(1980, 2019, 5)) +
  labs(
    title    = "Non-Core Conditions: Absolute Counts per Year, 1980-2019",
    subtitle = "Companion to fig_06 (shares). Solid lines: loess smoother. Faint lines: raw annual counts.",
    x        = NULL,
    y        = "Number of conditions"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_09_noncore_absolute_counts.png"),
  fig_09, width = 10, height = 6, dpi = 300
)

# ── fig_11: Non-core share, 1980-2024 (IMF Monitor + MONA) ────────────────────
# Extends fig_06 to the full 1980-2024 window by adding MONA 2020-2024
# (BERT-classified, bert_7cat). Non-core definition is identical to fig_06/fig_09
# (GOVERNANCE + SOCIAL + LABOUR), applied at the 7-cat level for both sources.
#
# Requires data/final/mona_classified.csv (produced by python/05 in Phase 4).
# This file must exist before this section will run.
#
# Two separate geom_smooth layers (no raw geom_line) to avoid clutter and to
# avoid a misleading visual connection across the 2019/2020 methodology seam.
# IMF Monitor (40 years, dense): loess. MONA (5 years, sparse): linear trend.

mona_path <- file.path("data/final", "mona_classified.csv")
if (!file.exists(mona_path)) stop("mona_classified.csv not found. Run python/05 first.")
mona_classified <- read_csv(mona_path, show_col_types = FALSE)
print(names(mona_classified))

cat("Producing fig_11...\n")

mona_for_fig11 <- mona_classified |>
  filter(test_year >= 2020, test_year <= 2024) |>
  filter(test_year >= 2020, test_year <= 2024) |>
  transmute(
    year       = test_year,
    is_noncore = bert_7cat %in% NON_CORE,
    source     = "mona_bert"
  )

imf_for_fig11 <- imf |>
  transmute(
    year,
    is_noncore = cat7 %in% NON_CORE,
    source = "imf_monitor"
  )

yearly <- bind_rows(imf_for_fig11, mona_for_fig11) |>
  group_by(year, source) |>
  summarise(noncore_share = mean(is_noncore), .groups = "drop")

fig_11_draft <- yearly |>
  ggplot(aes(x = year, y = noncore_share, colour = source)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(
    values = c("imf_monitor" = "#2c7bb6", "mona_bert" = "#d7191c"),
    labels = c("imf_monitor" = "IMF Monitor (hand-coded)", "mona_bert" = "MONA (BERT-classified)")
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(0, 0.35)) +  vline_2002 + label_2002 + vline_2009 + label_2009 +
  labs(
    title    = "Non-core Condition Share, 1980-2024",
    subtitle = "Share of conditions outside core IMF mandate (FIN, DEB, FP, RTP, EXT)",
    x        = NULL, y = "Non-core share",
    colour   = "Source"
  ) +
  theme_thesis()

ggsave(
  file.path(FIGURES_DIR, "fig_11_noncore_share_1980_2024.png"),
  fig_11_draft, width = 10, height = 5.5, dpi = 300
)

# ── Summary check ─────────────────────────────────────────────────────────────

cat("\n=== Output file check ===\n")

expected_files <- c(
  "fig_01_conditions_per_year.png",
  "fig_02_category_share_by_decade.png",
  "fig_03_condition_type_over_time.png",
  "fig_04_text_length_by_category.png",
  "fig_05_baseline_scope_per_year.png",
  "fig_06_noncore_share_over_time.png",
  "fig_07_scope_distribution_by_decade.png",
  "fig_08_high_scope_programs_over_time.png",
  "fig_09_noncore_absolute_counts.png",
  "fig_11_noncore_share_1980_2024.png"
)

tibble(
  filename = expected_files,
  exists   = file.exists(file.path(FIGURES_DIR, expected_files))
) |> print()

cat("\nPhase 2 (exploratory analysis) complete.\n")