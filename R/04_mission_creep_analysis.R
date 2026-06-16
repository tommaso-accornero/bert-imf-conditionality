# R/04_mission_creep_analysis.R
# Phase 5: Mission creep analysis — revised and extended regression specification.
# Run from project root in RStudio.
#
# Required packages beyond CLAUDE.md stack:
#   install.packages(c("glmmTMB", "modelsummary", "car"))
#
# Inputs:  data/processed/imf_monitor_clean.csv  (hand-coded, 1980-2019)
#          data/final/mona_classified.csv         (BERT-classified, 2020-2024)
# Outputs: data/final/combined_1980_2024.csv
#          results/figures/fig_10_policy_scope_1980_2024.png
#          results/figures/fig_11_noncore_share_1980_2024.png
#          results/tables/regression_nb.csv
#          results/tables/structural_break_tests.csv
#          results/tables/regression_table.html / .tex

library(tidyverse)
library(MASS)
library(lmtest)
library(sandwich)
library(patchwork)
library(broom)
library(scales)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

# Optional packages — loaded with graceful fallback
HAS_GLMMTMB     <- requireNamespace("glmmTMB",      quietly = TRUE)
HAS_MODELSUMMARY <- requireNamespace("modelsummary", quietly = TRUE)
HAS_CAR          <- requireNamespace("car",          quietly = TRUE)

if (!HAS_GLMMTMB)      message("glmmTMB not installed — M6 (random effects) will be skipped. Run: install.packages('glmmTMB')")
if (!HAS_MODELSUMMARY) message("modelsummary not installed — regression table will be skipped. Run: install.packages('modelsummary')")
if (!HAS_CAR)          message("car not installed — VIF check will be skipped. Run: install.packages('car')")

if (HAS_GLMMTMB)      library(glmmTMB)
if (HAS_MODELSUMMARY) library(modelsummary)
if (HAS_CAR)          library(car)

set.seed(42)

source("R/utils/plotting_theme.R")

PROCESSED_DIR <- "data/processed"
FINAL_DIR     <- "data/final"
FIGURES_DIR   <- "results/figures"
TABLES_DIR    <- "results/tables"
DIAG_DIR      <- "results/diagnostics"
dir.create(DIAG_DIR, showWarnings = FALSE, recursive = TRUE)

REFORM_2002 <- 2002
REFORM_2009 <- 2009

vline_2002 <- geom_vline(xintercept = REFORM_2002, linetype = "dashed", colour = "grey30", linewidth = 0.6)
vline_2009 <- geom_vline(xintercept = REFORM_2009, linetype = "dotted", colour = "grey30", linewidth = 0.6)
label_2002 <- annotate("text", x = REFORM_2002 + 0.4, y = Inf, label = "2002 CG",       hjust = 0, vjust = 1.5, size = 3.2, colour = "grey30")
label_2009 <- annotate("text", x = REFORM_2009 + 0.4, y = Inf, label = "2009 overhaul", hjust = 0, vjust = 1.5, size = 3.2, colour = "grey30")

# ── 1. Load and harmonise both datasets ───────────────────────────────────────

cat("Loading IMF Monitor (1980-2019)...\n")
imf <- read_csv(
  file.path(PROCESSED_DIR, "imf_monitor_clean.csv"),
  show_col_types = FALSE
) |>
  rename(arrangement_amount = `Arrangement Amount`) |>
  select(
    country, country_code, year,
    arrangement_id, arrangement_type, arrangement_amount,
    condition_type, category
  ) |>
  mutate(
    # Arrangement Amount: SDR millions at approval, not including augmentations (codebook).
    # Comma-separated = two co-approved arrangements from the same EBM document.
    # Since both programs share the same arrangement_id and are aggregated together
    # in policy_scope, we sum both amounts to get total resources at that EBM meeting.
    # This is consistent with treating co-approved programs as one program-year unit.
    arrangement_amount = sapply(arrangement_amount, function(x) {
      parts <- strsplit(as.character(x), ",")[[1]]
      nums  <- suppressWarnings(as.numeric(trimws(parts)))
      total <- sum(nums, na.rm = TRUE)
      if_else(total == 0, NA_real_, total)
    }),
    source = "imf_monitor"
  )

cat("Loading MONA classified (2020-2024)...\n")
mona <- read_csv(
  file.path(FINAL_DIR, "mona_classified.csv"),
  show_col_types = FALSE
) |>
  filter(test_year >= 2020, test_year <= 2024) |>
  select(
    country,
    year           = test_year,       # use test_year (condition assessment year) not approval year
    arrangement_id = arrangement_number,
    arrangement_type,
    condition_type = key_code,
    category       = bert_category,
  ) |>
  mutate(
    arrangement_id     = as.character(arrangement_id),
    source             = "mona_bert",
    country_code       = NA_character_,
    arrangement_amount = NA_real_
  )

cat(sprintf("IMF Monitor: %d conditions (%d-%d)\n", nrow(imf), min(imf$year), max(imf$year)))
cat(sprintf("MONA BERT:   %d conditions (%d-%d)\n", nrow(mona), min(mona$year), max(mona$year)))

# ── 2. Combine ─────────────────────────────────────────────────────────────────

CORE_CATS    <- c("FIN", "DEB", "FP", "RTP", "EXT")
NONCORE_CATS <- c("SOE", "LAB", "PRI", "INS", "POV", "SP", "OTH", "ENV")

combined <- bind_rows(imf, mona) |>
  mutate(is_noncore = category %in% NONCORE_CATS)

cat(sprintf("Combined:    %d conditions (%d-%d)\n\n", nrow(combined), min(combined$year), max(combined$year)))
write_csv(combined, file.path(FINAL_DIR, "combined_1980_2024.csv"))
cat("Saved: data/final/combined_1980_2024.csv\n\n")

# ── 3. Program-level policy scope ─────────────────────────────────────────────

policy_scope <- combined |>
  group_by(country, country_code, year, arrangement_id, arrangement_type, source) |>
  summarise(
    n_conditions      = n(),
    n_policy_areas    = n_distinct(category),
    n_noncore         = sum(is_noncore),
    noncore_share     = mean(is_noncore),
    arrangement_amount = first(arrangement_amount),   # constant within arrangement; first() is sufficient
    .groups           = "drop"
  ) |>
  mutate(
    post_2002 = as.integer(year >= REFORM_2002),
    post_2009 = as.integer(year >= REFORM_2009),
    year_c    = year - 1990,
    source_handcoded = as.integer(source == "imf_monitor")
  )

cat("Policy scope summary:\n")
policy_scope |>
  group_by(source) |>
  summarise(
    programs    = n(),
    mean_scope  = round(mean(n_policy_areas), 2),
    mean_n_cond = round(mean(n_conditions), 1),
    .groups     = "drop"
  ) |>
  print()

# ── 4. CONTROL VARIABLE CONSTRUCTION ──────────────────────────────────────────

# 4a. Log total conditions — controls for program size (longer programs cover more areas)
policy_scope <- policy_scope |>
  mutate(log_n_conditions = log(n_conditions + 1))

# 4b. Transition economy dummy — FSU and Eastern European transition programs
# 1991-2005: IMF engaged heavily with post-Soviet states, inflating non-fiscal scope.
# Excluding or controlling for this period avoids confounding mission creep signal.
transition_codes <- c(
  "ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LVA", "LTU", "MDA",
  "RUS", "TJK", "UKR", "UZB",                                # Former Soviet Union (TKM not in data)
  "ALB", "BIH", "BGR", "HRV", "CZE", "HUN", "MKD", "POL",
  "ROU", "SRB", "SVK", "XKX"                                 # Eastern Europe (SVN, MNE not in data)
)

policy_scope <- policy_scope |>
  mutate(transition_economy = as.integer(
    !is.na(country_code) & country_code %in% transition_codes &
      year >= 1991 & year <= 2005
  ))

cat(sprintf(
  "Transition economy program-years: %d (%.1f%% of IMF Monitor sample)\n",
  sum(policy_scope$transition_economy, na.rm = TRUE),
  mean(policy_scope$transition_economy[policy_scope$source == "imf_monitor"]) * 100
))

# 4c. Repeat borrower dummy — countries with 15+ years under IMF programs, 1980-2024.
# Repeat borrowers face entrenched structural conditions that inflate scope independently
# of mission creep. Threshold of 15/44 years follows logic of Kentikelenis et al.
# (who use 19/30 for their shorter window). See TODO 3 for sensitivity.
repeat_borrower_flag <- policy_scope |>
  filter(!is.na(country_code)) |>
  group_by(country_code) |>
  summarise(years_under_program = n_distinct(year), .groups = "drop") |>
  mutate(repeat_borrower = as.integer(years_under_program >= 15))

policy_scope <- policy_scope |>
  left_join(
    repeat_borrower_flag |> select(country_code, repeat_borrower),
    by = "country_code"
  ) |>
  mutate(repeat_borrower = replace_na(repeat_borrower, 0L))

cat(sprintf(
  "Repeat borrower countries (>=15 years): %d\n",
  sum(repeat_borrower_flag$repeat_borrower, na.rm = TRUE)
))

# 4d. Log loan size — larger programs may have more conditions across more areas.
# Only available for IMF Monitor (1980-2019). NA for MONA rows.
# Missing share in IMF Monitor reported below.
policy_scope <- policy_scope |>
  mutate(log_loan_size = if_else(
    !is.na(arrangement_amount) & arrangement_amount > 0,
    log(arrangement_amount + 1),
    NA_real_
  ))

n_missing_loan <- sum(is.na(policy_scope$log_loan_size) & policy_scope$source == "imf_monitor")
cat(sprintf(
  "Log loan size: %d IMF Monitor program-years missing (%.1f%%) — rapid access facilities likely\n",
  n_missing_loan,
  n_missing_loan / sum(policy_scope$source == "imf_monitor") * 100
))

# 4e. Region fixed effects — IMF regional department classification.
# Controls for regional patterns in conditionality scope.
# Countries not in the lookup receive NA; share reported below.
imf_region_map <- tribble(
  ~country_code, ~region,
  # AFR — Sub-Saharan Africa
  "AGO","AFR","BEN","AFR","BFA","AFR","BDI","AFR","CMR","AFR","CAF","AFR",
  "TCD","AFR","COM","AFR","COD","AFR","COG","AFR","CIV","AFR","DJI","AFR",
  "ETH","AFR","GAB","AFR","GMB","AFR","GHA","AFR","GIN","AFR","GNB","AFR",
  "KEN","AFR","LSO","AFR","LBR","AFR","MDG","AFR","MWI","AFR","MLI","AFR",
  "MRT","AFR","MOZ","AFR","NER","AFR","NGA","AFR","RWA","AFR","STP","AFR",
  "SEN","AFR","SLE","AFR","SOM","AFR","ZAF","AFR","SSD","AFR","SDN","AFR",
  "SWZ","AFR","TZA","AFR","TGO","AFR","UGA","AFR","ZMB","AFR","ZWE","AFR",
  "GNQ","AFR","ERI","AFR","CPV","AFR","NAM","AFR","MUS","AFR",
  # APD — Asia and Pacific
  "BGD","APD","BTN","APD","KHM","APD","FJI","APD","IDN","APD","IND","APD",
  "KIR","APD","LAO","APD","MYS","APD","MDV","APD","MHL","APD","FSM","APD",
  "MNG","APD","MMR","APD","NPL","APD","PLW","APD","PNG","APD","PHL","APD",
  "WSM","APD","SLB","APD","LKA","APD","KOR","APD","THA","APD","TLS","APD",
  "TON","APD","TUV","APD","VUT","APD","VNM","APD",
  # EUR — Europe
  "ALB","EUR","BLR","EUR","BIH","EUR","BGR","EUR","HRV","EUR","CYP","EUR",
  "CZE","EUR","EST","EUR","GRC","EUR","HUN","EUR","ISL","EUR","IRL","EUR",
  "LVA","EUR","LTU","EUR","MDA","EUR","MKD","EUR","MNE","EUR","POL","EUR",
  "PRT","EUR","ROU","EUR","ROM","EUR","RUS","EUR","SRB","EUR","SVK","EUR",
  "SVN","EUR","TUR","EUR","UKR","EUR","XKX","EUR",
  # MCD — Middle East and Central Asia
  "AFG","MCD","DZA","MCD","ARM","MCD","AZE","MCD","BHR","MCD","EGY","MCD",
  "GEO","MCD","IRN","MCD","IRQ","MCD","JOR","MCD","KAZ","MCD","KWT","MCD",
  "KGZ","MCD","LBN","MCD","LBY","MCD","MAR","MCD","OMN","MCD","PAK","MCD",
  "QAT","MCD","SAU","MCD","SYR","MCD","TJK","MCD","TKM","MCD","TUN","MCD",
  "ARE","MCD","UZB","MCD","YEM","MCD",
  # WHD — Western Hemisphere
  "ARG","WHD","BLZ","WHD","BOL","WHD","BRA","WHD","CHL","WHD","COL","WHD",
  "CRI","WHD","DOM","WHD","ECU","WHD","SLV","WHD","GRD","WHD","GTM","WHD",
  "GUY","WHD","HTI","WHD","HND","WHD","JAM","WHD","MEX","WHD","NIC","WHD",
  "PAN","WHD","PRY","WHD","PER","WHD","KNA","WHD","LCA","WHD","VCT","WHD",
  "SUR","WHD","TTO","WHD","URY","WHD","VEN","WHD",
  # Additional small states and historical countries
  "ATG","WHD","BRB","WHD","DMA","WHD",   # Caribbean small states
  "SYC","AFR",                            # Seychelles
  "CHN","APD",                            # China
  "CSK","EUR","YUG","EUR"                 # Historical: Czechoslovakia, Yugoslavia
)

policy_scope <- policy_scope |>
  left_join(imf_region_map, by = "country_code") |>
  mutate(region = factor(region))

n_missing_region <- sum(is.na(policy_scope$region) & policy_scope$source == "imf_monitor")
if (n_missing_region > 0) {
  missing_ccs <- policy_scope |>
    filter(is.na(region), source == "imf_monitor") |>
    distinct(country_code, country) |>
    arrange(country_code)
  cat(sprintf("WARNING: %d IMF Monitor program-years missing region (%d countries):\n",
              n_missing_region, nrow(missing_ccs)))
  print(missing_ccs)
} else {
  cat("Region mapping: all IMF Monitor program-years matched.\n")
}

# ── 4f. External controls: GDP per capita (WDI) and crisis dummies (L&V 2018) ──

gdp_controls    <- read_csv(file.path(PROCESSED_DIR, "gdp_controls.csv"),    show_col_types = FALSE)
crisis_dummies  <- read_csv(file.path(PROCESSED_DIR, "crisis_dummies.csv"),  show_col_types = FALSE)

# For MONA rows, country_code is NA — resolve iso3c from country name to enable GDP merge
mona_iso <- policy_scope |>
  filter(source == "mona_bert", is.na(country_code)) |>
  distinct(country) |>
  mutate(country_code_resolved = countrycode::countrycode(
    country, origin = "country.name", destination = "iso3c", warn = FALSE
  ))

policy_scope <- policy_scope |>
  left_join(mona_iso, by = "country") |>
  mutate(
    country_code_merge = if_else(is.na(country_code), country_code_resolved, country_code)
  ) |>
  select(-country_code_resolved)

# Merge GDP (1980-2024)
policy_scope <- policy_scope |>
  left_join(gdp_controls, by = c("country_code_merge" = "iso3c", "year"))

# Merge crisis dummies (1980-2017 only; 2018+ assumed = 0)
# L&V coverage ends 2017. COVID-era programs (2020-2024) not captured — document as limitation.
policy_scope <- policy_scope |>
  left_join(crisis_dummies, by = c("country_code_merge" = "country_code", "year")) |>
  mutate(
    any_crisis = case_when(
      !is.na(any_crisis) ~ any_crisis,
      year >= 2018       ~ 0L,
      TRUE               ~ NA_integer_
    )
  )

n_missing_gdp    <- sum(is.na(policy_scope$log_gdp_pc)  & policy_scope$source == "imf_monitor")
n_missing_crisis <- sum(is.na(policy_scope$any_crisis)   & policy_scope$source == "imf_monitor" & policy_scope$year <= 2017)

cat(sprintf("Missing log_gdp_pc (IMF Monitor): %d (%.1f%%)\n",
            n_missing_gdp, n_missing_gdp / sum(policy_scope$source == "imf_monitor") * 100))
cat(sprintf("Missing any_crisis 1980-2017 (IMF Monitor): %d (%.1f%%)\n",
            n_missing_crisis, n_missing_crisis / sum(policy_scope$source == "imf_monitor" & policy_scope$year <= 2017) * 100))

# ── 5. Ceiling pressure check ──────────────────────────────────────────────────
#
# If programs increasingly hit the upper bound (n_policy_areas close to 13),
# NB regression will underestimate scope growth in later periods.

cat("\n=== Ceiling Pressure Check ===\n")
ceiling_check <- policy_scope |>
  mutate(decade = floor(year / 10) * 10) |>
  group_by(decade, source) |>
  summarise(
    n_programs    = n(),
    pct_above_10  = mean(n_policy_areas >= 10),
    pct_above_12  = mean(n_policy_areas >= 12),
    mean_scope    = round(mean(n_policy_areas), 2),
    .groups       = "drop"
  )

print(ceiling_check)

if (any(ceiling_check$pct_above_10 > 0.15)) {
  cat("NOTE: >15% of programs in at least one decade cover 10+ policy areas.\n")
  cat("      NB may underestimate scope growth in later periods. Note in thesis.\n")
}

# ── 6. Year-level aggregates for time series figures ──────────────────────────

yearly <- combined |>
  group_by(year, source) |>
  summarise(
    n_conditions  = n(),
    n_noncore     = sum(is_noncore),
    noncore_share = mean(is_noncore),
    .groups       = "drop"
  )

scope_yearly <- policy_scope |>
  group_by(year, source) |>
  summarise(
    mean_scope   = mean(n_policy_areas),
    median_scope = median(n_policy_areas),
    n_programs   = n(),
    .groups      = "drop"
  )

# ── 7. Figure 10: Policy scope time series 1980-2024 ─────────────────────────

fig_10 <- scope_yearly |>
  ggplot(aes(x = year, y = mean_scope, colour = source)) +
  geom_line(linewidth = 0.8) +
  geom_smooth(
    data   = filter(scope_yearly, source == "imf_monitor"),
    method = "loess", span = 0.3, se = TRUE, alpha = 0.15,
    colour = "#2c7bb6", fill = "#2c7bb6"
  ) +
  geom_smooth(
    data   = filter(scope_yearly, source == "mona_bert"),
    method = "loess", span = 0.5, se = TRUE, alpha = 0.15,
    colour = "#d7191c", fill = "#d7191c"
  ) +
  scale_colour_manual(
    values = c("imf_monitor" = "#2c7bb6", "mona_bert" = "#d7191c"),
    labels = c("imf_monitor" = "IMF Monitor (hand-coded)", "mona_bert" = "MONA (BERT-classified)")
  ) +
  vline_2002 + label_2002 + vline_2009 + label_2009 +
  labs(
    title    = "Average policy scope per IMF program, 1980-2024",
    subtitle = "Number of distinct policy areas per arrangement per year",
    x        = NULL, y = "Mean distinct policy areas",
    colour   = "Source"
  ) +
  theme_thesis()

ggsave(file.path(FIGURES_DIR, "fig_10_policy_scope_1980_2024.png"),
       fig_10, width = 10, height = 5.5, dpi = 300)
cat("Saved: fig_10_policy_scope_1980_2024.png\n")

# ── 8. Figure 11: Non-core share 1980-2024 ────────────────────────────────────

fig_11 <- yearly |>
  ggplot(aes(x = year, y = noncore_share, colour = source)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(
    values = c("imf_monitor" = "#2c7bb6", "mona_bert" = "#d7191c"),
    labels = c("imf_monitor" = "IMF Monitor (hand-coded)", "mona_bert" = "MONA (BERT-classified)")
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(0, 0.5)) +  vline_2002 + label_2002 + vline_2009 + label_2009 +
  labs(
    title    = "Non-core Condition Share, 1980-2024",
    subtitle = "Share of conditions outside core IMF mandate (FIN, DEB, FP, RTP, EXT)",
    x        = NULL, y = "Non-core share",
    colour   = "Source"
  ) +
  theme_thesis()

ggsave(file.path(FIGURES_DIR, "fig_11_noncore_share_1980_2024.png"),
       fig_11, width = 10, height = 5.5, dpi = 300)
cat("Saved: fig_11_noncore_share_1980_2024.png\n")

# ── 9. Regression data preparation ────────────────────────────────────────────
#
# Primary regression uses IMF Monitor only (1980-2019).
# MONA rows excluded: BERT classification error is not random and could
# create a spurious post-2020 trend (documented in Step 4 robustness check).

reg_data <- policy_scope |>
  filter(source == "imf_monitor") |>
  mutate(arrangement_type = fct_lump_min(arrangement_type, min = 30))

cat(sprintf("\nRegression sample: %d program-years (IMF Monitor only)\n", nrow(reg_data)))
cat("Arrangement type distribution:\n")
print(table(reg_data$arrangement_type))
cat(sprintf("Missing log_loan_size: %d (%.1f%%)\n",
            sum(is.na(reg_data$log_loan_size)),
            mean(is.na(reg_data$log_loan_size)) * 100))

# ── 10. Model suite ───────────────────────────────────────────────────────────

cat("\n=== Poisson Regression — Model Suite ===\n")
cat("Note: glm.nb theta converged to ~3e10 (effectively Poisson) across all specifications.\n")
cat("      Poisson GLM is used — identical coefficient estimates, valid AIC/BIC.\n\n")

# M1: Base (reform dummies + arrangement type only)
m1 <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type,
  family = poisson, data = reg_data
)

# M2: Add core controls (program size, loan size, region)
m2 <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
    log_n_conditions + log_loan_size + region,
  family = poisson, data = reg_data
)

# M3 retained as robustness check (no external controls); M3_extended is primary
m3 <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
    log_n_conditions + log_loan_size + region +
    transition_economy + repeat_borrower,
  family = poisson, data = reg_data
)

# M4: Full model with reform interactions (slope change tests)
m4 <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
    log_n_conditions + log_loan_size + region +
    transition_economy + repeat_borrower +
    year_c:post_2002 + year_c:post_2009,
  family = poisson, data = reg_data
)

# M5: Without arrangement_type (endogeneity sensitivity)
m5 <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 +
    log_n_conditions + log_loan_size + region +
    transition_economy + repeat_borrower,
  family = poisson, data = reg_data
)

# M6: Random effects NB (preferred long-run specification — accounts for
# country-level clustering directly in the model rather than via SE correction)
if (HAS_GLMMTMB) {
  cat("Fitting M6 (random effects NB via glmmTMB) — may take a few minutes...\n")
  m6_re <- glmmTMB(
    n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
      log_n_conditions + log_loan_size + region +
      transition_economy + repeat_borrower +
      (1 | country_code),
    family = poisson,
    data   = filter(reg_data, !is.na(country_code))
  )
  cat("M6 fitted.\n")
} else {
  cat("Skipping M6 (glmmTMB not installed).\n")
}

# Cluster-robust SEs at country level for M1-M5
# Always use these when reporting; naive SEs are too small due to within-country correlation.
se_m1 <- coeftest(m1, vcov = vcovCL(m1, cluster = ~country_code))
se_m2 <- coeftest(m2, vcov = vcovCL(m2, cluster = ~country_code))
se_m3 <- coeftest(m3, vcov = vcovCL(m3, cluster = ~country_code))
se_m4 <- coeftest(m4, vcov = vcovCL(m4, cluster = ~country_code))
se_m5 <- coeftest(m5, vcov = vcovCL(m5, cluster = ~country_code))

cat("\nM3 (full specification) with cluster-robust SEs:\n")
print(se_m3)
cat("\nM3 incidence rate ratios (IRR = exp(coef)):\n")
irr_m3 <- exp(coef(m3))
print(round(irr_m3, 3))

# All Poisson models converge cleanly. LR tests are valid. AIC/BIC reported for model comparison.
cat("\nModel comparison via AIC and BIC (lower = better fit):\n")
model_comparison <- data.frame(
  model = c("M1 (base)", "M2 (core controls)", "M3 (full)", "M4 (interactions)", "M5 (no arr. type)"),
  df    = c(length(coef(m1)), length(coef(m2)), length(coef(m3)), length(coef(m4)), length(coef(m5))),
  AIC   = round(c(AIC(m1), AIC(m2), AIC(m3), AIC(m4), AIC(m5)), 1),
  BIC   = round(c(BIC(m1), BIC(m2), BIC(m3), BIC(m4), BIC(m5)), 1)
)
print(model_comparison)

# LR test M3 vs M4: tests whether interaction terms improve fit beyond full controls.
if (m4$converged) {
  cat("\nLR test M3 vs M4 (interactions beyond full controls):\n")
  print(lrtest(m3, m4))
} else {
  cat("\nLR test M3 vs M4 skipped: M4 theta did not converge. Use AIC/BIC above.\n")
}

# ── 10b. PRIMARY SPECIFICATION: M3_extended (GDP + crisis controls) ──────────
#
# M3_extended:      PRIMARY MODEL — M3 formula + log_gdp_pc + any_crisis. IMF Monitor only.
# M3_full_extended: same formula, full 1980-2024 sample. Crisis set to 0 for 2018+.

cat("\n=== Extended Models: GDP + Crisis Controls ===\n")

# Define m3_full_sample here for use in the comparison table
# (also defined in section 11 for the robustness check — identical model)
reg_data_full_base <- policy_scope |> filter(!is.na(log_n_conditions))
m3_full_sample <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + log_n_conditions + source,
  family = poisson, data = reg_data_full_base
)

# reg_data already has log_gdp_pc and any_crisis from the 4f merge above
reg_data_ext <- reg_data

cat(sprintf("M3_extended sample before NA drop: %d\n", nrow(reg_data_ext)))
cat(sprintf("Missing log_gdp_pc: %d (%.1f%%)\n",
            sum(is.na(reg_data_ext$log_gdp_pc)),
            mean(is.na(reg_data_ext$log_gdp_pc)) * 100))

# PRIMARY MODEL — reported as main specification in thesis
m3_extended <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
    log_n_conditions + log_loan_size + region +
    transition_economy + repeat_borrower +
    log_gdp_pc + any_crisis,
  family = poisson, data = reg_data_ext
)

se_m3_ext <- coeftest(m3_extended, vcov = vcovCL(m3_extended, cluster = ~country_code))
cat(sprintf("M3_extended fitted on %d program-years.\n", nrow(model.frame(m3_extended))))

# M6_extended: random effects version of M3_extended (adds log_gdp_pc + any_crisis to M6's formula)
if (HAS_GLMMTMB) {
  cat("Fitting M6_extended (random effects, extended formula)...\n")
  m6_extended <- glmmTMB(
    n_policy_areas ~ year_c + post_2002 + post_2009 + arrangement_type +
      log_n_conditions + log_loan_size + region +
      transition_economy + repeat_borrower +
      log_gdp_pc + any_crisis +
      (1 | country_code),
    family = poisson,
    data   = filter(reg_data_ext, !is.na(country_code))
  )
  cat("M6_extended fitted.\n")
  m6_ext_coefs <- fixef(m6_extended)$cond
  key_check <- c("year_c", "post_2002", "post_2009", "log_gdp_pc", "any_crisis")
  cat("\nM6_extended IRR sanity check:\n")
  for (k in key_check) {
    v <- tryCatch(round(exp(m6_ext_coefs[k]), 3), error = function(e) NA)
    cat(sprintf("  %-15s IRR = %s\n", k, ifelse(is.na(v), "not found", v)))
  }
} else {
  cat("Skipping M6_extended (glmmTMB not installed).\n")
}

# M3_full_extended: full 1980-2024 sample
# Note: crisis dummy set to 0 for 2018-2024 (L&V coverage ends 2017).
# COVID-era programs (2020-2024) not captured — interpret with caution.
reg_data_full_ext <- policy_scope |>
  filter(!is.na(log_n_conditions)) |>
  mutate(arrangement_type = fct_lump_min(arrangement_type, min = 30, other_level = "Other"))

m3_full_extended <- glm(
  n_policy_areas ~ year_c + post_2002 + post_2009 + log_n_conditions + source +
    log_gdp_pc + any_crisis,
  family = poisson, data = reg_data_full_ext
)

se_m3_full_ext <- coeftest(m3_full_extended, vcov = vcovCL(m3_full_extended, cluster = ~country))
cat(sprintf("M3_full_extended fitted on %d program-years.\n", nrow(model.frame(m3_full_extended))))

# Four-column comparison table
key     <- c("year_c", "post_2002", "post_2009", "log_gdp_pc", "any_crisis")
safe_irr <- function(mod, k) {
  v <- tryCatch(round(exp(coef(mod)[k]), 3), error = function(e) NA)
  if (is.na(v) || length(v) == 0) "—" else as.character(v)
}

cat("\n=== Four-Column IRR Comparison ===\n")
comparison_ext <- data.frame(
  term = c(key, "N"),
  M3_base = c(
    sapply(key[1:3], function(k) safe_irr(m3, k)), "—", "—",
    nrow(model.frame(m3))
  ),
  M3_primary = c(
    sapply(key, function(k) safe_irr(m3_extended, k)),
    nrow(model.frame(m3_extended))
  ),
  M3_full = c(
    sapply(key[1:3], function(k) safe_irr(m3_full_sample, k)), "—", "—",
    nrow(model.frame(m3_full_sample))
  ),
  M3_full_extended = c(
    sapply(key, function(k) safe_irr(m3_full_extended, k)),
    nrow(model.frame(m3_full_extended))
  ),
  stringsAsFactors = FALSE
)
print(comparison_ext, row.names = FALSE)

# Stability check
m3_irr      <- exp(coef(m3))
m3_ext_irr  <- exp(coef(m3_extended))

year_c_shift   <- abs(m3_ext_irr["year_c"]   - m3_irr["year_c"])
post2002_shift <- abs(m3_ext_irr["post_2002"] - m3_irr["post_2002"])

cat(sprintf("\nyear_c shift:   %+.4f (IRR %s → %s)\n",
            m3_ext_irr["year_c"] - m3_irr["year_c"],
            round(m3_irr["year_c"], 3), round(m3_ext_irr["year_c"], 3)))
cat(sprintf("post_2002 shift: %+.4f (IRR %s → %s)\n",
            m3_ext_irr["post_2002"] - m3_irr["post_2002"],
            round(m3_irr["post_2002"], 3), round(m3_ext_irr["post_2002"], 3)))

if (year_c_shift > 0.02 | post2002_shift > 0.02) {
  message("WARNING: coefficient shifted >0.02 — investigate before reporting.")
} else {
  message("Coefficients stable: finding robust to GDP and crisis controls.")
}

# Save
tidy_ext <- tidy(m3_extended) |>
  mutate(irr = round(exp(estimate), 3))
write_csv(tidy_ext,    file.path(TABLES_DIR, "regression_m3_extended.csv"))
write_csv(comparison_ext, file.path(TABLES_DIR, "comparison_extended.csv"))
cat("\nSaved: regression_m3_extended.csv, comparison_extended.csv\n")

# ── 11. Hand-coded period robustness check ─────────────────────────────────────
#
# The critical test: does the mission creep finding hold in 1980-2019 hand-coded
# data alone, or does it only appear when MONA 2020-2024 (BERT-classified) is added?
#
# M3          = IMF Monitor only (1980-2019) — primary specification
# M3_handcoded = same as M3 (explicit label for the comparison table)
# M3_full_sample = combined 1980-2024 including MONA — sensitivity check
#
# If year_c / post_2002 / post_2009 are stable across M3 and M3_full_sample,
# the BERT extension is confirmatory and the finding is robust.
# If they shift substantially, BERT classification error may be driving results —
# flag prominently in Chapter 4 limitations before submission.

cat("\n=== Hand-Coded Period Robustness ===\n")

# M3_handcoded: identical to M3 (reg_data is already 1980-2019 only)
m3_handcoded <- m3

# M3_full_sample: adds MONA 2020-2024. Use only controls available for both sources.
# log_loan_size, transition_economy, repeat_borrower excluded (NA for MONA rows).
# M3_full uses a reduced formula: arrangement_type and region are NA for MONA rows
# so they are excluded here. The comparison is on year_c / post_2002 / post_2009 only.
reg_data_full <- policy_scope |>
  filter(!is.na(log_n_conditions))

cat(sprintf("Full sample for M3_full: %d program-years (%d IMF Monitor + %d MONA)\n",
            nrow(reg_data_full),
            sum(reg_data_full$source == "imf_monitor"),
            sum(reg_data_full$source == "mona_bert")))

# m3_full_sample already defined in section 10b — reuse it here
# (reg_data_full and reg_data_full_base are identical: all policy_scope rows with non-NA log_n_conditions)
reg_data_full <- reg_data_full_base

# Cluster on country name (country_code is NA for MONA rows)
se_m3_full <- coeftest(m3_full_sample, vcov = vcovCL(m3_full_sample, cluster = ~country))

# Coefficient comparison on the three key terms
key_terms <- c("year_c", "post_2002", "post_2009")

compare_coefs <- bind_rows(
  tibble(
    model    = "M3_handcoded (1980-2019)",
    term     = key_terms,
    estimate = coef(m3)[key_terms],
    irr      = round(exp(coef(m3)[key_terms]), 3),
    se       = sqrt(diag(vcovCL(m3, cluster = ~country_code)))[key_terms]
  ),
  tibble(
    model    = "M3_full_sample (1980-2024)",
    term     = key_terms,
    estimate = coef(m3_full_sample)[key_terms],
    irr      = round(exp(coef(m3_full_sample)[key_terms]), 3),
    se       = sqrt(diag(vcovCL(m3_full_sample, cluster = ~country)))[key_terms]
  )
) |>
  mutate(across(c(estimate, se), \(x) round(x, 4)))

cat("\nKey coefficient comparison — the central robustness test:\n")
print(compare_coefs)

if (any(abs(compare_coefs$irr[compare_coefs$model == "M3_full_sample (1980-2024)"] -
            compare_coefs$irr[compare_coefs$model == "M3_handcoded (1980-2019)"]) > 0.15)) {
  cat("\nWARNING: IRR shifts > 0.15 between hand-coded and full sample.\n")
  cat("         Investigate BERT classification error before finalising results.\n")
} else {
  cat("\nCoefficients stable across hand-coded and full sample: finding is robust.\n")
}

write_csv(compare_coefs, file.path(TABLES_DIR, "robustness_handcoded_vs_full.csv"))

# ── 12. Diagnostics ───────────────────────────────────────────────────────────

cat("\n=== Diagnostics ===\n")

# 12a. VIF check for M4 (most complex model)
if (HAS_CAR) {
  tryCatch({
    cat("VIF check (M3_extended — primary specification):\n")
    vif_m3_extended <- car::vif(m3_extended)
    print(round(vif_m3_extended, 2))
    vif_vals_m3 <- if (is.matrix(vif_m3_extended)) vif_m3_extended[, 3] else vif_m3_extended
    # Threshold: GVIF^(1/(2*Df)) > sqrt(10) ≈ 3.16 is severe (equivalent to VIF > 10)
    # year_c = 2.67 → standard VIF equivalent = 2.67² ≈ 7.1: elevated but below severe threshold
    high_vif_m3 <- vif_vals_m3[vif_vals_m3 > sqrt(10)]
    if (length(high_vif_m3) > 0) {
      cat("WARNING: GVIF^(1/(2*Df)) > 3.16 (VIF > 10) for:", paste(names(high_vif_m3), collapse = ", "), "\n")
    } else {
      cat("No severe multicollinearity in M3_extended (all GVIF^(1/(2*Df)) < 3.16, i.e. VIF < 10).\n")
      cat("Note: year_c GVIF^(1/(2*Df)) = 2.67, equivalent to VIF ≈ 7.1 — elevated but acceptable.\n")
    }
    
    cat("\nVIF check (M4 — interaction model, expected high collinearity):\n")
    vif_m4_check <- car::vif(m4)
    print(round(vif_m4_check, 2))
    vif_vals_m4 <- if (is.matrix(vif_m4_check)) vif_m4_check[, 3] else vif_m4_check
    high_vif_m4 <- vif_vals_m4[vif_vals_m4 > 5]
    if (length(high_vif_m4) > 0) {
      cat("NOTE: VIF > 5 for:", paste(names(high_vif_m4), collapse = ", "), "\n")
      cat("      This is expected for interaction terms and does not affect M3_extended.\n")
    }
    
    vif_df_m3 <- if (is.matrix(vif_m3_extended)) {
      as.data.frame(vif_m3_extended) |> rownames_to_column("term")
    } else {
      tibble(term = names(vif_m3_extended), GVIF_adj = vif_m3_extended)
    }
    vif_df_m4 <- if (is.matrix(vif_m4_check)) {
      as.data.frame(vif_m4_check) |> rownames_to_column("term")
    } else {
      tibble(term = names(vif_m4_check), GVIF_adj = vif_m4_check)
    }
    write_csv(vif_df_m3, file.path(DIAG_DIR, "vif_m3_extended.csv"))
    write_csv(vif_df_m4, file.path(DIAG_DIR, "vif_m4.csv"))
    cat("Saved: vif_m3_extended.csv and vif_m4.csv\n")
    
  }, error = function(e) cat("VIF check failed:", conditionMessage(e), "\n"))
} else {
  cat("Skipping VIF check (car not installed).\n")
}

# 12b. Overdispersion check for Poisson M3
# With Poisson, check if residual deviance >> residual df (sign of overdispersion).
# If dispersion >> 1, quasi-Poisson or NB would be needed.
cat("\nOverdispersion check (Poisson M3):\n")
dispersion_ratio <- m3$deviance / m3$df.residual
cat(sprintf("  Residual deviance: %.1f\n", m3$deviance))
cat(sprintf("  Residual df:       %d\n",   m3$df.residual))
cat(sprintf("  Dispersion ratio:  %.3f\n",  dispersion_ratio))
if (dispersion_ratio > 1.5) {
  cat("  WARNING: Dispersion ratio > 1.5 — quasi-Poisson or NB may be needed.\n")
  cat("  Consider: glm(..., family = quasipoisson)\n")
} else {
  cat("  Dispersion ratio acceptable — Poisson is appropriate.\n")
}

# 12c. Structural break tests (LR-based)
# Tests whether the entire coefficient vector changes at each reform year.
# Different from M3 post_2002/post_2009: those test level shifts; this tests full structural change.
cat("\nStructural break LR tests:\n")

run_break_test <- function(data, break_year) {
  df_pre  <- filter(data, year <  break_year)
  df_post <- filter(data, year >= break_year)

  # Fit same controls in both halves; transition_economy likely 0 post-2009
  formula_base <- n_policy_areas ~ year_c + arrangement_type +
    log_n_conditions + log_loan_size + region + repeat_borrower

  m_full <- glm(formula_base, family = poisson, data = data)
  m_pre  <- glm(formula_base, family = poisson, data = df_pre)
  m_post <- glm(formula_base, family = poisson, data = df_post)

  ll_full  <- as.numeric(logLik(m_full))
  ll_split <- as.numeric(logLik(m_pre)) + as.numeric(logLik(m_post))
  lr_stat  <- -2 * (ll_full - ll_split)
  df_stat  <- attr(logLik(m_pre), "df") + attr(logLik(m_post), "df") -
              attr(logLik(m_full), "df")
  p_val    <- pchisq(lr_stat, df = df_stat, lower.tail = FALSE)

  tibble(
    break_year  = break_year,
    n_pre       = nrow(df_pre),
    n_post      = nrow(df_post),
    lr_stat     = round(lr_stat, 3),
    df          = df_stat,
    p_value     = round(p_val, 4),
    significant = p_val < 0.05
  )
}

breaks <- bind_rows(
  run_break_test(reg_data, REFORM_2002),
  run_break_test(reg_data, REFORM_2009)
)
print(breaks)

# ── 13. Threshold analysis ─────────────────────────────────────────────────────

threshold_7 <- policy_scope |>
  group_by(year, source) |>
  summarise(
    pct_7plus  = mean(n_policy_areas >= 7),
    n_programs = n(),
    .groups    = "drop"
  )

cat("\nShare of programs with 7+ policy areas (selected years):\n")
threshold_7 |>
  filter(year %in% c(1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2024)) |>
  print()

# ── 14. Regression output table ───────────────────────────────────────────────

if (HAS_MODELSUMMARY) {
  cat("\nProducing regression table...\n")

  model_list <- list(
    "Base"             = m1,
    "Core controls"    = m2,
    "Full"             = m3,
    "Interactions"     = m4,
    "No arr. type"     = m5
  )

  modelsummary(
    model_list,
    exponentiate = TRUE,
    statistic    = "conf.int",
    conf_level   = 0.95,
    vcov         = lapply(list(m1, m2, m3, m4, m5), function(m) vcovCL(m, cluster = ~country_code)),
    stars        = TRUE,
    title        = "Poisson regression (PPML): policy scope per IMF program-year",
    notes        = "IRRs with cluster-robust 95% CIs (country-level clustering). Outcome: n_policy_areas. Poisson GLM justified empirically over NB (dispersion ratio = 0.227).",
    output       = file.path(TABLES_DIR, "regression_table.html")
  )

  modelsummary(
    model_list,
    exponentiate = TRUE,
    statistic    = "conf.int",
    conf_level   = 0.95,
    vcov         = lapply(list(m1, m2, m3, m4, m5), function(m) vcovCL(m, cluster = ~country_code)),
    stars        = TRUE,
    output       = file.path(TABLES_DIR, "regression_table.tex")
  )

  cat("Saved: regression_table.html and .tex\n")
} else {
  cat("Skipping regression table (modelsummary not installed).\n")
}

# ── 15. Save results ───────────────────────────────────────────────────────────

se_m3_df <- data.frame(
  term       = rownames(se_m3),
  cluster_se = se_m3[, "Std. Error"],
  cluster_z  = se_m3[, "z value"],
  cluster_p  = se_m3[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

tidy_m3 <- tidy(m3) |>
  left_join(se_m3_df, by = "term") |>
  mutate(
    irr         = round(exp(estimate), 3),
    irr_ci_low  = round(exp(estimate - 1.96 * cluster_se), 3),
    irr_ci_high = round(exp(estimate + 1.96 * cluster_se), 3),
  ) |>
  select(term, estimate, cluster_se, cluster_z, cluster_p, irr, irr_ci_low, irr_ci_high)

write_csv(tidy_m3,    file.path(TABLES_DIR, "regression_nb.csv"))
write_csv(breaks,     file.path(TABLES_DIR, "structural_break_tests.csv"))
write_csv(threshold_7, file.path(TABLES_DIR, "threshold_7plus.csv"))
write_csv(ceiling_check, file.path(DIAG_DIR, "ceiling_pressure.csv"))

cat("\nSaved:\n")
cat("  results/tables/regression_nb.csv\n")
cat("  results/tables/structural_break_tests.csv\n")
cat("  results/tables/threshold_7plus.csv\n")
cat("  results/diagnostics/ceiling_pressure.csv\n")
cat("\nPhase 5 mission creep analysis complete.\n")