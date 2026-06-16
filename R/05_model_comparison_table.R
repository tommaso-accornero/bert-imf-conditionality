# R/05_model_comparison_table.R
# Produces a four-model comparison table: M3, M3_extended, M6, M6_extended.
# Run from project root AFTER R/04_mission_creep_analysis.R has been sourced,
# so that m3, m3_extended, m6_re, and m6_extended are in memory.
# Output: results/tables/model_comparison_m3_m6.html
# This script is additive — it does not modify any existing model objects or output files.

library(tidyverse)
library(modelsummary)
library(sandwich)
library(lmtest)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

TABLES_DIR <- "results/tables"

# ── Verify required objects are in memory ────────────────────────────────────

required <- c("m3", "m3_extended", "m6_re", "m6_extended")
missing  <- required[!sapply(required, exists)]
if (length(missing) > 0) {
  stop(sprintf(
    "Missing model objects: %s\nSource R/04_mission_creep_analysis.R first.",
    paste(missing, collapse = ", ")
  ))
}

# ── Rows to keep (drop arrangement_type and region for readability) ───────────

coef_map <- c(
  "(Intercept)"      = "(Intercept)",
  "year_c"           = "year_c",
  "post_2002"        = "post_2002",
  "post_2009"        = "post_2009",
  "log_gdp_pc"       = "log_gdp_pc",
  "any_crisis"       = "any_crisis",
  "log_n_conditions" = "log_n_conditions",
  "log_loan_size"    = "log_loan_size",
  "transition_economy" = "transition_economy",
  "repeat_borrower"  = "repeat_borrower"
)

# ── Build model list ──────────────────────────────────────────────────────────

model_list <- list(
  "M3 (cluster-robust)"            = m3,
  "M3_extended (cluster-robust)"   = m3_extended,
  "M6 (random effects)"            = m6_re,
  "M6_extended (random effects)"   = m6_extended
)

# Cluster-robust vcov for M3 and M3_extended; default for M6 objects
vcov_list <- list(
  vcovCL(m3,          cluster = ~country_code),
  vcovCL(m3_extended, cluster = ~country_code),
  NULL,   # glmmTMB — modelsummary uses its own default
  NULL    # glmmTMB — modelsummary uses its own default
)

# ── Produce HTML table ────────────────────────────────────────────────────────

cat("Producing model_comparison_m3_m6.html...\n")

modelsummary(
  model_list,
  exponentiate = TRUE,
  statistic    = "conf.int",
  conf_level   = 0.95,
  vcov         = vcov_list,
  coef_map     = coef_map,
  stars        = TRUE,
  title        = "Primary specification robustness: cluster-robust SEs vs. random effects, with and without demand-side controls",
  notes        = "IRRs with 95% CIs. M3/M3_extended: cluster-robust SEs (country level). M6/M6_extended: country-level random intercepts. Outcome: n_policy_areas.",
  output       = file.path(TABLES_DIR, "model_comparison_m3_m6.html")
)

cat("Saved: results/tables/model_comparison_m3_m6.html\n")

# ── Console summary: year_c, post_2002, post_2009 across all four models ─────

cat("\n=== IRR Stability Check Across Four Models ===\n")
cat(sprintf("%-12s  %-30s  %-30s  %-28s  %-28s\n",
            "Term",
            "M3 (cluster-robust)",
            "M3_extended (cluster-robust)",
            "M6 (random effects)",
            "M6_extended (random effects)"))
cat(strrep("-", 125), "\n")

irr <- function(mod, k) {
  v <- tryCatch(round(exp(coef(mod)[k]), 3), error = function(e) {
    tryCatch(round(exp(fixef(mod)$cond[k]), 3), error = function(e2) NA)
  })
  if (is.na(v) || length(v) == 0) "—" else as.character(v)
}

for (k in c("year_c", "post_2002", "post_2009")) {
  cat(sprintf("%-12s  %-30s  %-30s  %-28s  %-28s\n",
              k,
              irr(m3, k),
              irr(m3_extended, k),
              irr(m6_re, k),
              irr(m6_extended, k)))
}
