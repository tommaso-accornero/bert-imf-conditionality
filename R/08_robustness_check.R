# R/08_model_fit_diagnostics.R
# McFadden pseudo-R² and likelihood ratio test for M3_extended.
# Run AFTER R/04_mission_creep_analysis.R (requires m3_extended in memory).

library(tidyverse)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

if (!exists("m3_extended")) {
  stop("m3_extended not found. Source R/04_mission_creep_analysis.R first.")
}

m3_data <- model.frame(m3_extended)
m_null  <- glm(n_policy_areas ~ 1, data = m3_data, family = poisson)

lr_test <- anova(m_null, m3_extended, test = "Chisq")
print(lr_test)

pseudo_r2 <- 1 - (as.numeric(logLik(m3_extended)) / as.numeric(logLik(m_null)))
cat("McFadden's pseudo-R²:", round(pseudo_r2, 3), "\n")

TABLES_DIR <- "results/tables"
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

fit_summary <- tibble(
  model       = "M3_extended",
  logLik_full = as.numeric(logLik(m3_extended)),
  logLik_null = as.numeric(logLik(m_null)),
  lr_stat     = round(-2 * (as.numeric(logLik(m_null)) - as.numeric(logLik(m3_extended))), 1),
  df          = attr(logLik(m3_extended), "df") - 1,
  pseudo_r2   = round(pseudo_r2, 3)
)
write_csv(fit_summary, file.path(TABLES_DIR, "model_fit_m3_extended.csv"))
cat("Saved: results/tables/model_fit_m3_extended.csv\n")