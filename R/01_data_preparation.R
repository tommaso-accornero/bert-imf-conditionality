# R/01_data_preparation.R
# Phase 1: Load, clean, and validate both datasets.
# Output: data/processed/imf_monitor_clean.csv, data/processed/mona_clean.csv

library(tidyverse)
library(readxl)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR       <- "data/raw"
PROCESSED_DIR <- "data/processed"
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Text cleaning ──────────────────────────────────────────────────────────────

clean_condition_text <- function(text) {
  text |>
    str_trim() |>
    str_squish() |>
    str_replace_all("[\\r\\n]+", " ") |>
    str_to_lower()
}

# ── 1. IMF Monitor ─────────────────────────────────────────────────────────────

cat("Loading IMF Monitor...\n")
imf_raw <- read_excel(
  file.path(RAW_DIR, "IMFMonitor_Conditionality_Raw.xlsx"),
  sheet = "Dataset"
)

imf <- imf_raw |>
  rename(
    country        = `Country Name`,
    country_code   = `Country Code`,
    arrangement_id = `Arrangement ID`,
    arrangement_type = `Arrangement Type`,
    condition_type = `Condition Type`,
    text           = `Condition Text`,
    year           = `Condition Year`,
    month          = `Condition Month`,
    category       = `Condition Policy Area`,
    impl_status    = `Condition Implementation Status`,
    waiver         = `Condition Waiver`
  ) |>
  filter(year >= 1980, year <= 2019) |>
  mutate(text = clean_condition_text(text))

# Validation
cat("\n=== IMF Monitor ===\n")
cat(sprintf("Rows (1980-2019): %d\n", nrow(imf)))
cat(sprintf("Years covered: %d - %d\n", min(imf$year), max(imf$year)))
cat(sprintf("Countries: %d\n", n_distinct(imf$country)))

cat("\nMissing values in key columns:\n")
imf |>
  select(country, year, text, category, condition_type) |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  print()

cat("\nCategory distribution:\n")
imf |>
  count(category, sort = TRUE) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print(n = Inf)

cat("\nCondition type distribution:\n")
imf |> count(condition_type, sort = TRUE) |> print()

cat("\nDuplicates (country + year + text):", sum(duplicated(imf[c("country", "year", "text")])), "\n")

# Drop true duplicates: rows that are 100% identical across all columns.
# Cross-program duplicates (same text, different arrangement_id) are kept here
# and deduplicated in python/01 before the BERT train/val/test split.
n_before <- nrow(imf)
imf <- distinct(imf)
cat(sprintf("Dropped %d true duplicates (fully identical rows). Rows now: %d\n",
            n_before - nrow(imf), nrow(imf)))

# ── 2. MONA ────────────────────────────────────────────────────────────────────

cat("\nLoading MONA...\n")
mona_raw <- read_excel(
  file.path(RAW_DIR, "Combined.xlsx"),
  sheet = "MonaData"
)

mona <- mona_raw |>
  rename(
    arrangement_number = `Arrangement Number`,
    country            = `Country Name`,
    country_code       = `Country Code`,
    arrangement_type   = `Arrangement Type`,
    year               = `Approval Year`,
    key_code           = `Key Code`,
    descriptor         = `Economic Descriptor`,
    text               = `Description`,
    unique_id          = `Unique ID`
  ) |>
  mutate(text = clean_condition_text(text))

# Validation
cat("\n=== MONA ===\n")
cat(sprintf("Total rows: %d\n", nrow(mona)))
cat(sprintf("Years covered: %d - %d\n", min(mona$year, na.rm = TRUE), max(mona$year, na.rm = TRUE)))
cat(sprintf("Countries: %d\n", n_distinct(mona$country)))

cat("\nMissing values in key columns:\n")
mona |>
  select(country, year, text, descriptor, key_code) |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  print()

cat("\nKey code distribution:\n")
mona |> count(key_code, sort = TRUE) |> print()

cat("\nRows per year (2020-2024 target window):\n")
mona |>
  filter(year >= 2020, year <= 2024) |>
  count(year) |>
  print()

cat("\nTotal rows in 2020-2024 window:", nrow(filter(mona, year >= 2020, year <= 2024)), "\n")

cat("\nUnique MONA descriptors:", n_distinct(mona$descriptor), "\n")

# ── 3. Save ────────────────────────────────────────────────────────────────────

# Strip newlines from all character columns before writing to avoid broken CSV rows.
strip_newlines <- function(df) {
  mutate(df, across(where(is.character), ~ str_replace_all(., "[\\r\\n]+", " ")))
}

write_csv(strip_newlines(imf),  file.path(PROCESSED_DIR, "imf_monitor_clean.csv"))
write_csv(strip_newlines(mona), file.path(PROCESSED_DIR, "mona_clean.csv"))

cat("\nSaved:\n")
cat(sprintf("  %s/imf_monitor_clean.csv  (%d rows)\n", PROCESSED_DIR, nrow(imf)))
cat(sprintf("  %s/mona_clean.csv         (%d rows)\n", PROCESSED_DIR, nrow(mona)))
cat("\nPhase 1 complete.\n")
