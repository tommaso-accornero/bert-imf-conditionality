# R/03_b_build_controls.R
# Build GDP per capita (WDI) and crisis dummy (Laeven & Valencia 2018) controls.
# Run from project root before executing R/04.
# Outputs: data/processed/gdp_controls.csv, data/processed/crisis_dummies.csv

library(tidyverse)
library(WDI)
library(readxl)
library(countrycode)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

PROCESSED_DIR <- "data/processed"
RAW_DIR       <- "data/raw"

# ── PART 1a: GDP per capita (WDI) ─────────────────────────────────────────────

cat("=== Part 1a: GDP per capita (WDI) ===\n")
cat("Downloading NY.GDP.PCAP.KD (constant 2015 USD)...\n")

# NOTE FOR REPRODUCIBILITY:
# WDI() downloads live data from the World Bank API. Data may be revised
# over time. The output gdp_controls.csv should be committed to the repo
# so that downstream scripts (R/04) use a fixed snapshot.
# This script only needs to be re-run if intentionally refreshing the data.
# Data accessed: [add date when you ran this]

gdp_raw <- WDI(
  indicator = "NY.GDP.PCAP.KD",
  country   = "all",
  start     = 1980,
  end       = 2024,
  extra     = TRUE
)

# Keep sovereign countries only (drop aggregates, regions, income groups)
gdp_clean <- gdp_raw |>
  filter(region != "Aggregates", !is.na(region)) |>
  rename(gdp_pc = NY.GDP.PCAP.KD) |>
  mutate(log_gdp_pc = if_else(!is.na(gdp_pc) & gdp_pc > 0, log(gdp_pc), NA_real_)) |>
  select(iso3c, year, gdp_pc, log_gdp_pc) |>
  filter(!is.na(iso3c))

cat(sprintf("Total country-years: %d\n", nrow(gdp_clean)))
cat(sprintf("Unique countries: %d\n", n_distinct(gdp_clean$iso3c)))
cat(sprintf("Years covered: %d-%d\n", min(gdp_clean$year), max(gdp_clean$year)))

# Most recent year with substantial data
recent_coverage <- gdp_clean |>
  group_by(year) |>
  summarise(pct_available = mean(!is.na(gdp_pc)), .groups = "drop") |>
  filter(pct_available >= 0.80) |>
  slice_max(year, n = 1)
cat(sprintf("Most recent year with >=80%% coverage: %d\n", recent_coverage$year))

# Missing by decade
cat("\n% missing log_gdp_pc by decade:\n")
gdp_clean |>
  filter(year >= 1980, year <= 2024) |>
  mutate(decade = floor(year / 10) * 10) |>
  group_by(decade) |>
  summarise(pct_missing = round(mean(is.na(log_gdp_pc)) * 100, 1), .groups = "drop") |>
  print()

write_csv(gdp_clean, file.path(PROCESSED_DIR, "gdp_controls.csv"))
cat(sprintf("\nSaved: %s/gdp_controls.csv (%d rows)\n", PROCESSED_DIR, nrow(gdp_clean)))

# ── PART 1b: Crisis dummies (Laeven & Valencia 2018) ──────────────────────────

cat("\n=== Part 1b: Crisis dummies (Laeven & Valencia 2018) ===\n")

crisis_file <- file.path(RAW_DIR, "SYSTEMIC BANKING CRISES DATABASE_2018.xlsx")
crisis_raw  <- read_excel(crisis_file, sheet = "Crisis Years", skip = 1)

# Rename columns
crisis_raw <- crisis_raw |>
  rename(
    country          = 1,
    banking_years    = 2,
    currency_years   = 3,
    sovereign_years  = 4
  ) |>
  select(country, banking_years, currency_years, sovereign_years) |>
  filter(!is.na(country), country != "(year)")

cat(sprintf("L&V countries in raw file: %d\n", nrow(crisis_raw)))

# Helper: parse comma-separated year strings → integer vector
parse_years <- function(x) {
  if (is.na(x) || x == "") return(integer(0))
  parts <- strsplit(as.character(x), "[,;\\s]+")[[1]]
  years <- suppressWarnings(as.integer(trimws(parts)))
  years[!is.na(years) & years >= 1900 & years <= 2020]
}

# Expand to long format then aggregate to country-year binary flags
expand_crisis <- function(df, col, flag_name) {
  df |>
    select(country, years = {{ col }}) |>
    rowwise() |>
    mutate(year = list(parse_years(years))) |>
    ungroup() |>
    unnest(year) |>
    select(country, year) |>
    mutate(!!flag_name := 1L)
}

banking  <- expand_crisis(crisis_raw, banking_years,   "banking_crisis")
currency <- expand_crisis(crisis_raw, currency_years,  "currency_crisis")
sovereign <- expand_crisis(crisis_raw, sovereign_years, "sovereign_crisis")

# All country-year combinations from 1980-2017
all_years <- crisis_raw |>
  select(country) |>
  crossing(year = 1980:2017)

crisis_panel <- all_years |>
  left_join(banking,   by = c("country", "year")) |>
  left_join(currency,  by = c("country", "year")) |>
  left_join(sovereign, by = c("country", "year")) |>
  mutate(
    banking_crisis  = replace_na(banking_crisis,  0L),
    currency_crisis = replace_na(currency_crisis, 0L),
    sovereign_crisis = replace_na(sovereign_crisis, 0L),
    any_crisis      = as.integer(banking_crisis | currency_crisis | sovereign_crisis)
  )

# Match L&V country names to ISO3 codes
crisis_panel <- crisis_panel |>
  mutate(
    country_code = countrycode(country, origin = "country.name", destination = "iso3c",
                               warn = FALSE)
  )

# Print unmatched countries
unmatched <- crisis_panel |>
  filter(is.na(country_code)) |>
  distinct(country) |>
  pull(country)

if (length(unmatched) > 0) {
  cat("\nL&V country names that failed auto-matching:\n")
  print(unmatched)
}

# Manual crosswalk for known problem cases
manual_map <- c(
  "China, P.R."                    = "CHN",
  "Congo, Dem. Rep. of"            = "COD",
  "Congo, Democratic Republic of"  = "COD",
  "Côte d'Ivoire"                  = "CIV",
  "Cote d'Ivoire"                  = "CIV",
  "Korea"                          = "KOR",
  "Macedonia"                      = "MKD",
  "Serbia, Republic of"            = "SRB",
  "Yugoslavia"                     = NA_character_,   # dissolved — no ISO3
  "Czechoslovakia"                  = NA_character_    # dissolved — no ISO3
)

crisis_panel <- crisis_panel |>
  mutate(
    country_code = case_when(
      country %in% names(manual_map) ~ manual_map[country],
      TRUE ~ country_code
    )
  )

# Remove rows where country_code is still NA (historical/dissolved states)
n_dropped <- sum(is.na(crisis_panel$country_code))
if (n_dropped > 0) {
  dropped_countries <- crisis_panel |> filter(is.na(country_code)) |> distinct(country) |> pull()
  cat(sprintf("\nDropped %d rows for %d dissolved/unresolvable countries: %s\n",
              n_dropped, length(dropped_countries), paste(dropped_countries, collapse = ", ")))
  crisis_panel <- filter(crisis_panel, !is.na(country_code))
}

# Final panel
crisis_out <- crisis_panel |>
  select(country_code, year, any_crisis, banking_crisis, currency_crisis, sovereign_crisis)

cat(sprintf("\nCrisis panel: %d country-years (%d countries, 1980-2017)\n",
            nrow(crisis_out), n_distinct(crisis_out$country_code)))

cat("\nCountry-years with any_crisis = 1 by decade:\n")
crisis_out |>
  mutate(decade = floor(year / 10) * 10) |>
  group_by(decade) |>
  summarise(n_crisis_years = sum(any_crisis), pct = round(mean(any_crisis) * 100, 1), .groups = "drop") |>
  print()

write_csv(crisis_out, file.path(PROCESSED_DIR, "crisis_dummies.csv"))
cat(sprintf("\nSaved: %s/crisis_dummies.csv (%d rows)\n", PROCESSED_DIR, nrow(crisis_out)))
cat("\nPart 1 complete. Validate outputs before running Part 2.\n")
