# R/07_country_coverage_map.R

library(countrycode)
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)
library(tidyverse)
library(conflicted)
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

PROCESSED_DIR <- "data/processed"
FIGURES_DIR   <- "results/figures"
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

imf <- read_csv(file.path(PROCESSED_DIR, "imf_monitor_clean.csv"), show_col_types = FALSE)

arrangement_counts <- imf |>
  distinct(country, arrangement_id) |>
  count(country, name = "n_arrangements") |>
  mutate(
    category = case_when(
      n_arrangements == 1                       ~ "1 program",
      n_arrangements >= 2 & n_arrangements <= 4 ~ "2-4 programs",
      n_arrangements >= 5                       ~ "5+ programs"
    ),
    iso3 = countrycode(country, "country.name", "iso3c")
  )

# REVIEW unmatched names as before
unmatched <- arrangement_counts |> filter(is.na(iso3)) |> pull(country)
if (length(unmatched) > 0) {
  cat("Unmatched country names (review and fix manually):\n")
  print(unmatched)
}

world <- ne_countries(scale = "medium", returnclass = "sf") |>
  left_join(arrangement_counts |> select(iso3, category), by = c("iso_a3" = "iso3")) |>
  mutate(category = factor(
    replace_na(category, "No program"),
    levels = c("No program", "1 program", "2-4 programs", "5+ programs")
  ))

fig_map <- ggplot(world) +
  geom_sf(aes(fill = category), colour = "white", linewidth = 0.1) +
  scale_fill_manual(
    values = c(
      "No program"   = "grey90",
      "1 program"    = "#fed976",
      "2-4 programs" = "#f16913",
      "5+ programs"  = "darkred"
    ),
    name = NULL
  ) +
  labs(
    title    = "IMF Programs by Country, 1980-2019",
    subtitle = "Number of distinct programs per country over the period (IMF Monitor, Kentikelenis & Stubbs 2023)"
  ) +
  coord_sf(crs = "+proj=robin") +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.title    = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(colour = "grey40", size = 10, hjust = 0.5)
  )

ggsave(file.path(FIGURES_DIR, "fig_00_country_coverage_map.png"), fig_map, width = 10, height = 5.5, dpi = 300)