# R/02_category_mapping.R
# Phase 1: Map MONA Economic Descriptors → IMF Monitor 13 categories.
# Output: data/raw/category_mapping.csv, data/processed/mona_with_labels.csv

library(tidyverse)
library(conflicted)

set.seed(42)

RAW_DIR       <- "data/raw"
PROCESSED_DIR <- "data/processed"

# ── 1. Load MONA clean ─────────────────────────────────────────────────────────

mona <- read_csv(file.path(PROCESSED_DIR, "mona_clean.csv"), show_col_types = FALSE)

# ── 2. Descriptor counts (for reference / documentation) ─────────────────────

descriptor_counts <- mona |>
  count(descriptor, sort = TRUE) |>
  rename(n_conditions = n)

cat("Unique MONA descriptors:", nrow(descriptor_counts), "\n")

# ── 3. Manual mapping: MONA descriptor → IMF Monitor 13-category ──────────────
#
# IMF Monitor 13 categories:
#   FIN  Financial sector
#   DEB  Debt
#   FP   Fiscal policy
#   RTP  Revenue/Tax policy
#   EXT  External sector
#   SOE  State-owned enterprise
#   LAB  Labour
#   PRI  Privatisation
#   INS  Institutional
#   POV  Poverty
#   SP   Social Protection
#   OTH  Other
#   ENV  Environment
#
# Confidence levels:
#   HIGH   — unambiguous; descriptor maps cleanly to one category
#   MEDIUM — reasonable primary mapping but spans multiple categories
#   LOW    — genuinely ambiguous; flag for Ignacio review
#
# Decision log for ambiguous cases is printed at the end of this script.

mapping_raw <- tribble(
  ~descriptor,                                                                          ~imf_category, ~confidence, ~notes,

  # ── FINANCIAL SECTOR ──────────────────────────────────────────────────────────
  "6.1. Financial sector legal reforms, regulation, and supervision",                   "FIN",         "HIGH",      "",
  "2.1. Central bank operations and reforms",                                           "FIN",         "HIGH",      "",
  "2.2. Central bank auditing, transparency, and financial controls",                   "FIN",         "HIGH",      "",
  "6. Financial sector",                                                                "FIN",         "HIGH",      "Catch-all financial sector heading; FIN is correct",
  "2. Central Bank",                                                                    "FIN",         "HIGH",      "Catch-all central bank heading",
  "6.2. Restructuring and privatization of financial institutions",                     "FIN",         "MEDIUM",    "PRIMARY: FIN (financial institutions). Could be PRI. Financial sector context takes precedence here.",

  # ── DEBT ─────────────────────────────────────────────────────────────────────
  "1.5. Debt Management",                                                               "DEB",         "HIGH",      "See 7-cat decision below",

  # ── FISCAL POLICY ─────────────────────────────────────────────────────────────
  "1.6. Expenditure auditing, accounting, and financial controls",                      "FP",          "HIGH",      "",
  "1.3. Expenditure measures, including arrears clearance",                             "FP",          "HIGH",      "",
  "1.8. Budget preparation (e.g., submission or approval)",                             "FP",          "HIGH",      "",
  "1. General government",                                                              "FP",          "HIGH",      "Catch-all fiscal heading; FP is the best fit",
  "1.4. Combined expenditure and revenue measures",                                     "FP",          "MEDIUM",    "Spans FP and RTP. Assigned FP as the broader fiscal category.",
  "1.7. Fiscal transparency (publication, parliamentary oversight)",                    "FP",          "MEDIUM",    "PRIMARY: FP (fiscal governance). Could be INS. Transparency is about fiscal management, not institutional reform.",
  "1.9. Inter-governmental relations",                                                  "INS",         "MEDIUM",    "PRIMARY: INS (fiscal federalism/governance). Could be FP but structural/governance framing dominates.",

  # ── REVENUE / TAX POLICY ─────────────────────────────────────────────────────
  "1.1. Revenue measures, excluding trade policy",                                      "RTP",         "HIGH",      "",
  "1.2. Revenue administration, including customs",                                     "RTP",         "HIGH",      "Customs admin is revenue, not trade policy per descriptor label",

  # ── EXTERNAL SECTOR ───────────────────────────────────────────────────────────
  "7. Exchange systems and restrictions (current and capital)",                         "EXT",         "HIGH",      "",
  "8. International trade policy, excluding customs reforms",                           "EXT",         "HIGH",      "",

  # ── STATE-OWNED ENTERPRISE ────────────────────────────────────────────────────
  "5.1. Public enterprise pricing and subsidies",                                       "SOE",         "HIGH",      "",
  "5. Public enterprise reform and pricing  (non financial sector)",                    "SOE",         "HIGH",      "Catch-all SOE heading",
  "5.3. Price controls and marketing restrictions",                                     "SOE",         "MEDIUM",    "Under MONA section 5 (public enterprise). Could be EXT (trade). SOE assigned given section context.",

  # ── LABOUR ────────────────────────────────────────────────────────────────────
  "9. Labor markets, excluding public sector employment",                               "LAB",         "HIGH",      "",
  "3. Civil service and public employment reforms, and wages",                          "LAB",         "MEDIUM",    "PRIMARY: LAB (employment conditions, wages). Could be INS (structural reform of civil service). Wage/employment dimension is the primary IMF concern.",

  # ── PRIVATISATION ─────────────────────────────────────────────────────────────
  "5.2. Privatization, public enterprise reform and restructuring, other than pricing",  "PRI",         "HIGH",      "",

  # ── INSTITUTIONAL ─────────────────────────────────────────────────────────────
  "11.1. Private sector legal and regulatory environment reform (non financial sector)", "INS",         "HIGH",      "",
  "11.4. Anti-corruption legislation/policy",                                           "INS",         "MEDIUM",    "PRIMARY: INS (governance reform). Could be OTH. Anti-corruption is institutional capacity; INS preferred over catch-all OTH.",
  "10. Economic statistics (excluding fiscal and central bank transparency and similar measures)", "INS", "HIGH",   "Statistical capacity is an institutional/governance condition",

  # ── POVERTY ───────────────────────────────────────────────────────────────────
  "11.3. PRSP development and implementation",                                          "POV",         "HIGH",      "PRSP = Poverty Reduction Strategy Paper; POV is unambiguous",

  # ── SOCIAL PROTECTION ─────────────────────────────────────────────────────────
  "4.1. Pension reforms",                                                               "SP",          "HIGH",      "",
  "4.2. Other social sector reforms (e.g., social safety nets, health and education)",  "SP",          "HIGH",      "",
  "4. Pension and other social sector reforms",                                         "SP",          "HIGH",      "Catch-all social sector heading",

  # ── OTHER ─────────────────────────────────────────────────────────────────────
  "11. Other structural measures",                                                      "OTH",         "HIGH",      "Residual MONA category maps to residual IMF category",

  # ── ENVIRONMENT ───────────────────────────────────────────────────────────────
  "11.2. Natural resource and agricultural policies (excl. public enterprises and pricing)", "ENV",    "MEDIUM",    "PRIMARY: ENV (natural resource governance). Could be OTH. Natural resource policy is the closest IMF Monitor analogue to ENV."
)

# ── 4. Merge counts into mapping ──────────────────────────────────────────────

category_mapping <- mapping_raw |>
  left_join(descriptor_counts, by = "descriptor") |>
  arrange(imf_category, desc(n_conditions))

# ── 5. Validation: check all MONA descriptors are covered ─────────────────────

unmapped <- descriptor_counts |>
  anti_join(category_mapping, by = "descriptor")

if (nrow(unmapped) > 0) {
  cat("\nWARNING: unmapped descriptors found:\n")
  print(unmapped)
  stop("Fix mapping before proceeding.")
} else {
  cat("All", nrow(descriptor_counts), "descriptors mapped successfully.\n")
}

# ── 6. Decision log: DEB in 7-category scheme ─────────────────────────────────
#
# DECISION: DEB → FISCAL in the 7-category collapse.
#
# Debt management conditions (borrowing ceilings, debt reporting obligations,
# debt sustainability frameworks) are central fiscal compliance instruments.
# Assigning DEB to GOVERNANCE would conflate mandatory debt compliance with
# discretionary institutional reform. DEB is therefore grouped with FP and RTP
# under FISCAL.
#
# The 7-cat COLLAPSE_MAP is:
#   FISCAL        = FP + RTP + DEB
#   FINANCIAL     = FIN
#   EXTERNAL      = EXT
#   PRIVATISATION = SOE + PRI
#   LABOUR        = LAB
#   SOCIAL        = SP + POV
#   GOVERNANCE    = INS + ENV + OTH

collapse_map <- c(
  FP  = "FISCAL",   RTP = "FISCAL",
  FIN = "FINANCIAL",
  EXT = "EXTERNAL",
  SOE = "PRIVATISATION", PRI = "PRIVATISATION",
  LAB = "LABOUR",
  SP  = "SOCIAL",   POV = "SOCIAL",
  INS = "GOVERNANCE", ENV = "GOVERNANCE", OTH = "GOVERNANCE",
  DEB = "FISCAL"       # see decision log above
)

category_mapping <- category_mapping |>
  mutate(collapsed_7cat = collapse_map[imf_category])

# ── 7. Summary by IMF category ────────────────────────────────────────────────

cat("\n=== Mapping summary (IMF Monitor 13 categories) ===\n")
category_mapping |>
  group_by(imf_category) |>
  summarise(
    n_descriptors = n(),
    n_conditions  = sum(n_conditions, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(n_conditions)) |>
  print(n = Inf)

cat("\nMEDIUM/LOW confidence cases (flag for Ignacio):\n")
category_mapping |>
  filter(confidence != "HIGH") |>
  select(descriptor, imf_category, confidence, notes) |>
  print(n = Inf, width = 120)

# ── 8. Save category_mapping.csv ──────────────────────────────────────────────

write_csv(category_mapping, file.path(RAW_DIR, "category_mapping.csv"))
cat(sprintf("\nSaved: %s/category_mapping.csv (%d rows)\n", RAW_DIR, nrow(category_mapping)))

# ── 9. Apply mapping to MONA → mona_with_labels.csv ──────────────────────────

mona_labeled <- mona |>
  left_join(
    category_mapping |> select(descriptor, imf_category, collapsed_7cat, confidence),
    by = "descriptor"
  )

# Check for any join failures
n_missing_label <- sum(is.na(mona_labeled$imf_category))
if (n_missing_label > 0) {
  cat(sprintf("\nWARNING: %d rows could not be labeled. Check descriptor strings for whitespace/encoding.\n",
              n_missing_label))
  mona_labeled |>
    filter(is.na(imf_category)) |>
    distinct(descriptor) |>
    print()
} else {
  cat(sprintf("All %d MONA rows labeled successfully.\n", nrow(mona_labeled)))
}

cat("\nLabeled MONA — category distribution (all years):\n")
mona_labeled |>
  count(imf_category, sort = TRUE) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print(n = Inf)

cat("\nLabeled MONA — 7-category distribution (all years):\n")
mona_labeled |>
  count(collapsed_7cat, sort = TRUE) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print(n = Inf)

write_csv(mona_labeled, file.path(PROCESSED_DIR, "mona_with_labels.csv"))
cat(sprintf("\nSaved: %s/mona_with_labels.csv (%d rows)\n", PROCESSED_DIR, nrow(mona_labeled)))
cat("\nPhase 1 (category mapping) complete.\n")
