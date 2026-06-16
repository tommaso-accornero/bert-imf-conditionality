# R/utils/plotting_theme.R
# Custom ggplot2 theme for all thesis figures. Source at top of every R script.

theme_thesis <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(colour = "grey40", size = 11),
      axis.title       = element_text(size = 11),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom"
    )
}
