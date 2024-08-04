# Import necessary libraries
library(ggplot2)

# Set path to data
setwd("/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results") # nolint

# Import data
data <- read.csv(file = "fst_sims.txt", sep = "\t")

# Extract FST values greater than zero
data <- subset(data, FST > 0)

# Subset data by migration rates
low <- subset(data, Migration_Rate == 1e-09)
medium <- subset(data, Migration_Rate == 1e-01)
high <- subset(data, Migration_Rate == 9e-01)

# Plot overlapping FSTs for each migration rate
density_histogram_plots <- ggplot(data, aes(x = `FST`, fill = factor(Migration_Rate))) + 
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  geom_density(alpha = 0.5) +
  labs(title = "Overlayed FST > 0", x = "Hudson's FST", y = NULL, fill = "Migration Rates") +  # Add titles
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.line = element_line(color = "black"),
    plot.title = element_text(hjust = 0.5, face = "bold")  # Centralize and bold the title
  )

# Save the plot
ggsave("density_histogram_plots.png", plot = density_histogram_plots, width = 10, height = 6, dpi = 300)