# Import necessary libraries
library(ggplot2)
install.packages("dplyr")  # Run this line if dplyr is not installed
library(dplyr)

# Set path to data
setwd("/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results")

# Import data
data <- read.csv(file = "fst_sims.txt", sep = "\t")

# Extract FST values greater than zero
data <- subset(data, FST > 0)

# Subset data by migration rates
low <- subset(data, Migration_Rate == 1e-09)
medium <- subset(data, Migration_Rate == 1e-01)
high <- subset(data, Migration_Rate == 9e-01)

# Function to calculate mode
calculate_mode <- function(x) {
  dens <- density(x)
  mode_index <- which.max(dens$y)
  mode_value <- dens$x[mode_index]
  return(mode_value)
}

# Calculate mode for each migration rate
low_mode <- calculate_mode(low$FST)
medium_mode <- calculate_mode(medium$FST)
high_mode <- calculate_mode(high$FST)

# Filter and count FST values between 0 and 0.001 for each migration rate
fst_count <- data %>%
  filter(FST > 0 & FST <= 0.001) %>%
  group_by(Migration_Rate) %>%
  summarize(count = n())

# Plot overlapping FSTs for each migration rate
density_histogram_plots <- ggplot(data, aes(x = `FST`, fill = factor(Migration_Rate))) + 
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  geom_density(alpha = 0.5) +
  geom_vline(aes(xintercept = low_mode), color = "blue", linetype = "dashed", size = 1.01, show.legend = FALSE) +
  geom_vline(aes(xintercept = medium_mode), color = "green", linetype = "dashed", size = 1, show.legend = FALSE) +
  geom_vline(aes(xintercept = high_mode), color = "red", linetype = "dashed", size = 1, show.legend = FALSE) +
  labs(x = "Hudson's FST", y = NULL, fill = "Migration Rates") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.line = element_line(color = "black"),
    axis.title.x = element_text(size = 24, face = "bold", margin = margin(t = 15)),
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    legend.text = element_text(size = 18),
    legend.title = element_text(size = 18),
    plot.title = element_blank(),
    legend.position = c(0.895, 0.875)
  )

# Set path to save plot
setwd("/Users/jeremiahmushtaq/Documents/University/MSc Research Project")

# Save the plot
ggsave("fst_plot.png", plot = density_histogram_plots, width = 10, height = 6, dpi = 300)
