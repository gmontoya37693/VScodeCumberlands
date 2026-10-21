# Low Margin Business Analysis

An R and Quarto report template for evaluating how a price increase changes
break-even sales volume across businesses with different margins.

## Getting started

1. Open `Low Margin Business Analysis.Rproj` in RStudio.
2. Edit the scenario values and company data in `pricing-analysis.qmd`.
3. Render the report with Quarto:

   ```sh
   quarto render pricing-analysis.qmd
   ```

The report uses base R and does not require additional R packages.

## Model scope

The break-even calculation assumes unit costs stay constant, the price increase
applies uniformly, and lost volume is measured against the original volume. It
compares total contribution profit before and after the price change; it does
not predict customer response. Use company-specific data and test demand
response before making a pricing decision.

The video/McKinsey study reference is not independently verified in this
template. Add a citation and validate the study details before using that claim
in a published report.
