from pathlib import Path
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

TITLE = "Weekly Progress Report"
FILENAME = "Sommayadeep_Week-01.pdf"

content = """
Name: Sommayadeep Saha
Domain: Machine Learning / Predictive Maintenance
Date of Submission: __________________
Week Ending: 01
Project: Predict the number of remaining operational cycles before failure for Turbofan engine

I. Overview
This week, I worked on building an end-to-end Remaining Useful Life (RUL) prediction pipeline for turbofan engines using NASA C-MAPSS data. I completed environment setup, data preparation, model training, evaluation, and report generation.

II. Achievements
1. Project Setup and Understanding
- Understood problem statement and dataset structure (FD001-FD004).
- Set up virtual environment and installed required libraries (numpy, pandas, scikit-learn, matplotlib, seaborn, joblib).

2. Implementation Work
- Implemented training and evaluation scripts for baseline and improved models.
- Ran experiments across all subsets (FD001, FD002, FD003, FD004).
- Generated final outputs:
  - results/metrics.csv
  - results/improvement_summary.csv
  - visualization plots in results/figures/
  - final technical report REPORT.md

3. Model Performance Summary
- Baseline and improved model metrics were generated for all four subsets.
- Improved model showed RMSE gains across all subsets.

III. Challenges and Hurdles
1. Dependency Installation Timeout
- Faced pip ReadTimeoutError while installing packages.
- Solved by upgrading pip and using retry + timeout options.

2. Dataset File Path Issues
- Training failed initially due to missing data/raw/train_FD00X.txt files.
- Also faced incorrect/old ZIP path and nested ZIP extraction confusion.
- Solved by locating correct ZIP and extracting required files into data/raw/.

3. Command Execution Mistakes
- Accidentally pasted output lines as shell commands, causing zsh: command not found.
- Corrected by running only valid command blocks.

IV. Lessons Learned
- Learned practical debugging of Python and shell workflow errors.
- Improved understanding of data pipeline dependencies (correct path + file names).
- Gained hands-on experience in reproducible ML experimentation and reporting.
- Learned to systematically verify outputs after each pipeline step.

V. Next Week's Goals
1. Perform hyperparameter tuning for better MAE on FD004.
2. Add deeper analysis of feature importance and error patterns.
3. Prepare final presentation slides with methodology, results, and conclusions.
4. Improve report formatting and add concise interpretation of plots.

VI. Additional Comments
This week established a complete working pipeline from raw data to final results and report artifacts. Major blockers were related to setup and data extraction, and all were resolved successfully. The project is now in a stable state for further optimization and final presentation.
""".strip()

out_path = Path(FILENAME)

lines = []
for para in content.split("\n"):
    if not para.strip():
        lines.append("")
        continue
    wrapped = textwrap.wrap(para, width=100) or [""]
    lines.extend(wrapped)

with PdfPages(out_path) as pdf:
    i = 0
    while i < len(lines):
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        y = 0.96
        if i == 0:
            ax.text(0.5, y, TITLE, ha="center", va="top", fontsize=16, fontweight="bold")
            y -= 0.05

        while i < len(lines) and y > 0.05:
            line = lines[i]
            fs = 10.5
            if line.startswith(("I.", "II.", "III.", "IV.", "V.", "VI.")):
                fs = 12
            ax.text(0.07, y, line, ha="left", va="top", fontsize=fs, family="DejaVu Sans")
            y -= 0.022
            i += 1

        pdf.savefig(fig)
        plt.close(fig)

print(f"Created {out_path.resolve()}")
