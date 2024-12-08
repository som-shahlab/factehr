import pandas as pd
import collections
import numpy as np

df = pd.read_csv("results_without_ds.csv")

# pdac and breastca are progress notes
df["note_type"] = df.note_type.apply(lambda x: "progress_note" if x in ["pdac", "breastca"] else x)

# f1 should only be nan if precision and recall are both zero
assert (df.f1.isna() == (df.precision == 0.0) & (df.recall == 0)).all()

# if f1 is nan, set it to zero
df["f1"] = df.f1.fillna(0)

translate_models = {
    "gemini": "Gemini",
    "medlm": "MedLM",
    "GPT4": "GPT-4",
    "shc-gpt-4o": "GPT-4o",
    "llama3": "Llama3-8b",
}

MODELS = df.model.unique()
DATASETS = df.dataset.unique()
PROMPTS = df.prompt.unique()
NOTE_TYPES = df.note_type.unique()

results = {}
for model in MODELS:
    results[model] = {}
    for note_type in NOTE_TYPES:
        subset_df = df.query(f"model == '{model}' and note_type == '{note_type}'")
        results[model][note_type] = {}
        for statistic in ["f1", "precision", "recall"]:
            col = subset_df[statistic] * 100
            mean = np.round(col.mean(), 1)
            std = np.round(np.std(col), 1)
            results[model][note_type][statistic] = {
                "mean": mean, "std": std,
            }

lines = []
for model in MODELS.tolist():
    line = "\n \\rule{0pt}{12pt} \\textbf{" + translate_models[model] + "} & & \\\\"
    for summary_stat in ["mean", "std"]:
        line += "\n \hspace{0.25cm} \n \\textsc{" + summary_stat + "} & "
        summary_stat_line = []
        for note_type in ["radiology_report", "progress_note", "nursing_note", "discharge_summary"]:
            stats = []
            for statistic in ["precision", "recall", "f1"]:
                stats.append(str(results[model][note_type][statistic][summary_stat]))
            summary_stat_line.append(" \ \ ".join(stats))
        line += " & ".join(summary_stat_line) + "\\\\"
    lines.append(line)
ans = "\n".join(lines)

table_header = """

\\begin{table*}[t!]
    \\small
        \\centering
        \\begin{tabular}{l c c c c}
            \\toprule
             \\multirow{3}{2cm}{\\textbf{Model}} & \\textbf{Radiology Notes} & \\textbf{Progress Notes} & \\textbf{Nursing Notes} & \\textbf{Discharge Summaries}\\\\ [0.5ex]
             \\cmidrule{2-5}
            & Prec. \\ \\ Rec. \\ \\ F1  & Prec. \\ \\ Rec. \\ \\ F1 & Prec. \\ \\ Rec. \\ \\ F1  & Prec. \\ \\ Rec. \\ \\ F1 \\\\
            \\hline 

"""

table_footer = """
\n \\bottomrule
    \\end{tabular}
    \\vspace{0.25cm}
    \\caption{Core results table with all datasets and models. Scores are averaged across prompts and ICL examples.}
    \\label{tab:main_result}
\\end{table*}
"""

# This is the latex source for the prec/recall/F1 version of the table
print(table_header + ans + table_footer)
