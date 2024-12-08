import pandas as pd
import collections
import numpy as np


def load_df():
    df = pd.read_csv("result_fp_fn_without_ds.csv")

    # pdac and breastca are progress notes
    df["note_type"] = df.note_type.apply(lambda x: "progress_note" if x in ["pdac", "breastca"] else x)

    # f1 should only be nan if precision and recall are both zero
    assert (df.f1.isna() == (df.precision == 0.0) & (df.recall == 0)).all()

    # if f1 is nan, set it to zero
    df["f1"] = df.f1.fillna(0)

    # Make sure there are no nans
    assert len(df.dropna()) == len(df)

    df["FP"] = df.count_zeros_precision / df.total_sentences_precision
    df["FN"] = df.count_zeros_recall / df.total_sentences_recall
    df = df.rename(columns={"f1": "F1"})
    return df


def main_table():
    df = load_df()

    MODELS = df.model.unique()
    NOTE_TYPES = df.note_type.unique()

    results = {"model": MODELS}
    for note_type in NOTE_TYPES:
        for col in ["F1", "FP", "FN"]:
            val = []
            for model in MODELS:
                subset_df = df.query(f"model == '{model}' and note_type == '{note_type}'")
                val.append(np.round((subset_df[col] * 100).mean(), 1))
            results[f"{note_type}_{col}"] = val

    result_df = pd.DataFrame(results)

    result_df.to_csv("f1_fp_fn_results.csv")

    updated_results = {}
    for k, v in results.items():
        if k == "model":
            continue
        for i, model in enumerate(MODELS):
            updated_results[f"{model}_{k}"] = v[i]

    translate_models = {
        "gemini": "Gemini",
        "medlm": "MedLM",
        "GPT4": "GPT-4",
        "shc-gpt-4o": "GPT-4o",
        "llama3": "Llama3-8b",
    }

    lines = []
    for model in MODELS.tolist():
        line = "\n \\rule{0pt}{12pt} \\textbf{" + translate_models[model] + "}"
        for note_type in ["radiology_report", "progress_note", "nursing_note", "discharge_summary"]:
            line += " & "
            stats = []
            for statistic in ["F1", "FP", "FN"]:
                stats.append(str(updated_results.get(f"{model}_{note_type}_{statistic}", "todo")))
            line += " \\ \\ ".join(stats)
        lines.append(line + " \\\\ ")
    ans = "\n".join(lines)

    table_header = '''
    \\begin{table*}[t!]
        \\small
            \\centering
            \\begin{tabular}{l c c c c}
                \\toprule
                \\multirow{3}{2cm}{\\textbf{Model}} & \\textbf{Radiology Notes} & \\textbf{Progress Notes} & \\textbf{Nursing Notes} & \\textbf{Discharge Summaries}\\\\ [0.5ex]
                & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$  & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ \\\\
                \hline \n
    '''

    table_footer = """\n \\bottomrule
        \\end{tabular}
        \\vspace{0.25cm}
        \\caption{Core results table with all datasets and models. Scores are averaged across prompts and ICL examples. \color{red} Scores as of 08-11.}
        \\label{tab:main_result}
    \\end{table*}
    """

    print(table_header + ans + table_footer)


def ablations():
    df = load_df()
    PROMPTS = df.prompt.unique()
    NOTE_TYPES = df.note_type.unique()
    model = "gemini"

    results = {"prompt": PROMPTS}
    for note_type in NOTE_TYPES:
        for col in ["F1", "FP", "FN"]:
            val = []
            for prompt in PROMPTS:
                subset_df = df.query(f"model == '{model}' and note_type == '{note_type}' and prompt == '{prompt}'")
                val.append(np.round((subset_df[col] * 100).mean(), 1))
            results[f"{note_type}_{col}"] = val

    result_df = pd.DataFrame(results)
    import pdb; pdb.set_trace()
    result_df.to_csv("f1_fp_fn_results_prompt_ablation.csv")

    updated_results = {}
    for k, v in results.items():
        if k == "prompt":
            continue
        for i, model in enumerate(PROMPTS):
            updated_results[f"{model}_{k}"] = v[i]

    translate_prompts = {
        "PROMPT1_entailment": "General, zero-shot",
        "PROMPT1_ICL_entailment": "General, two-shot",
        "PROMPT2_entailment": "Clinical, zero-shot",
        "PROMPT2_ICL_entailment": "Clinical, two-shot"
    }

    lines = []
    for prompt in [
        "PROMPT1_entailment",
        "PROMPT1_ICL_entailment",
        "PROMPT2_entailment",
        "PROMPT2_ICL_entailment"
    ]:
        line = "\n \\rule{0pt}{12pt} \\textbf{" + translate_prompts[prompt] + "}"
        for note_type in ["radiology_report", "progress_note", "nursing_note", "discharge_summary"]:
            line += " & "
            stats = []
            for statistic in ["F1", "FP", "FN"]:
                stats.append(str(updated_results.get(f"{prompt}_{note_type}_{statistic}", "todo")))
            line += " \\ \\ ".join(stats)
        lines.append(line + " \\\\ ")
    ans = "\n".join(lines)

    table_header = '''
    \\begin{table*}[t!]
        \\small
            \\centering
            \\begin{tabular}{l c c c c}
                \\toprule
                \\multirow{3}{2cm}{\\textbf{Prompt}} & \\textbf{Radiology Notes} & \\textbf{Progress Notes} & \\textbf{Nursing Notes} & \\textbf{Discharge Summaries}\\\\ [0.5ex]
                & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$  & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ & F1 $\\uparrow$ \\ \\ FP $\\downarrow$ \\ \\ FN $\\downarrow$ \\\\
                \hline \n
    '''

    table_footer = """\n \\bottomrule
        \\end{tabular}
        \\vspace{0.25cm}
        \\caption{Ablation of GPT-4o performance based on prompt and ICL examples. \color{red} Scores as of 08-11.}
        \\label{tab:prompt_ablation}
    \\end{table*}
    """

    print(table_header + ans + table_footer)


ablations()