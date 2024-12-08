""" 

"""

import pandas as pd


def get_groupby_stats(df: pd.DataFrame, group_key: str):
    # per-document precision, across all models and prompts
    stats = df.groupby(group_key).agg(
        {
            "fact_precision": ["mean", "var", "min", "max"],
            "fact_recall": ["mean", "var", "min", "max"],
            "fact_f1": ["mean", "var", "min", "max"],
            "fact_fp": ["mean", "var", "min", "max"],
            "fact_fn": ["mean", "var", "min", "max"],
        }
    )
    stats.columns = ["_".join(col) for col in stats.columns]
    stats.columns = [
        col.replace("precision", "p").replace("recall", "r") for col in stats.columns
    ]

    stats_sorted = stats.sort_values(by="fact_f1_mean", ascending=False)
    return stats_sorted


df = pd.read_csv("data/manuscript/entailment_pair_scores.tsv", sep="\t")
# HACK remove incomplete experiment rows
df = df.dropna(subset=["fact_precision", "fact_recall"])
# nan_rows = df[df["note_text_hash"].isna()]

n_docs = len(df.note_text_hash.unique())

# per-document precision, across all models and prompts
stats = get_groupby_stats(df, "note_text_hash")
print(stats)

stats = get_groupby_stats(df, "note_type")
print(stats)

stats = get_groupby_stats(df, "llm")
print(stats)

stats = get_groupby_stats(df, "prompt")
print(stats)


#
# Fact Generation
#

# 4 x 5 LLMs

fact_df = df.sort_values(by=df.columns[0])
print(fact_df[["note_text_hash", "n_facts"]])

stats = df.groupby(["note_text_hash", "prompt"]).agg(
    {
        "n_facts": ["min", "max"],
    }
)
print(stats)
