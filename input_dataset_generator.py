"""
Generates a high-quality sample dataset of unique long-form paragraph comments.

Output: data/inputs.csv with columns [comment_text, sentiment]
"""

import os
import random
from typing import Tuple

import pandas as pd
from tqdm import tqdm


# --- Persona-Based Master Templates ---
MASTER_TEMPLATES = [
    {
        "sentiment": "positive",
        "template": (
            "We wholeheartedly support the ministry's recent notification regarding the amendments to {topic}. "
            "This is an incredibly encouraging step that clarifies the framework for {stakeholder}s. "
            "The decision to {positive_action} is a welcome change that will undoubtedly benefit thousands of enterprises "
            "and strengthen the overall corporate governance structure, leading to greater investor confidence."
        ),
    },
    {
        "sentiment": "negative",
        "template": (
            "We must express our significant concern regarding the proposed changes to {topic}. "
            "While we understand the intent, the new framework imposes a substantial {negative_impact} that will be detrimental to {stakeholder}s. "
            "The guidelines are ambiguous on several key points, particularly concerning the calculation of penalties, which creates a high degree "
            "of uncertainty and will hinder growth."
        ),
    },
    {
        "sentiment": "neutral",
        "template": (
            "This circular pertains to the updated regulations for {topic}, as mandated under Section {num} of the Companies Act. "
            "The notification, dated last month, states that the due date for submission has been extended. "
            "According to the document, this extension applies to all {stakeholder}s required to report on this matter following a meeting of the regulatory committee."
        ),
    },
    {
        "sentiment": "negative",
        "template": (
            "While we appreciate the intent to {positive_action}, the execution outlined in the draft for {topic} is deeply flawed. "
            "The new rules will unfortunately create a significant {negative_impact} for {stakeholder}s. "
            "There is a major disconnect between the stated goals and the actual text of the legislation, which will likely lead to widespread confusion and an increase in litigation."
        ),
    },
    {
        "sentiment": "positive",
        "template": (
            "The recent clarification on {topic} is a commendable effort by the ministry. "
            "For years, {stakeholder}s have struggled with ambiguity in this area, and this pragmatic reform will {positive_action}. "
            "We believe this will foster a much better investment climate and encourage more compliance from all parties involved, which is a very positive development for the market."
        ),
    },
]


# --- Component Dictionaries for Variation ---
COMPONENTS = {
    "topic": [
        "Corporate Social Responsibility (CSR) spending",
        "the Insolvency and Bankruptcy Code (IBC)",
        "ESG reporting standards",
        "auditor rotation policies",
        "related party transaction disclosures",
        "startup compliance requirements",
        "foreign direct investment (FDI) rules",
        "board diversity mandates",
        "insider trading regulations",
    ],
    "stakeholder": [
        "startups and MSMEs",
        "large listed public companies",
        "foreign institutional investors",
        "chartered accountants and auditors",
        "independent directors",
        "venture capitalists",
        "retail shareholders",
        "industry associations",
    ],
    "positive_action": [
        "streamline the reporting mechanism",
        "simplify the compliance obligations",
        "enhance corporate transparency",
        "level the playing field",
        "provide much-needed clarity",
        "boost investor confidence",
    ],
    "negative_impact": [
        "compliance burden",
        "administrative overhead",
        "market uncertainty",
        "cost structure",
        "regulatory ambiguity",
        "litigation risk",
    ],
}


def fill_template(template_obj: dict) -> str:
    return template_obj["template"].format(
        topic=random.choice(COMPONENTS["topic"]),
        stakeholder=random.choice(COMPONENTS["stakeholder"]),
        positive_action=random.choice(COMPONENTS["positive_action"]),
        negative_impact=random.choice(COMPONENTS["negative_impact"]),
        num=random.randint(10, 500),
    )


def generate_long_comment() -> Tuple[str, str]:
    """Chains multiple templates to create a long, coherent paragraph and returns (text, sentiment)."""
    num_clauses = random.randint(2, 3)
    chosen = random.choices(MASTER_TEMPLATES, k=num_clauses)
    filled = [fill_template(t) for t in chosen]
    final_sentiment = chosen[-1]["sentiment"]
    final_comment_text = " ".join(filled)
    return final_comment_text, final_sentiment


def main(target_rows: int = 100, out_path: str = "data/inputs.csv", seed: int | None = 42) -> None:
    if seed is not None:
        random.seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Generating {target_rows} unique long-form comments -> {out_path} ...")
    unique_entries: set[Tuple[str, str]] = set()
    pbar = tqdm(total=target_rows)
    try:
        while len(unique_entries) < target_rows:
            unique_entries.add(generate_long_comment())
            pbar.update(len(unique_entries) - pbar.n)
    finally:
        pbar.close()

    df = pd.DataFrame(list(unique_entries), columns=["comment_text", "sentiment"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Done. Wrote {len(df)} rows to {out_path}.")
    print("\n--- Sample Entry ---")
    print(df.iloc[0]["comment_text"]) 
    print(f"\nSentiment: {df.iloc[0]['sentiment']}")


if __name__ == "__main__":
    main()
