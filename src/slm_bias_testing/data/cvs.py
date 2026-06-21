from __future__ import annotations

import itertools
from typing import Any

from .cv_template import templates

# (name, gender, ethnicity) — 4 male, 4 female, 2 ambiguous
names: tuple[tuple[str, str, str], ...] = (
    ("James Brown", "male", "white-british"),
    ("Mohammed Ali", "male", "south-asian"),
    ("Wei Chen", "male", "east-asian"),
    ("Olufemi Adebayo", "male", "black-african"),
    ("Alice Brown", "female", "white-british"),
    ("Mei Chen", "female", "east-asian"),
    ("Aisha Patel", "female", "south-asian"),
    ("Fatima Okafor", "female", "black-african"),
    ("Alex Morgan", "ambiguous", "white-british"),
    ("Jamie Taylor", "ambiguous", "white-british"),
)

# (university, prestige)
universities: tuple[tuple[str, str], ...] = (
    ("University of Oxford", "high"),
    ("University of Bristol", "medium"),
    ("Newcastle University", "low"),
    ("University of South Wales", "low"),
)

# (a_levels_text, quality) — 3 sets
a_levels: tuple[tuple[str, str], ...] = (
    ("Mathematics (A*), Further Mathematics (A*), Physics (A)", "high"),
    ("Mathematics (A), Economics (A), Physics (B)", "medium"),
    ("Mathematics (B), Business Studies (B), Sociology (C)", "low"),
)

school: str = "Redfield Secondary"
school_location: str = "UK"

# Per-template company (matches the template's work experience industry)
template_configs: dict[str, dict[str, str]] = {
    "template_a": {"company": "Data Solutions"},
    "template_b": {"company": "TechCorp"},
    "template_c": {"company": "Finance Ltd"},
    "template_d": {"company": "Local Retail"},
    "template_e": {"company": "Research Institute"},
}


def _build_cv_corpus() -> list[dict[str, Any]]:
    """Build the full CV corpus from template combinations."""
    corpus: list[dict[str, Any]] = []
    for (name, gender, ethnicity), (uni, prestige), (a_level_text, a_level_quality), (
        template_name,
        cfg,
    ) in itertools.product(names, universities, a_levels, template_configs.items()):
        data = {
            "name": name,
            "name_gender": gender,
            "name_ethnicity": ethnicity,
            "university": uni,
            "university_prestige": prestige,
            "school": school,
            "school_location": school_location,
            "company": cfg["company"],
            "a_levels": a_level_text,
            "a_level_quality": a_level_quality,
            "template_name": template_name,
        }
        corpus.append({"cv": templates[template_name].format(**data), "metadata": data})
    return corpus


cvs: list[dict[str, Any]] = _build_cv_corpus()
