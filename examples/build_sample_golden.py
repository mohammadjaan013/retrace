"""Regenerate `src/retrace/sample_data/golden.jsonl`.

Spans are located by searching for an anchor string rather than being typed by
hand, so the fixture stays correct when a sample document is edited. Run this
after changing `src/retrace/sample_data/documents.json`.

    python examples/build_sample_golden.py
"""

from __future__ import annotations

import json

from retrace import SAMPLE_DIR, EvalCase, Provenance, Span, fixture_stats, write_fixture

DATA = SAMPLE_DIR

# (case_id, question, answer, doc_id, anchor, labels)
#
# `anchor` is a verbatim slice of the source document. The span is derived from
# where it is found, which keeps the label exact without hand-counting offsets.
SPECS = [
    (
        "q_mli_material",
        "What are multilayer insulation blankets made of?",
        "A stack of aluminised polyimide sheets separated by a low-conductance net.",
        "tm-001-thermal",
        "Multilayer insulation blankets, typically a stack of aluminised polyimide",
        {"difficulty": "easy", "expect": "lexical overlap is strong"},
    ),
    (
        "q_radiator_sizing",
        "What determines how large a radiator needs to be?",
        "The worst-case hot condition, not the average.",
        "tm-001-thermal",
        "Radiator area is almost always sized by the worst-case hot condition",
        {"difficulty": "easy"},
    ),
    (
        "q_heat_rejection_synonym",
        "How does a spacecraft get rid of excess warmth in orbit?",
        "It radiates it away as infrared; there is no convection in vacuum.",
        "tm-001-thermal",
        "In vacuum there is no convection, so essentially all waste heat",
        {
            "difficulty": "hard",
            "expect": "vocabulary mismatch: 'get rid of warmth' vs 'reject heat'",
        },
    ),
    (
        "q_absorptivity_ratio",
        "Why does the ratio of solar absorptivity to infrared emissivity matter?",
        "A low ratio keeps a sunlit surface cool; a high ratio helps reject heat.",
        "tm-001-thermal",
        "Surface finishes are selected for their ratio of solar absorptivity",
        {"difficulty": "medium"},
    ),
    (
        "q_battery_cycles",
        "Why does a low Earth orbit satellite wear out its batteries faster?",
        "It passes through roughly fifteen eclipses a day, accumulating cycles quickly.",
        "tm-002-power",
        "a low Earth orbit vehicle passing through roughly fifteen eclipses per day",
        {"difficulty": "medium"},
    ),
    (
        "q_eol_margin",
        "Why are solar arrays specified against end-of-life rather than nominal demand?",
        "Output degrades from radiation, thermal cycling and contamination.",
        "tm-002-power",
        "Array output degrades over the mission from radiation damage",
        {"difficulty": "medium"},
    ),
    (
        "q_wheel_desaturation",
        "How is stored momentum removed from reaction wheels?",
        "With magnetic torque rods or thrusters.",
        "tm-003-attitude",
        "with magnetic torque rods or thrusters used to desaturate the wheels",
        {"difficulty": "medium"},
    ),
    (
        "q_star_tracker_weakness",
        "What is the main limitation of a star tracker?",
        "It needs an unobstructed view and can be blinded by the Sun or a bright limb.",
        "tm-003-attitude",
        "Star trackers give the most accurate absolute attitude but need",
        {"difficulty": "easy"},
    ),
    (
        "q_path_loss_frequency",
        "How does carrier frequency affect free-space path loss?",
        "Loss grows with the square of both distance and frequency.",
        "tm-004-comms",
        "Free-space path loss grows with the square of both distance and frequency",
        {"difficulty": "medium"},
    ),
    (
        "q_link_margin",
        "How much link margin is conventional?",
        "At least three decibels over the required signal-to-noise ratio.",
        "tm-004-comms",
        "A margin of at least three decibels over the required ratio is conventional",
        {"difficulty": "easy"},
    ),
]

# Answerable from the wider technical literature, but the document holding the
# answer was never indexed. Exercises CORPUS_GAP: no amount of retrieval tuning
# can fix these, and telling a team so is the most valuable output Retrace has.
GAP_SPECS = [
    (
        "q_hall_thruster",
        "What propellant does a Hall-effect thruster typically use?",
        "Xenon.",
        "tm-005-propulsion",
        {"difficulty": "n/a", "expect": "CORPUS_GAP: propulsion doc not indexed"},
    ),
    (
        "q_hohmann",
        "How many burns does a Hohmann transfer require?",
        "Two.",
        "tm-006-orbits",
        {"difficulty": "n/a", "expect": "CORPUS_GAP: orbits doc not indexed"},
    ),
]


def main() -> None:
    documents: dict[str, str] = json.loads(
        (DATA / "documents.json").read_text(encoding="utf-8")
    )

    cases: list[EvalCase] = []

    for case_id, question, answer, doc_id, anchor, labels in SPECS:
        text = documents[doc_id]
        start = text.find(anchor)
        if start < 0:
            raise SystemExit(
                f"anchor for {case_id!r} not found in {doc_id!r}:\n  {anchor!r}\n"
                "Edit the anchor to match the current document text."
            )
        # Extend to the end of the sentence so the span covers the whole answer.
        stop = text.find(". ", start + len(anchor))
        end = len(text) if stop < 0 else stop + 1

        cases.append(
            EvalCase(
                id=case_id,
                question=question,
                answer=answer,
                gold_spans=(Span(doc_id, start, end),),
                gold_doc_ids=(doc_id,),
                provenance=Provenance.HUMAN,
                labels=labels,
            )
        )

    for case_id, question, answer, missing_doc, labels in GAP_SPECS:
        cases.append(
            EvalCase(
                id=case_id,
                question=question,
                answer=answer,
                gold_doc_ids=(missing_doc,),
                provenance=Provenance.HUMAN,
                labels=labels,
            )
        )

    out = DATA / "golden.jsonl"
    written = write_fixture(out, cases)
    print(f"wrote {written} cases -> {out}")
    for key, value in fixture_stats(cases).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
