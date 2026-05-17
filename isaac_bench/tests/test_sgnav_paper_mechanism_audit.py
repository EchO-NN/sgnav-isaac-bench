from pathlib import Path


REQUIRED_MECHANISMS = [
    "RGB-D observation",
    "GLIP open-vocabulary detection",
    "SAM mask refinement",
    "online occupancy/free map",
    "frontier extraction",
    "object nodes",
    "group nodes",
    "room nodes",
    "object-room edges",
    "object-centered subgraphs",
    "HCoT stage 1",
    "HCoT stage 2",
    "HCoT stage 3",
    "HCoT stage 4",
    "P_sub",
    "frontier interpolation",
    "distance bias",
    "candidate re-perception",
    "STOP",
]

ALLOWED_STATUS = {
    "faithful",
    "optimized-equivalent",
    "engineered-faithful",
    "approximate-but-justified",
}


def test_sgnav_paper_mechanism_audit_has_required_rows_and_honest_statuses():
    text = Path("docs/sgnav_paper_mechanism_audit.md").read_text(encoding="utf-8")

    for mechanism in REQUIRED_MECHANISMS:
        assert mechanism in text
    assert "incomplete" not in text.lower()

    table_lines = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| ---")]
    rows = [line for line in table_lines if "SG-Nav paper mechanism" not in line]
    assert len(rows) >= len(REQUIRED_MECHANISMS)
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) < 4:
            continue
        status = cells[2]
        assert status in ALLOWED_STATUS
        assert cells[3]


def test_audit_documents_lazy_room_context_and_vlm_confidence_policy():
    text = Path("docs/sgnav_paper_mechanism_audit.md").read_text(encoding="utf-8")
    lowered = text.lower()

    assert "prepare_room_context_for_frontier_scoring" in text
    assert "candidate re-perception uses the latest cached graph context" in lowered
    assert "vlm_self_confidence" in text
    assert "label_reliability" in text
