"""Verified tournament start dates and court surfaces for five singles draws.

The (date, surface) pairs describe an entire draw. For each entry the players in
every opening-round pairing matched the linked official draw. The date is the
start of the tournament week printed in its header, including for qualifying.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawCorrection:
    name: str
    level: str
    reported: tuple[str, str]
    verified: tuple[str, str]
    draw_url: str


CORRECTIONS = {
    ("qualifying", "2025-8996"): DrawCorrection(
        "Santiago", "250", ("20250224", "Hard"), ("20250224", "Clay"),
        "https://www.protennislive.com/posting/2025/8996/qs.pdf",
    ),
    ("challenger", "2024-7740"): DrawCorrection(
        "Nottingham", "C", ("20240211", "Grass"), ("20240610", "Grass"),
        "https://www.protennislive.com/posting/2024/7740/mds.pdf",
    ),
    ("challenger", "2021-444"): DrawCorrection(
        "Bratislava", "C", ("20210613", "Clay"), ("20211108", "Hard"),
        "https://www.protennislive.com/posting/2021/444/mds.pdf",
    ),
    ("challenger", "2023-444"): DrawCorrection(
        "Bratislava 1", "C", ("20230618", "Hard"), ("20231009", "Hard"),
        "https://www.protennislive.com/posting/2023/444/mds.pdf",
    ),
    ("challenger", "2024-444"): DrawCorrection(
        "Bratislava 1", "C", ("20240616", "Hard"), ("20241028", "Hard"),
        "https://www.protennislive.com/posting/2024/444/mds.pdf",
    ),
}


def corrected_inventory_row(row, source, season):
    """Return effective metadata while preserving every field in the source row."""
    event = row["tourney_id"]
    correction = CORRECTIONS.get((source, event))
    if correction is None:
        return row

    identity = (season, row["tourney_name"], row["tourney_level"])
    expected = (int(event[:4]), correction.name, correction.level)
    if identity != expected:
        raise ValueError(f"Unexpected identity for {source} {event}: {identity}. Recheck {correction.draw_url}")

    observed = (row["tourney_date"], row["surface"])
    if observed == correction.verified:
        return row
    if observed != correction.reported:
        raise ValueError(f"Unexpected date/surface for {source} {event}: {observed}. Recheck {correction.draw_url}")
    date, surface = correction.verified
    return dict(row, tourney_date=date, surface=surface)
