"""Reconcile known duplicate and spurious records against their pinned source rows."""

# These knockout matches each have two source rows identical except for match_num.
# Keep the smaller match_num. Preserve all other source rows, including round-robin rematches.
DUPLICATE_RECORDS = {
    "tour:2021-416:25": "tour:2021-416:14",
    "tour:2021-321:13": "tour:2021-321:5",
    "qualifying:2011-533:14": "qualifying:2011-533:13",
}
# Zagreb's 2007 qualifying file repeats each of its three knockout rounds.
# These are the 19 duplicated matches that pass the rank and completion filters.
for offset, retained_numbers in (
    (10, (1, 2, 4, 5, 6, 7, 8, 10)),
    (8, (21, 22, 23, 24, 25, 27, 28)),
    (4, (37, 38, 39, 40)),
):
    for number in retained_numbers:
        DUPLICATE_RECORDS[f"qualifying:2007-451:{number + offset}"] = f"qualifying:2007-451:{number}"
# The official Båstad draw gives Robredo a first-round bye. The source copies
# the Paire–Robredo final into R32 as #1. Paire's actual R32 is already #4.
# https://www.protennislive.com/posting/2015/316/mds.pdf
BASTAD_SPURIOUS = "tour:2015-316:1"
BASTAD_FINAL = "tour:2015-316:28"
BASTAD_R32 = "tour:2015-316:4"
CHECK_IDS = set(DUPLICATE_RECORDS) | set(DUPLICATE_RECORDS.values()) | {
    BASTAD_SPURIOUS, BASTAD_FINAL, BASTAD_R32,
}


class DrawReconciliation:
    """Collect evidence for known source errors before verifying each affected season."""

    def __init__(self):
        self.check_rows = {}

    def reason(self, match_id, row):
        if match_id in CHECK_IDS:
            self.check_rows[match_id] = {k: v for k, v in row.items() if k != "match_num"}
        if match_id in DUPLICATE_RECORDS:
            return "duplicate"
        if match_id == BASTAD_SPURIOUS:
            return "spurious"
        return None

    def verify_season(self, source, season):
        prefix = f"{source}:{season}-"
        for duplicate, retained in DUPLICATE_RECORDS.items():
            if duplicate.startswith(prefix):
                twin = self.check_rows.get(duplicate)
                if not twin or twin["round"] == "RR" or twin != self.check_rows.get(retained):
                    raise ValueError(f"Source rows {duplicate} and {retained} no longer verify as duplicates.")
        if not BASTAD_SPURIOUS.startswith(prefix):
            return
        expected_draw = {
            BASTAD_SPURIOUS: ("R32", "PD31", "R419", "7-6(7) 6-3"),
            BASTAD_FINAL: ("F", "PD31", "R419", "7-6(7) 6-3"),
            BASTAD_R32: ("R32", "PD31", "E688", "7-6(4) 6-3"),
        }
        for match_id, expected in expected_draw.items():
            row = self.check_rows.get(match_id, {})
            actual = tuple(row.get(k) for k in ("round", "winner_id", "loser_id", "score"))
            if row.get("tourney_id") != "2015-316" or actual != expected:
                raise ValueError(f"Recheck {match_id} against the official 2015 Båstad draw.")
