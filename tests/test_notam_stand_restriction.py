import unittest
from datetime import datetime, time, timezone

from zoneinfo import ZoneInfo

from rampflow.notam import Notam, NotamRegistry, Recurrence
from rampflow.stand_orchestration import (
    StandConflict,
    StandDemand,
    StandOrchestrator,
    StandProfile,
    StandReassignment,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc


class NotamStandRestrictionTests(unittest.TestCase):
    def test_recurrence_follows_local_window_across_midnight(self):
        effective = Window(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 2, 1, tzinfo=UTC),
        )
        notam = Notam(
            "N1",
            "stand",
            "S1",
            effective,
            1,
            "closed",
            Recurrence(
                frozenset(range(7)),
                time(22),
                time(2),
                "Asia/Shanghai",
            ),
        )
        horizon = Window(
            datetime(2026, 1, 5, 13, tzinfo=UTC),
            datetime(2026, 1, 6, 13, tzinfo=UTC),
        )

        effects = NotamRegistry([notam]).effects("stand", "S1", horizon)

        self.assertEqual(len(effects), 1)
        self.assertEqual(
            effects[0].window.start,
            datetime(2026, 1, 5, 22, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
        self.assertEqual(
            effects[0].window.end,
            datetime(2026, 1, 6, 2, tzinfo=ZoneInfo("Asia/Shanghai")),
        )

    def test_reverse_adjacency_blocks_simultaneous_assignment(self):
        orchestrator = StandOrchestrator(
            [
                StandProfile("S1", "T", 80, True, 1),
                StandProfile(
                    "S2",
                    "T",
                    80,
                    True,
                    1,
                    incompatible_simultaneous=frozenset({"S1"}),
                ),
            ]
        )
        window = Window(datetime(2026, 1, 1, 10, tzinfo=UTC), datetime(2026, 1, 1, 11, tzinfo=UTC))

        orchestrator.commit(orchestrator.plan(StandDemand("F1", window, 50, False)))

        self.assertEqual(
            orchestrator.candidates(StandDemand("F2", window, 50, False)),
            (),
        )

    def test_batch_reassignment_applies_all_moves_together(self):
        orchestrator = StandOrchestrator(
            [StandProfile(stand_id, "T", 80, True, 1) for stand_id in ("S1", "S2", "S3", "S4")]
        )
        window = Window(datetime(2026, 1, 1, 10, tzinfo=UTC), datetime(2026, 1, 1, 12, tzinfo=UTC))
        for flight_id in ("F1", "F2", "F3"):
            orchestrator.commit(orchestrator.plan(StandDemand(flight_id, window, 50, False)))
        tow_start = datetime(2026, 1, 1, 10, tzinfo=UTC)

        moves = orchestrator.reassign_many(
            [
                StandReassignment("F1", "S2", tow_start, 1, "NOTAM closure"),
                StandReassignment("F2", "S1", tow_start, 1, "NOTAM closure"),
            ]
        )

        self.assertEqual([(move.flight_id, move.to_stand) for move in moves], [("F1", "S2"), ("F2", "S1")])
        self.assertEqual(orchestrator.assignments["F3"].stand_id, "S3")

    def test_failed_batch_reassignment_leaves_existing_allocation_intact(self):
        orchestrator = StandOrchestrator(
            [StandProfile(stand_id, "T", 80, True, 1) for stand_id in ("S1", "S2", "S3", "S4")]
        )
        window = Window(datetime(2026, 1, 1, 10, tzinfo=UTC), datetime(2026, 1, 1, 12, tzinfo=UTC))
        for flight_id in ("F1", "F2", "F3"):
            orchestrator.commit(orchestrator.plan(StandDemand(flight_id, window, 50, False)))
        before = {
            flight_id: assignment.stand_id
            for flight_id, assignment in orchestrator.assignments.items()
        }
        tow_start = datetime(2026, 1, 1, 10, tzinfo=UTC)

        with self.assertRaises(StandConflict):
            orchestrator.reassign_many(
                [
                    StandReassignment("F1", "S3", tow_start, 1, "NOTAM closure"),
                    StandReassignment("F2", "S4", tow_start, 1, "NOTAM closure"),
                ]
            )

        self.assertEqual(
            {
                flight_id: assignment.stand_id
                for flight_id, assignment in orchestrator.assignments.items()
            },
            before,
        )

    def test_notam_workflow_does_not_publish_when_adjustment_fails(self):
        control = RampWorkflowControl()

        def fail():
            raise StandConflict("no safe destination")

        with self.assertRaises(StandConflict):
            control.apply_notam_stand_restriction(
                "op-1",
                "stand-plan",
                datetime(2026, 1, 1, 10, tzinfo=UTC),
                fail,
            )

        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})


if __name__ == "__main__":
    unittest.main()
