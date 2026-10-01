import unittest
from datetime import datetime, time, timedelta, timezone

from rampflow.models import Decision
from rampflow.notam import Notam, NotamRegistry, Recurrence
from rampflow.stand_orchestration import (
    StandAssignment,
    StandConflict,
    StandDemand,
    StandOrchestrator,
    StandProfile,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
D = lambda day, h, m=0: datetime(2026, 1, day, h, m, tzinfo=UTC)


def cross_midnight_notam() -> Notam:
    return Notam(
        "N1",
        "apron",
        "A1",
        Window(D(1, 0), D(4, 0)),
        5,
        "closed",
        recurrence=Recurrence(
            frozenset(range(7)), time(22, 0), time(2, 0), "UTC"
        ),
    )


class CrossMidnightNotamTests(unittest.TestCase):
    def test_cross_midnight_recurrence_runs_past_midnight_local(self):
        registry = NotamRegistry([cross_midnight_notam()])
        horizon = Window(D(1, 0), D(4, 0))
        effects = registry.effects("apron", "A1", horizon)
        spans = [(e.window.start, e.window.end) for e in effects]
        self.assertEqual(
            spans,
            [
                (D(1, 0), D(1, 2)),    # tail of the Dec-31 occurrence
                (D(1, 22), D(2, 2)),
                (D(2, 22), D(3, 2)),
                (D(3, 22), D(4, 0)),   # clipped by the effective end
            ],
        )
        # the closure must still bite in the early hours of the next day
        early = registry.decision("apron", "A1", Window(D(2, 0, 30), D(2, 1, 30)))
        self.assertFalse(early.accepted)
        self.assertEqual(early.code, "notam-closure")
        # and lift once the local window ends
        after = registry.decision("apron", "A1", Window(D(2, 3), D(2, 4)))
        self.assertTrue(after.accepted)

    def test_same_day_recurrence_unchanged(self):
        n = Notam(
            "N2",
            "apron",
            "A1",
            Window(D(1, 0), D(3, 0)),
            5,
            "closed",
            recurrence=Recurrence(frozenset(range(7)), time(6, 0), time(8, 0), "UTC"),
        )
        effects = NotamRegistry([n]).effects("apron", "A1", Window(D(1, 0), D(3, 0)))
        self.assertEqual([(e.window.start, e.window.end) for e in effects], [(D(1, 6), D(1, 8)), (D(2, 6), D(2, 8))])


def legacy_orchestrator() -> StandOrchestrator:
    # B forbids A; A does not list B back (asymmetric survey data).
    return StandOrchestrator(
        [
            StandProfile("A", "T1", 60.0, True, 1),
            StandProfile("B", "T1", 60.0, True, 1, incompatible_simultaneous=frozenset({"A"})),
            StandProfile("C", "T1", 60.0, True, 1),
        ]
    )


def inject(orchestrator: StandOrchestrator, flight_id: str, stand_id: str, window: Window) -> None:
    # recreate a pre-fix allocation that the corrected checks would now reject
    assignment = StandAssignment(flight_id, stand_id, window, window, 0, (0, 0, stand_id))
    orchestrator.assignments[flight_id] = assignment
    orchestrator.by_stand[stand_id].append(assignment)


class ReverseAdjacencyTests(unittest.TestCase):
    def test_reverse_incompatibility_blocks_candidate(self):
        o = legacy_orchestrator()
        w = Window(D(1, 10), D(1, 12))
        demand = StandDemand("F1", w, 36.0, False)
        o.commit([c for c in o.candidates(demand) if c.stand_id == "B"][0])
        candidates = o.candidates(StandDemand("F2", w, 36.0, False))
        self.assertEqual([c.stand_id for c in candidates], ["C"])

    def test_reverse_incompatibility_blocks_commit_and_reassign(self):
        o = legacy_orchestrator()
        w = Window(D(1, 10), D(1, 12))
        o.commit([c for c in o.candidates(StandDemand("F1", w, 36.0, False)) if c.stand_id == "B"][0])
        inject(o, "F2", "C", w)
        with self.assertRaises(StandConflict):
            o.reassign("F2", "A", D(1, 10, 30), 15, "test")


class RemediationTests(unittest.TestCase):
    def test_closure_migrates_victim_and_keeps_unaffected(self):
        o = legacy_orchestrator()
        w = Window(D(1, 10), D(1, 12))
        inject(o, "F1", "A", w)
        inject(o, "F2", "B", w)
        demands = {fid: StandDemand(fid, w, 36.0, False) for fid in ("F1", "F2")}
        result = o.remediate(demands, closures={"A": (w,)})
        self.assertEqual(result.kept, ("F2",))
        self.assertEqual([(r.flight_id, r.from_stand, r.to_stand) for r in result.relocated], [("F1", "A", "C")])
        self.assertEqual(o.assignments["F1"].stand_id, "C")
        self.assertEqual(o.assignments["F2"].stand_id, "B")

    def test_incompatible_pair_is_separated_keeping_earlier_flight(self):
        o = legacy_orchestrator()
        early = Window(D(1, 10), D(1, 12))
        late = Window(D(1, 11), D(1, 13))
        inject(o, "F1", "B", early)
        inject(o, "F2", "A", late)
        demands = {
            "F1": StandDemand("F1", early, 36.0, False),
            "F2": StandDemand("F2", late, 36.0, False),
        }
        result = o.remediate(demands)
        self.assertEqual(result.kept, ("F1",))
        self.assertEqual([(r.flight_id, r.to_stand) for r in result.relocated], [("F2", "C")])
        self.assertEqual(o.assignments["F2"].stand_id, "C")

    def test_failed_adjustment_moves_nobody(self):
        # only the incompatible pair exists; nowhere to migrate to
        o = StandOrchestrator(
            [
                StandProfile("A", "T1", 60.0, True, 1),
                StandProfile("B", "T1", 60.0, True, 1, incompatible_simultaneous=frozenset({"A"})),
            ]
        )
        w = Window(D(1, 10), D(1, 12))
        inject(o, "F1", "A", w)
        inject(o, "F2", "B", w)
        generation = o.generation
        demands = {fid: StandDemand(fid, w, 36.0, False) for fid in ("F1", "F2")}
        with self.assertRaises(StandConflict):
            o.remediate(demands, closures={"A": (w,)})
        # not a partial fix: both aircraft are still where they were
        self.assertEqual(o.assignments["F1"].stand_id, "A")
        self.assertEqual(o.assignments["F2"].stand_id, "B")
        self.assertEqual(o.generation, generation)

    def test_missing_demand_rolls_back(self):
        o = legacy_orchestrator()
        w = Window(D(1, 10), D(1, 12))
        inject(o, "F1", "A", w)
        with self.assertRaises(StandConflict):
            o.remediate({}, closures={"A": (w,)})
        self.assertEqual(o.assignments["F1"].stand_id, "A")

    def test_clean_plan_is_untouched(self):
        o = legacy_orchestrator()
        w = Window(D(1, 10), D(1, 12))
        inject(o, "F1", "A", w)
        inject(o, "F2", "C", w)
        result = o.remediate({})
        self.assertEqual(result.kept, ("F1", "F2"))
        self.assertEqual(result.relocated, ())


class NotamRestrictionWorkflowTests(unittest.TestCase):
    def test_requirements_are_enforced(self):
        control = RampWorkflowControl()
        denied = Decision(False, "notam-closure", "area closed")
        with self.assertRaises(WorkflowRejected):
            control.apply_notam_stand_restriction(
                "op-1", "A1", D(1, 10), lambda: "ok", requirements=(denied,)
            )
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_failed_domain_change_publishes_nothing(self):
        control = RampWorkflowControl()

        def boom():
            raise RuntimeError("domain change failed")

        with self.assertRaises(RuntimeError):
            control.apply_notam_stand_restriction("op-2", "A1", D(1, 10), boom)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_success_publishes_once_and_replays(self):
        control = RampWorkflowControl()
        calls = []
        first = control.apply_notam_stand_restriction(
            "op-3", "A1", D(1, 10), lambda: calls.append("commit") or "ok"
        )
        second = control.apply_notam_stand_restriction(
            "op-3", "A1", D(1, 10), lambda: calls.append("duplicate")
        )
        self.assertIs(first, second)
        self.assertEqual(calls, ["commit"])
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(first.event.topic, "notam.stand.restriction.applied")


if __name__ == "__main__":
    unittest.main()
