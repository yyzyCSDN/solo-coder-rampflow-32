from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from .timeutil import Window, utc


@dataclass(frozen=True)
class StandProfile:
    stand_id: str
    terminal: str
    max_wingspan_m: float
    international_capable: bool
    jetbridge_count: int
    tow_required_for_push: bool = False
    adjacent: frozenset[str] = frozenset()
    incompatible_simultaneous: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StandDemand:
    flight_id: str
    window: Window
    wingspan_m: float
    international: bool
    terminal_preferences: tuple[str, ...] = ()
    needs_jetbridge: bool = True
    buffer_before_min: int = 5
    buffer_after_min: int = 10
    priority: int = 0


@dataclass(frozen=True)
class StandAssignment:
    flight_id: str
    stand_id: str
    window: Window
    protected_window: Window
    generation: int
    score: tuple[int, int, str]


@dataclass(frozen=True)
class StandMove:
    flight_id: str
    from_stand: str
    to_stand: str
    tow_window: Window
    reason: str


@dataclass(frozen=True)
class StandReassignment:
    flight_id: str
    new_stand: str
    tow_start: datetime
    tow_duration_min: int
    reason: str


class StandConflict(RuntimeError):
    pass


class StandOrchestrator:
    def __init__(self, profiles: Iterable[StandProfile]):
        self.profiles = {p.stand_id: p for p in profiles}
        if not self.profiles:
            raise ValueError("at least one stand required")
        self.assignments: dict[str, StandAssignment] = {}
        self.by_stand: dict[str, list[StandAssignment]] = {
            sid: [] for sid in self.profiles
        }
        self.generation = 0

    def _protected(self, demand: StandDemand) -> Window:
        return Window(
            demand.window.start - timedelta(minutes=demand.buffer_before_min),
            demand.window.end + timedelta(minutes=demand.buffer_after_min),
        )

    def _profile_ok(self, p: StandProfile, d: StandDemand) -> bool:
        if d.wingspan_m > p.max_wingspan_m:
            return False
        if d.international and not p.international_capable:
            return False
        if d.needs_jetbridge and p.jetbridge_count < 1:
            return False
        return True

    def _adjacency_ok(self, stand_id: str, protected: Window, ignore: str | None = None) -> bool:
        profile = self.profiles[stand_id]
        forbidden = set(profile.incompatible_simultaneous)
        for other_id, other in self.profiles.items():
            if stand_id in other.incompatible_simultaneous:
                forbidden.add(other_id)
        for other_id in forbidden:
            for assignment in self.by_stand.get(other_id, ()):
                if assignment.flight_id == ignore:
                    continue
                if assignment.protected_window.overlaps(protected):
                    return False
        return True

    def _stand_free(self, stand_id: str, protected: Window, ignore: str | None = None) -> bool:
        for assignment in self.by_stand[stand_id]:
            if assignment.flight_id == ignore:
                continue
            if assignment.protected_window.overlaps(protected):
                return False
        return self._adjacency_ok(stand_id, protected, ignore)

    def candidates(self, demand: StandDemand) -> tuple[StandAssignment, ...]:
        protected = self._protected(demand)
        out = []
        preference_rank = {
            terminal: rank for rank, terminal in enumerate(demand.terminal_preferences)
        }
        for stand_id, p in self.profiles.items():
            if not self._profile_ok(p, demand):
                continue
            if not self._stand_free(stand_id, protected):
                continue
            rank = preference_rank.get(p.terminal, len(preference_rank) + 1)
            bridge_penalty = max(0, p.jetbridge_count - 1)
            score = (rank, bridge_penalty, stand_id)
            out.append(
                StandAssignment(
                    demand.flight_id,
                    stand_id,
                    demand.window,
                    protected,
                    self.generation,
                    score,
                )
            )
        return tuple(sorted(out, key=lambda a: a.score))

    def plan(self, demand: StandDemand) -> StandAssignment:
        options = self.candidates(demand)
        if not options:
            raise StandConflict(f"no feasible stand for {demand.flight_id}")
        return options[0]

    def commit(self, assignment: StandAssignment) -> None:
        if assignment.flight_id in self.assignments:
            existing = self.assignments[assignment.flight_id]
            if existing == assignment:
                return
            raise StandConflict("flight already has a stand")
        if assignment.stand_id not in self.profiles:
            raise StandConflict("unknown stand")
        if not self._stand_free(assignment.stand_id, assignment.protected_window):
            raise StandConflict("stand became unavailable")
        self.assignments[assignment.flight_id] = assignment
        self.by_stand[assignment.stand_id].append(assignment)
        self.by_stand[assignment.stand_id].sort(
            key=lambda a: (a.protected_window.start, a.flight_id)
        )
        self.generation += 1

    def release(self, flight_id: str) -> None:
        old = self.assignments.pop(flight_id, None)
        if old is None:
            return
        self.by_stand[old.stand_id] = [
            a for a in self.by_stand[old.stand_id] if a.flight_id != flight_id
        ]
        self.generation += 1

    def _reassignment_state(
        self,
        request: StandReassignment,
        current: StandAssignment,
    ) -> tuple[StandAssignment, StandMove]:
        if request.new_stand == current.stand_id:
            raise ValueError("new stand equals current stand")
        if request.new_stand not in self.profiles:
            raise KeyError(request.new_stand)

        tow_start = utc(request.tow_start)
        tow_window = Window(
            tow_start,
            tow_start + timedelta(minutes=request.tow_duration_min),
        )
        if not current.window.contains(tow_start):
            raise StandConflict("tow must start while aircraft is on stand")

        remaining = Window(tow_window.end, current.window.end)
        protected = Window(remaining.start, current.protected_window.end)
        updated = replace(
            current,
            stand_id=request.new_stand,
            window=remaining,
            protected_window=protected,
            generation=self.generation,
        )
        move = StandMove(
            request.flight_id,
            current.stand_id,
            request.new_stand,
            tow_window,
            request.reason,
        )
        return updated, move

    def _shadow_stand_free(
        self,
        by_stand: Mapping[str, list[StandAssignment]],
        stand_id: str,
        protected: Window,
        ignored_flights: frozenset[str],
    ) -> bool:
        profile = self.profiles[stand_id]
        forbidden = set(profile.incompatible_simultaneous)
        for other_id, other in self.profiles.items():
            if stand_id in other.incompatible_simultaneous:
                forbidden.add(other_id)

        for candidate_id in (stand_id, *sorted(forbidden)):
            for assignment in by_stand.get(candidate_id, ()):
                if assignment.flight_id in ignored_flights:
                    continue
                if assignment.protected_window.overlaps(protected):
                    return False
        return True

    def reassign(
        self,
        flight_id: str,
        new_stand: str,
        tow_start: datetime,
        tow_duration_min: int,
        reason: str,
    ) -> StandMove:
        return self.reassign_many(
            (
                StandReassignment(
                    flight_id,
                    new_stand,
                    tow_start,
                    tow_duration_min,
                    reason,
                ),
            )
        )[0]

    def reassign_many(
        self,
        requests: Iterable[StandReassignment],
    ) -> tuple[StandMove, ...]:
        requests = tuple(requests)
        if len({r.flight_id for r in requests}) != len(requests):
            raise ValueError("duplicate flight reassignment")

        staged: dict[str, StandAssignment] = {}
        moves: list[StandMove] = []
        for request in requests:
            current = self.assignments.get(request.flight_id)
            if current is None:
                raise KeyError(request.flight_id)
            updated, move = self._reassignment_state(request, current)
            staged[request.flight_id] = updated
            moves.append(move)

        final = dict(self.assignments)
        final.update(staged)
        final_by_stand: dict[str, list[StandAssignment]] = {
            stand_id: [] for stand_id in self.profiles
        }
        for assignment in final.values():
            final_by_stand[assignment.stand_id].append(assignment)

        # Validate the complete destination layout first so a failed swap cannot
        # leave one aircraft already moved while the other has nowhere to go.
        staged_flights = frozenset(staged)
        staged_rows = tuple(staged.values())
        for left_index, left in enumerate(staged_rows):
            left_profile = self.profiles[left.stand_id]
            for right in staged_rows[left_index + 1 :]:
                right_profile = self.profiles[right.stand_id]
                same_stand = left.stand_id == right.stand_id
                reverse_incompatible = (
                    right.stand_id in left_profile.incompatible_simultaneous
                    or left.stand_id in right_profile.incompatible_simultaneous
                )
                if (
                    (same_stand or reverse_incompatible)
                    and left.protected_window.overlaps(right.protected_window)
                ):
                    raise StandConflict("destination stand unavailable after tow")
            if not self._shadow_stand_free(
                final_by_stand,
                left.stand_id,
                left.protected_window,
                staged_flights,
            ):
                raise StandConflict("destination stand unavailable after tow")

        for rows in final_by_stand.values():
            rows.sort(key=lambda a: (a.protected_window.start, a.flight_id))
        self.assignments = final
        self.by_stand = final_by_stand
        self.generation += 1
        return tuple(moves)

    def reassign_batch(
        self,
        requests: Iterable[StandReassignment],
    ) -> tuple[StandMove, ...]:
        return self.reassign_many(requests)

    def occupancy(self, stand_id: str) -> tuple[StandAssignment, ...]:
        return tuple(self.by_stand[stand_id])

    def utilization(self, stand_id: str, window: Window) -> float:
        occupied = 0.0
        for assignment in self.by_stand[stand_id]:
            start = max(window.start, assignment.window.start)
            end = min(window.end, assignment.window.end)
            if start < end:
                occupied += (end - start).total_seconds()
        total = (window.end - window.start).total_seconds()
        return min(1.0, occupied / total) if total else 0.0
