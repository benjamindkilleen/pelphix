from __future__ import annotations
from typing import Any, Optional, Union, NamedTuple
from collections import Counter
import numpy as np
from enum import auto
from strenum import StrEnum
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import logging

log = logging.getLogger(__name__)


def normalize(p: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """Normalize the transition probabilities to sum to 1.

    Args:
        transitions (Union[np.ndarray, dict]): Either a 2D numpy array of transition probabilities,
            or a dictionary of 2D numpy arrays of transition probabilities.

    Returns:
        np.ndarray: Normalized transition probabilities.
    """
    if isinstance(p, np.ndarray):
        q = p / np.sum(p, axis=-1, keepdims=True)
    elif isinstance(p, dict):
        q = {}
        for key, value in p.items():
            q[key] = normalize(value)
    else:
        raise ValueError(f"Invalid type for transitions: {type(p)}")

    return q


def sample_transition(
    transitions: dict[str, float],
    exclude: set[str] = set(),
    weights: dict[str, float] = dict(),
) -> str:
    """Sample a transition from the dictionary.

    Args:
        transitions (dict[str, float]): A mapping of choices to relative probabilities.
        exclude: A set of choices to exclude.
        weights: A mapping of choices to weights.

    Returns:
        str: The chosen
    """
    choices = list(transitions.keys())
    p = np.array(list(transitions.values()), dtype=np.float32)

    log.debug(f"Sampling from {choices} with initial weights {p}")

    for i, c in enumerate(choices):
        log.debug(f"Choice {c}:")
        if c in weights:
            log.debug(f"  Weight: {weights[c]}")
            p[i] *= weights[c]
        if c in exclude:
            log.debug(f"  Excluding {c}")
            p[i] = 0

    log.debug(f"Sampling from {choices} with weights {p}")

    if np.sum(p) == 0:
        log.warning(f"No valid choices left. Returning 'end'.")
        return "end"

    p = p / np.sum(p)
    log.debug(f"Sampling from {choices} with probabilities {p}")

    return np.random.choice(choices, p=p)


# Each level workflow state has a record of which things have been visited before for that
# step/task/activity/acquisition, and how many times.
#
# And we need to know the visited steps (each step can only be visited once, expect for the `final_screws`
# step, which is always the last step in the procedure, finishing any screws that haven't been done yet.
#
# So we need to know the visited tasks (each task can only be visited once). Maintain the set of
# visited tasks and possible tasks.


class Task(StrEnum):
    start = auto()
    s1_left = auto()
    s1_right = auto()
    s1 = auto()
    s2 = auto()
    ramus_left = auto()
    ramus_right = auto()
    teardrop_left = auto()
    teardrop_right = auto()
    screw_s1_left = auto()
    screw_s1_right = auto()
    screw_s1 = auto()
    screw_s2 = auto()
    screw_ramus_left = auto()
    screw_ramus_right = auto()
    screw_teardrop_left = auto()
    screw_teardrop_right = auto()
    end = auto()

    def is_screw(self) -> bool:
        return self.name.startswith("screw_")

    def is_wire(self) -> bool:
        return not self.is_screw() and self != Task.start and self != Task.end

    def get_wire(self) -> Task:
        if self.is_screw():
            return Task(self.name.replace("screw_", ""))
        else:
            return self

    def get_screw(self) -> Task:
        if self.is_wire():
            return Task(f"screw_{self.name}")
        else:
            return self


class Activity(StrEnum):
    start = auto()
    position_wire = auto()
    insert_wire = auto()
    insert_screw = auto()
    end = auto()


class Acquisition(StrEnum):
    start = auto()
    ap = auto()
    lateral = auto()
    inlet = auto()
    outlet = auto()
    oblique_left = auto()
    oblique_right = auto()
    teardrop_left = auto()
    teardrop_right = auto()
    end = auto()


class Frame(StrEnum):
    start = auto()
    fluoro_hunting = auto()
    assessment = auto()
    end = auto()


class FrameState(NamedTuple):
    """Contains a frozen workflow state, for simple comparison and hashing."""

    task: Task
    activity: Activity
    acquisition: Acquisition
    frame: Frame


class SimState:
    """The state of the simulation.

    This defines a finite state machine, but in a bit of a hacky way. The "inputs" are
    class attributes that are set externally based on observations of the environment.

    More robust way to do this would be to define a finite set of input observations.

    Attributes:
        task (Task): The task.
        activity (Activity): The activity.
        acquisition (Acquisition): The acquisition.
        max_corridors (int): The maximum number of corridors for this procedure. When this is reached, the procedures goes to final screw insertion.
        steps_done (set[Step]): All the steps completed.
        wires_done (set[Task]): All the wire tasks completed.
        screws_done (set[Task]): All the screw tasks completed.
        acquisitions_done (Counter[Acquisition]): Counter for the acquisitions completed during this activity.
            This enables us to gradually lower the likelihood of re-acquiring an acquisition already done.
            When the activity is completed, the acquisitions are reset. Gets reset when the activity is reset.
        looks_good (bool): Whether the current activity looks good from the current acquisition. While looks_good is False,
            the activity will be repeated. As soon as looks_good is True, a different acquisition
            goal can be sampled, and the next state will start with looks_good=False.

    The steps/tasks/activities/acquisitions in each X_done should be ones that won't be visited
    again, not just those visited (can visit multiple times).

    """

    transitions: DictConfig = OmegaConf.load(
        Path(__file__).parent.parent.parent.parent / "data" / "transitions.yaml"
    )
    task_transitions = transitions["task"]
    activity_transitions = transitions["activity"]
    acquisition_transitions = transitions["acquisition"]

    def __init__(
        self,
        task: Task = Task.start,
        activity: Activity = Activity.start,
        acquisition: Acquisition = Acquisition.start,
        frame: Frame = Frame.start,
        max_corridors: int = 7,
        wires_done: set[Task] = set(),
        screws_done: set[Task] = set(),
        acquisition_counter: Counter[Acquisition] = Counter(),
        previous: Optional[SimState] = None,
    ):
        """Create a workflow state.

        Args:
            step (str, optional): The step. Defaults to "start".
            task (str, optional): The task. Defaults to "start".
            activity (str, optional): The activity. Defaults to "start".
            acquisition (str, optional): The acquisition. Defaults to "start".
            max_corridors (int, optional): The maximum number of tasks for this procedure.
        """
        self.task = Task(task)
        self.activity = Activity(activity)
        self.acquisition = Acquisition(acquisition)
        self.frame = Frame(frame)
        self.max_corridors = max_corridors
        self.wires_done = set(wires_done)
        self.screws_done = set(screws_done)
        self.acquisition_counter = Counter(acquisition_counter)
        self.previous = previous

        if previous is None:
            self.n = 0
        elif self.ready():
            self.n = previous.n + 1
        else:
            self.n = previous.n

        # view_looks_good may be updated externally. If the frame is an assessment frame, then
        # view_looks_good is always True.
        self.view_looks_good = self.frame in {Frame.assessment, Frame.end}

        if self.frame == Frame.end:
            self.need_new_view = self.previous.need_new_view
        else:
            self.need_new_view = False

        self.fix_wire = False

        # If the frame or acquisition is in end, then the previous state was an acquisition, which
        # means that it should know whether the wire looks good.
        # Otherwise, these values will be set externally by the assessment.
        # Could be moving onto new wire, or going to the screw for the same wire...
        if (
            self.frame == Frame.end
            or self.acquisition in [Acquisition.end]
            or self.activity in [Activity.end]
        ):
            assert self.previous is not None
            self.wire_looks_good = self.previous.wire_looks_good
            self.wire_looks_inserted = self.previous.wire_looks_inserted
            self.screw_looks_inserted = self.previous.screw_looks_inserted

        else:
            # Whether the wire looks like it's on a good trajectory.
            # This fundamentally assumes that before being done with inserting the wire, it has been checked
            # That may not be correct, but we can't go back from inserting a screw.
            self.wire_looks_good = False

            # used primarily to signal when to go back to wire insertion
            # Wire
            self.wire_looks_inserted = (
                self.activity in {Activity.insert_screw, Activity.end} or self.task.is_screw()
            )

            # Whether the screw looks fully inserted
            self.screw_looks_inserted = self.activity in {Activity.end}

    # A given frame can never belong to "start" or "end" states. These exist solely for simulating
    # transition probabilities.

    # Each transition graph is unique depending on the state you're in.
    # - But for simplicity, the task only depends on the step.
    # - The activity only depends on the task,
    # - And the acquisition only dpeends on the task.
    # - The franme transition only depend on the acquisition (assumption that all trajectory views
    #   are equally difficult, but whatever)
    # Note:
    # - transitions[i, j] is the probability of transitioning from state i to state j.
    # - steps and tasks cannot be revisited, once completed. Activities, acquisitions, and frames can
    #   be revisited.

    def sample_task(self) -> Task:
        assert self.task != Task.end

        if (
            len(self.wires_done) >= self.max_corridors
            and len(self.screws_done) >= self.max_corridors
        ):
            return Task.end
        elif len(self.wires_done) >= self.max_corridors:
            # If we've done all the wires, then we need to do the screws. Sample with equal probability.
            screws_possible = set(f"screw_{s}" for s in self.wires_done)
            transitions = dict((s, 1) for s in screws_possible)
            return Task(sample_transition(transitions))
        elif self.task in self.task_transitions:
            transitions = self.task_transitions[self.task]

            # Set of screws for which wires have been done.
            screws_possible = set(f"screw_{s}" for s in self.wires_done)
            # Set of screws for which wires have not been done.
            exclude_screws = set(t for t in Task if t.startswith("screw_")) - screws_possible
            return Task(
                sample_transition(
                    transitions, exclude=self.wires_done | self.screws_done | exclude_screws
                )
            )
        else:
            log.critical(f"Task {self.step} / {self.task} has no transitions defined!")
            return Task.end

    def sample_activity(self) -> Activity:
        assert self.activity != Activity.end

        if not self.task.is_screw() and not self.wire_looks_good:
            # If the wire looks bad, then we need to go back to wire insertion
            log.debug(f"Wire looks bad, going back to wire positioning from {self.activity}.")
            return Activity.position_wire
        elif self.activity == Activity.position_wire and self.wire_looks_good:
            # Actually we should check the other view, unless it's been checked.
            return Activity.insert_wire
        elif self.activity == Activity.insert_wire and self.wire_looks_inserted:
            return Activity.end
        elif self.activity == Activity.insert_screw and self.screw_looks_inserted:
            return Activity.end
        elif (
            self.task in self.activity_transitions
            and self.activity in self.activity_transitions[self.task]
        ):
            transitions = self.activity_transitions[self.task][self.activity]
            return Activity(sample_transition(transitions))
        else:
            log.critical(f"Activity {self.task} / {self.activity} has no transitions defined!")
            return Activity.end

    @property
    def object_looks_good(self) -> bool:
        if self.activity == Activity.position_wire:
            return self.wire_looks_good
        elif self.activity == Activity.insert_wire:
            return self.wire_looks_inserted
        elif self.activity == Activity.insert_screw:
            return self.screw_looks_inserted
        else:
            return False

    def sample_acquisition(self) -> Acquisition:
        assert self.acquisition != Acquisition.end

        if (
            self.activity not in [Activity.start, Activity.position_wire]
            and not self.wire_looks_good
        ):
            # If we're not positioning the wire, then we need to go back to positioning the wire.
            # Therefore end the current acquisition. This causes a new activity to be sampled (which
            # will inherit wire_looks_good).
            log.debug(f"Wire looks bad, ending acquisition from {self.acquisition}")
            self.fix_wire = True
            return Acquisition.end
        elif (
            self.activity == Activity.insert_wire
            and self.get_previous_activity() == Activity.position_wire
            and self.acquisition == Acquisition.start
        ):
            # If we are inserting the wire, and we just started inserting it,
            # then we want to actually insert it, so don't change the view.
            out = self.get_previous_acquisition()
            log.debug(f"Inserting wire, don't change view from {out}")
            return out
        elif (
            self.activity == Activity.insert_wire
            and self.wire_looks_good
            and self.wire_looks_inserted
        ):
            # If the wire is being inserted, and it looks good, then we would want to change views to check it.
            log.debug(f"Wire looks good, ending acquisition from {self.acquisition}")
            self.need_new_view = True
            return Acquisition.end
        elif self.activity == Activity.insert_screw and self.screw_looks_inserted:
            # If the screw is being inserted, and it looks god, then we're done.
            log.debug(f"Screw looks good, ending acquisition from {self.acquisition}")
            return Acquisition.end
        elif (
            not self.need_new_view
            and not self.object_looks_good
            and self.acquisition != Acquisition.start
        ):
            # If the previous acquisition looks good, and the current object being manipulated
            # doesn't look good, then we'll just repeat it.
            log.debug(
                f"Object looks bad, repeating acquisition {self.acquisition} while in {self.activity}"
            )
            return self.acquisition
        elif (
            self.task in self.acquisition_transitions
            and self.activity in self.acquisition_transitions[self.task]
            and self.acquisition in self.acquisition_transitions[self.task][self.activity]
        ):
            # Exclude the current aqcquisition and weight the remaining ones by the number of
            # times they've been done.
            transitions = self.acquisition_transitions[self.task][self.activity][self.acquisition]
            exclude = {self.acquisition}
            counter = self.acquisition_counter.copy()
            counter[self.acquisition] = 0  # Don't sample the same acquisition if it looks good.
            if self.need_new_view:
                exclude.add(Acquisition.end)
            total = sum(counter.values())
            if total == 0:
                weights = dict()
            else:
                weights = dict((k, 1 / (v + 1)) for k, v in counter.items())
            log.debug(
                f"Sampling acquisition from {transitions} with weights {weights}, excluding {exclude}"
            )
            return Acquisition(sample_transition(transitions, exclude=exclude, weights=weights))
        else:
            log.critical(
                f"Acquisition {self.task} / {self.activity} / {self.acquisition} has no transitions defined!"
            )
            log.debug(f"acquisition_transitions: {self.acquisition_transitions[self.task]}")
            exit()
            return Acquisition.end

    def sample_frame(self) -> Frame:
        if not self.view_looks_good:
            return Frame.fluoro_hunting
        elif self.frame in [Frame.fluoro_hunting, Frame.start]:
            return Frame.assessment
        elif self.frame == Frame.assessment:
            return Frame.end
        else:
            log.critical(f"Frame {self.frame} has no transitions defined!")
            return Frame.end

    def next(self) -> SimState:
        """Sample a new workflow state, given the current one.

        Goes from left to right and transitions everything that needs to be transitioned. The frame
        type is sampled separately, by evaluating whether the view sampled in `sample_next_view` is
        accurate.

        Workflow states are returned in "start" to let the caller know that the state has restarted
        and reset any simulation state for that type, but the "end" state will never be returned
        unless the whole procedure is over (ending a task, activity, or acquisition will roll over
        to the start of the next)

        Returns:
            The new workflow state. This will never be in the "start" state for any level, but it could be in the "end"
                state for one or more levels. This is so the caller has a chance to reset any simulation state.

        """
        # log.debug(
        #     f"Sampling next state from {self} with\n"
        #     f"\tview_looks_good={self.view_looks_good},\n"
        #     f"\twire_looks_good={self.wire_looks_good},\n"
        #     f"\twire_looks_inserted={self.wire_looks_inserted}\n"
        #     f"\tscrew_looks_inserted={self.screw_looks_inserted}\n"
        #     f"\tfix_wire={self.fix_wire}\n"
        # )
        log.debug(f"sampling from: {self}\n\t{self.acquisition_counter}")

        if self.fix_wire and self.task not in [Task.start, Task.end]:
            # Special case where we need to go back to positioning for the current wire.
            log.debug(f"Fixing wire, going back to positioning wire from {self}")
            state = SimState(
                task=self.task.get_wire(),
                activity=Activity.position_wire,
                acquisition=Acquisition.start,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done - {self.task.get_wire()},
                screws_done=self.screws_done - {self.task.get_screw()},
                acquisition_counter=Counter(),
                previous=self,
            )

        # Sample the task.
        elif self.task == Task.end:
            # If the task is at end, then the procedure is over. No recursive call.
            state = SimState(
                task=Task.end,
                activity=Activity.end,
                acquisition=Acquisition.end,
                frame=Frame.end,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )
        elif self.task == Task.start:
            # If the task is at start, sample a new task.
            state = SimState(
                task=self.sample_task(),
                activity=Activity.start,
                acquisition=Acquisition.start,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )

        elif self.activity == Activity.end:
            # If the activity is at end, then the task is over.
            # The acquistion counter should be reset.
            state = SimState(
                task=self.sample_task(),
                activity=Activity.start,
                acquisition=Activity.start,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=Counter(),
                previous=self,
            )
        elif self.activity == Activity.start:
            # If the activity is at start, sample a new activity.
            state = SimState(
                task=self.task,
                activity=self.sample_activity(),
                acquisition=Acquisition.start,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )

        elif self.acquisition == Acquisition.end:
            # Actually, this is when you update the screws/wires done, so that when we go to sample a new activity,
            # we can check if we're done. Only add them if the sampled activity is "end".

            activity = self.sample_activity()
            if activity == Activity.end:
                if self.task.is_wire():
                    log.debug(f"Adding {self.task} to wires_done")
                    wires_done = self.wires_done | {self.task}
                    screws_done = self.screws_done
                else:
                    log.debug(f"Adding {self.task} to screws_done")
                    wires_done = self.wires_done
                    screws_done = self.screws_done | {self.task}
            else:
                wires_done = self.wires_done
                screws_done = self.screws_done

            # If acquisition is at "end", then the activity is over, and no more acquisitions need to be done. Increment the counter for
            # this acquisition, set Acquisition to start and sample a new activity.
            state = SimState(
                task=self.task,
                activity=activity,
                acquisition=Acquisition.start,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=wires_done,
                screws_done=screws_done,
                # acquisition_counter=Counter(),  # TODO: resulting in never moving on to insert_wire?
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )
        elif self.acquisition == Acquisition.start:
            # If the acquisition is not at end, then sample a new acquisition.
            acquisition = self.sample_acquisition()
            log.debug(f"Sampled new acquisition: {acquisition} from {self.acquisition}")
            state = SimState(
                task=self.task,
                activity=self.activity,
                acquisition=acquisition,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )

        elif self.frame == Frame.end:
            # If frame is at end, then current acquisition is over
            acquisition = self.sample_acquisition()
            # log.debug(f"Sampling new acquisition: {acquisition} from {self.acquisition}")
            state = SimState(
                task=self.task,
                activity=self.activity,
                acquisition=acquisition,
                frame=Frame.start,
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter + Counter({self.acquisition: 1}),
                previous=self,
            )
        else:
            # Otherwise, sample a new frame.
            state = SimState(
                task=self.task,
                activity=self.activity,
                acquisition=self.acquisition,
                frame=self.sample_frame(),
                max_corridors=self.max_corridors,
                wires_done=self.wires_done,
                screws_done=self.screws_done,
                acquisition_counter=self.acquisition_counter,
                previous=self,
            )

        if state.is_end() and not state.is_finished():
            return state.next()
        else:
            return state

    def __iter__(self):
        """Iterate over the workflow state. This makes it possible to call dict() on a workflow state."""
        yield ("task", self.task)
        yield ("activity", self.activity)
        yield ("acquisition", self.acquisition)
        yield ("frame", self.frame)

    def values(self) -> tuple[Task, Activity, Acquisition, Frame]:
        """Return the values of the state."""
        return self.task, self.activity, self.acquisition, self.frame

    def framestate(self) -> FrameState:
        """Get the state as a frozen namedtuple.

        This should only be used when the state is ready to be sampled, i.e. when the state is at
        the start or end phases at any level."""
        if not self.ready():
            raise ValueError("Cannot get framestate if not ready.")
        return FrameState(**dict(self))

    def is_finished(self) -> bool:
        """Whether the procedures is finished."""
        return self.task == "end"

    def is_end(self) -> bool:
        """Whether the state is at the end of any level."""
        return "end" in self.values()

    def finish_activity(self) -> SimState:
        """Return the state with the acquisition set to end.

        This marks the end of the activity, and the next state sampled will be a new activity, in the Acquisition.start.
        """
        return SimState(
            task=self.task,
            activity=self.activity,
            acquisition=Acquisition.end,
            max_corridors=self.max_corridors,
            wires_done=self.wires_done,
            screws_done=self.screws_done,
            acquisition_counter=self.acquisition_counter,
            previous=self,
        )

    def get_previous_activity(self) -> Activity:
        """Get the previous activity."""
        if self.previous is None:
            return Activity.start
        elif self.previous.activity in ["start", "end"]:
            return self.previous.get_previous_activity()
        else:
            return self.previous.activity

    def get_previous_acquisition(self) -> Acquisition:
        """Get the previous acquisition."""
        if self.previous is None:
            return Acquisition.start
        elif self.previous.acquisition in ["start", "end"]:
            return self.previous.get_previous_acquisition()
        else:
            return self.previous.acquisition

    def get_previous_frame(self) -> Frame:
        """Get the previous frame."""
        if self.previous is None:
            return Frame.start
        elif self.previous.frame in ["start", "end"]:
            return self.previous.get_previous_frame()
        else:
            return self.previous.frame

    def ready(self) -> bool:
        """Whether the state is ready for an image."""
        return (
            self.task not in ["start", "end"]
            and self.activity not in ["start", "end"]
            and self.acquisition not in ["start", "end"]
        )

    def __str__(self) -> str:
        return f"WorkflowState(task={self.task.name}, activity={self.activity.name}, acquisition={self.acquisition.name}, frame={self.frame.name})"
