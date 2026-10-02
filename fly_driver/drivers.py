"""Drivers: the stages composed into one thing that speaks the shared contract.

A :class:`~fly_driver.interface.Driver` is what an environment or the evaluation harness
talks to -- a camera frame in, a :class:`~fly_driver.interface.ControlVector` out. Two
compositions live here, and they are the two rungs `AGENTS.md` §6 names:

- :class:`DirectDriveAgent`: eye → (brain) → policy → car. The body is skipped; the policy's
  control goes straight to the cockpit. This is the floor, and the primary RQ1 comparison.
  The brain is optional. With one, it is the vision+brain grade path (GH-23): the policy
  reads the brain, not the eye.
- :class:`EmbodiedDriveAgent`: eye → (brain) → policy → body → car. The policy's control
  is the *intent*; the :class:`~fly_driver.interface.Body` turns it into whatever the
  cockpit reads once the fly has moved. The tethered fly body (GH-21) plugs in here.

:class:`PassthroughBody` relays the intent unchanged, so an embodied driver with it is
direct drive spelled the long way -- asserted in the tests, because "both implement the
same contract" is the ticket's acceptance criterion, not a hope.

Every seam is checked, never patched (`AGENTS.md` §11): a frame of the wrong shape, an eye
or brain emitting the wrong number of features, a policy returning an out-of-range action,
or a body returning something that is not a control vector all raise at the seam they broke.
"""

from __future__ import annotations

from fly_driver.interface import (
    Body,
    Brain,
    ControlVector,
    Eye,
    Features,
    Frame,
    Policy,
    validate_features,
    validate_frame,
)

__all__ = ["DirectDriveAgent", "EmbodiedDriveAgent", "PassthroughBody"]


class DirectDriveAgent:
    """Eye → optional brain → policy → car, with the body skipped.

    Args:
        eye: Turns frames into features. Its ``frame_shape`` becomes the driver's.
        policy: Turns features into a control. With no brain it must expect exactly the
            eye's ``feature_dim``. With a brain it must expect the brain's ``output_dim``.
            A mismatch is a wiring error and fails here, not on the first frame.
        brain: Central-complex module between the eye and the policy. ``None`` leaves the
            policy reading the eye, which is the RQ1 direct-drive floor. Keyword-only so
            it cannot be confused with a body.

    Raises:
        ValueError: If the eye, brain, and policy disagree on feature width.
    """

    def __init__(self, eye: Eye, policy: Policy, *, brain: Brain | None = None) -> None:
        if brain is None:
            if int(eye.feature_dim) != int(policy.feature_dim):
                raise ValueError(
                    f"eye emits {eye.feature_dim} features but the policy expects "
                    f"{policy.feature_dim}; the stages are not wired to each other"
                )
        else:
            if int(eye.feature_dim) != int(brain.input_dim):
                raise ValueError(
                    f"eye emits {eye.feature_dim} features but the brain expects "
                    f"{brain.input_dim}; the stages are not wired to each other"
                )
            if int(brain.output_dim) != int(policy.feature_dim):
                raise ValueError(
                    f"brain emits {brain.output_dim} features but the policy expects "
                    f"{policy.feature_dim}; the stages are not wired to each other"
                )
        self.eye = eye
        self.policy = policy
        self.brain = brain

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """The ``(height, width, 3)`` frame this driver accepts: the eye's."""
        return tuple(int(size) for size in self.eye.frame_shape)  # type: ignore[return-value]

    @property
    def feature_dim(self) -> int:
        """Width of the vector the policy reads.

        The eye's feature count when no brain is in the loop, and the brain's output
        count when one is.
        """
        if self.brain is not None:
            return int(self.brain.output_dim)
        return int(self.eye.feature_dim)

    def reset(self, seed: int | None = None) -> None:
        """Start an episode: reset every stage, in pipeline order.

        Args:
            seed: Forwarded to the policy. The eye's reset takes none -- an optic lobe
                warms up from grey the same way every time. The brain's reset clears its
                membrane; it takes none for the same reason.
        """
        self.eye.reset()
        if self.brain is not None:
            self.brain.reset()
        self.policy.reset(seed=seed)

    def act(self, frame: Frame) -> ControlVector:
        """One frame in, one control out.

        Raises:
            TypeError: If the frame is not uint8, or the policy returns a non-number.
            ValueError: If the frame has the wrong shape (never resized), the eye or the
                brain emits the wrong feature vector, or the policy's control is out of
                range (never clipped).
        """
        return self._intend(frame)

    def _intend(self, frame: Frame) -> ControlVector:
        frame = validate_frame(frame, self.frame_shape)
        features: Features = validate_features(self.eye.encode(frame), int(self.eye.feature_dim))
        if self.brain is not None:
            features = validate_features(self.brain.step(features), int(self.brain.output_dim))
        return ControlVector.from_any(self.policy.act(features))


class EmbodiedDriveAgent(DirectDriveAgent):
    """Eye → policy → body → car: the same contract, with the fly in the loop.

    Args:
        eye: As for :class:`DirectDriveAgent`.
        policy: As for :class:`DirectDriveAgent`; its output is the *intended* control.
        body: Turns the intent into the control the cockpit actually reads.
    """

    def __init__(self, eye: Eye, policy: Policy, body: Body) -> None:
        super().__init__(eye, policy)
        self.body = body

    def reset(self, seed: int | None = None) -> None:
        """Reset eye, policy, then body."""
        super().reset(seed=seed)
        self.body.reset()

    def act(self, frame: Frame) -> ControlVector:
        """One frame in, one control out, via the body.

        Raises:
            TypeError: If the body returns anything but a
                :class:`~fly_driver.interface.ControlVector`. A body that hands back a
                raw array has skipped the range check, which is the silent failure the
                contract exists to stop.
        """
        realised = self.body.actuate(self._intend(frame))
        if not isinstance(realised, ControlVector):
            raise TypeError(
                f"body must return a ControlVector, got {type(realised).__name__}; build "
                "one with ControlVector(...) or ControlVector.clipped(...) so the range "
                "is checked"
            )
        return realised


class PassthroughBody:
    """A body with no dynamics: the cockpit reads exactly what the brain intended.

    Direct drive expressed as an embodied driver. Useful as the control condition against
    a real body, and as the stand-in until GH-21 lands.
    """

    def reset(self) -> None:
        """Nothing to reset."""

    def actuate(self, intent: ControlVector) -> ControlVector:
        """Return the intent unchanged."""
        return intent
