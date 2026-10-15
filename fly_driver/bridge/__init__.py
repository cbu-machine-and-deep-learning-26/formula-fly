"""The Assetto Corsa bridge (GH-24): frames from the car, controls from the fly, over TCP.

Assetto Corsa runs only on the Windows x86 box and the fly stack may run on a Mac or a DGX
Spark, so the two halves talk over the LAN (or localhost when they share a machine):

- :mod:`fly_driver.bridge.wire`: the stdlib wire format both ends speak.
- :mod:`fly_driver.bridge.fly_side`: serves any :class:`~fly_driver.interface.Driver`.
- :mod:`fly_driver.bridge.car_side`: the real-time loop next to AC, with the deadline rule.
- :mod:`fly_driver.bridge.assetto`: AssettoCorsaGym behind the car side's env protocol.

None of these import AssettoCorsaGym at module scope, so the package and its tests run
where AC is not installed. See ``docs/assetto-corsa-bridge.md`` for the procedure.
"""
