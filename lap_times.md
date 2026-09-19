# Lap times

The record book for the MuJoCo practice track (Silverstone). Every completed lap is
appended here by `scripts/drive.py`.

**The best lap is simply the fastest row in the table below.** Nothing is cached: the file
is re-read whenever it changes, so if you delete the rows you drove yourself, the best time
updates straight away -- even while the simulator is running. That is the whole point of
keeping this in a document rather than in the code.

A lap only counts if it was driven the whole way round; cutting back across the start line
does not register one.

`Lap` is the time that counts. It is `Raw` plus `Penalty`, where the penalty is the
seconds earned by going off the circuit -- see `fly_driver/envs/penalty.py` for what an
excursion costs. A lap driven entirely on the track has a penalty of zero and a `Lap` equal
to `Raw`. Rows set before penalties existed have a zero penalty for that reason.

Times are `M:SS.mmm`. Rows may be deleted or reordered freely, but keep the table header.

The car itself changes as it is calibrated, so a time is only comparable against others set
on the same car. Rows above a `car changed` marker were driven on an earlier one.

<!-- segments -->
<!-- /segments -->

| Date | Lap | Penalty | Raw | Driver | Note |
| --- | --- | --- | --- | --- | --- |
| 2026-09-17 13:14:07 | 2:21.178 | 0.000 | 2:21.178 | gamepad (PS5 Controller) | new best |
| 2026-09-17 13:41:44 | 2:16.673 | 0.000 | 2:16.673 | gamepad (PS5 Controller) | new best |
| 2026-09-17 | -- |  |  | car changed | aero and tyres calibrated from Assetto Corsa data |
| 2026-09-17 17:29:13 | 1:52.229 | 0.000 | 1:52.229 | gamepad (PS5 Controller) | new best |
