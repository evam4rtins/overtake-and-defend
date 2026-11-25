## T5 : priority to lateral acceleration in "project_to_limits"

when max steering and max breaking, priority to steering and not breaking

maybe a 50/50 projection ?

## T10 -> T16 : choice of t_wh and d_offset

right now, default settings :
- ACC (opponent): t_wh = 0.6 (s) and d_offset = 5m
- ACC (ego) : 0.3(s) and 2m

maybe we should do the same for both, in real life it is more around 0.2(s) and 0.5m but with the current mesurement (circles to represent the car), of the distance we should put 0.2(s) and 2m

## choice of k_v and k_d

k_d (1/s) : gain for choosing from where the ACC is starting to work
k_v (1/s) : gain to choose how much the ACC reacts when it has to brake

how do we justify these values ? by assumption or by testing different values ?
right now : 
- k_d = 0.5
- k_v = 2

## T10 / T14 : the ACC cap drops the braking anticipation of the plan (correction 2, not applied yet)

In `PurePursuit.control` (`control.py`), when the ACC cap is lower than the reference speed:



So as soon as the ACC is active, the car only brakes with k_v (v_cap − v), about −10 m/s², even when the plan asks for −15 m/s² to make the corner. With the correction Jean did, a much faster ego now arrives fast behind the opponent just before the corner:
- it enters the corner 3 to 9 m/s too fast ;
- then the priority to lateral acceleration cancels the braking (T5 point) ;
- result: it goes wide, and with Δv0 = +20 m/s, τ0 = 1 s, there are 2 collisions.

**Proposed fix (= min(v_opt, v_ACC) of slides 6a, at the acceleration level).** Compute both commands and keep the one that brakes the most:



Its natural place is `control.py` (Eva, T10), so that the `RacingLineAgent` also uses it.


the correction I did in the T14 :

The follow planner used to cap the ego's speed at the leader's speed right away, even when it was far behind, so a much faster car never used its speed advantage. Now the cap only applies from the point where the ego would catch up with the leader (comparing both positions at the same time), so the ego closes the gap at its own speed and only slows down when it arrives behind the leader.
