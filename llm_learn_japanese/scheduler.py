import datetime
import math
from typing import Tuple


def sm2_schedule(
    interval: int,
    ease_factor: float,
    repetitions: int,
    quality: int,
) -> Tuple[int, float, int, datetime.datetime]:
    """
    SM-2 (SuperMemo 2) scheduling algorithm.

    Maintains three pieces of state per card:
      - interval   – current inter-repetition interval in days
      - ease_factor – E-Factor (minimum 1.3, default 2.5)
      - repetitions – how many consecutive correct reviews (n)

    Quality grades (0-5):
      0 – complete blackout
      1 – very poor, wrong answer remembered after seeing correct one
      2 – wrong answer but correct one seemed easy to recall
      3 – correct answer with serious difficulty
      4 – correct answer after some hesitation
      5 – perfect, instant recall

    Algorithm (per https://super-memory.com/english/ol/sm2.htm):
      1. Update E-Factor first.
      2. If quality < 3 (lapse): reset repetitions to 0, interval to 1.
         Otherwise increment repetitions and compute interval:
           n == 1  → 1 day
           n == 2  → 6 days
           n >= 3  → previous interval × updated E-Factor (ceiling)
      3. Compute next review datetime.

    Returns:
        (new_interval, new_ease_factor, new_repetitions, next_review)
    """
    # Clamp quality to valid range
    quality = max(0, min(5, quality))

    # Step 1: Update ease factor (using updated EF for the interval calculation)
    new_ef = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if new_ef < 1.3:
        new_ef = 1.3

    # Step 2: Determine new interval and repetition count
    if quality < 3:
        # Lapse: reset repetitions and interval
        new_reps = 0
        new_interval = 1
    else:
        new_reps = repetitions + 1
        if new_reps == 1:
            new_interval = 1
        elif new_reps == 2:
            new_interval = 6
        else:
            new_interval = math.ceil(interval * new_ef)

    # Step 3: Next review date
    next_review = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=new_interval)

    return new_interval, new_ef, new_reps, next_review


# Backward-compatible alias (deprecated – use sm2_schedule instead)
def fsrs_schedule(interval: int, ease_factor: float, quality: int) -> Tuple[int, float, datetime.datetime]:
    """
    DEPRECATED: Use sm2_schedule() instead.

    Thin wrapper that calls sm2_schedule with repetitions=0 and drops the
    repetitions return value, preserving the old 3-arg / 3-return signature.
    """
    # Infer repetitions from interval as a rough heuristic for legacy callers:
    #   interval == 1 → likely first review (reps=0)
    #   interval <= 6 → likely second review (reps=1)
    #   otherwise     → at least third review (reps=2)
    if interval <= 1:
        inferred_reps = 0
    elif interval <= 6:
        inferred_reps = 1
    else:
        inferred_reps = 2

    new_interval, new_ef, _new_reps, next_review = sm2_schedule(
        interval, ease_factor, inferred_reps, quality
    )
    return new_interval, new_ef, next_review
