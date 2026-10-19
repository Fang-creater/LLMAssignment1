import math

def learning_rate_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif it <= cosine_cycle_iters:
        unit = math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)
        return min_learning_rate + 0.5 * (1 + unit) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate