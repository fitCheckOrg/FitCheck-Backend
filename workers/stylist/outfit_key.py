"""
workers/stylist/outfit_key.py

One stable id for "this combination of items", independent of order.
Used by feedback events now, and by Generate Similar (exclude_outfit_keys)
next, so both sides always compute it the same way.
"""


def outfit_key(item_ids) -> str:
    return ":".join(sorted(str(i).lower() for i in item_ids))