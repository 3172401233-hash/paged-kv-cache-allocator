"""Scenario demo: shared long system prompt, fork, cancel, eviction, OOM.

Run: python3 demo.py
"""

from kv_cache import OutOfMemoryError, PagedKVCacheAllocator


def show(title, alloc):
    s = alloc.stats()
    print(f"\n== {title} ==")
    print(f"  used/peak pages : {s['used_pages']}/{s['peak_used_pages']} "
          f"of {s['num_pages']}")
    print(f"  reuse rate      : {s['reuse_rate']:.2%} "
          f"({s['cache_hits']} hits / {s['cache_misses']} misses)")
    print(f"  fragmentation   : {s['fragmentation']:.2%} "
          f"({s['wasted_token_slots']} wasted token slots)")
    print(f"  reclaimed pages : {s['reclaimed_pages']}")
    print(f"  evictions       : {s['evictions']}")
    print(f"  cow copies      : {s['cow_copies']}")


def main():
    page_size, num_pages = 16, 64
    alloc = PagedKVCacheAllocator(page_size=page_size, num_pages=num_pages)
    system_prompt = list(range(1000, 1000 + 4 * page_size))  # 4-page prompt

    # Three requests share the same long system prompt.
    for rid in ("req-1", "req-2", "req-3"):
        alloc.create_request(rid)
        alloc.append_tokens(rid, system_prompt)
        alloc.append_tokens(rid, [hash(rid) % 997] * 8)
    show("3 requests sharing a 64-token system prompt", alloc)

    # Beam-search style fork, then copy-on-write divergence.
    alloc.fork("req-1", "req-1-beam")
    alloc.append_tokens("req-1-beam", [42] * 20)
    show("fork req-1 and diverge (copy-on-write)", alloc)

    # Cancel a request: its private pages are reclaimed, the shared prefix
    # stays cached for future requests.
    alloc.cancel_request("req-2")
    show("cancel req-2", alloc)

    # Finish everything; prefix pages become evictable cache entries.
    for rid in ("req-1", "req-3", "req-1-beam"):
        alloc.finish_request(rid)
    show("all requests finished (prefix cached)", alloc)

    # A new request with the same prompt reuses the cached prefix.
    alloc.create_request("req-4")
    alloc.append_tokens("req-4", system_prompt + [7] * 30)
    show("req-4 reuses cached prefix", alloc)

    # Fill the pool to trigger LRU eviction of cold cached blocks, then OOM.
    try:
        i = 0
        while True:
            alloc.append_tokens("req-4", [10_000 + i] * page_size)
            i += 1
    except OutOfMemoryError as exc:
        print(f"\n== memory exhausted ==\n  {exc}")
    show("after LRU eviction and OOM", alloc)

    # Crash-recovery: snapshot, restore, keep serving.
    snap = alloc.snapshot()
    restored = PagedKVCacheAllocator.restore(snap)
    assert restored.snapshot() == snap
    print("\n== recovery ==\n  snapshot/restore round-trip OK, "
          f"{len(snap['requests'])} request page table(s) recovered")


if __name__ == "__main__":
    main()
