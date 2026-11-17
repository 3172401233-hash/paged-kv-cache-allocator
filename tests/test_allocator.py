import unittest

from kv_cache import OutOfMemoryError, PagedKVCacheAllocator


class TestPrefixSharing(unittest.TestCase):
    def test_long_prefix_shared_across_requests(self):
        alloc = PagedKVCacheAllocator(page_size=4, num_pages=32)
        prompt = list(range(100, 116))  # 16-token system prompt = 4 pages

        alloc.create_request("a")
        alloc.append_tokens("a", prompt + [1, 2])
        pages_a = list(alloc.requests["a"].block_table)
        self.assertEqual(alloc.used_pages(), 5)
        alloc.finish_request("a")

        alloc.create_request("b")
        alloc.append_tokens("b", prompt + [3, 4])
        pages_b = alloc.requests["b"].block_table

        # The whole 4-page prefix is reused from the cache, not re-allocated.
        self.assertEqual(pages_b[:4], pages_a[:4])
        stats = alloc.stats()
        self.assertEqual(stats["cache_hits"], 4)
        # 4 shared prefix pages + 1 new tail; a's old tail was reclaimed.
        self.assertEqual(stats["used_pages"], 5)
        self.assertGreater(stats["reuse_rate"], 0.0)

    def test_identical_tail_blocks_deduped(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=8)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4])
        alloc.create_request("b")
        alloc.append_tokens("b", [1, 2, 3, 4])
        self.assertEqual(
            alloc.requests["a"].block_table, alloc.requests["b"].block_table
        )
        self.assertEqual(alloc.used_pages(), 2)


class TestCancelAndFinish(unittest.TestCase):
    def test_cancel_releases_all_pages(self):
        alloc = PagedKVCacheAllocator(page_size=4, num_pages=8)
        alloc.create_request("a")
        alloc.append_tokens("a", list(range(10)))  # 3 pages
        self.assertEqual(alloc.used_pages(), 3)

        alloc.cancel_request("a")
        self.assertEqual(alloc.used_pages(), 0)
        self.assertNotIn("a", alloc.requests)
        stats = alloc.stats()
        self.assertEqual(stats["reclaimed_pages"], 3)
        # Hashes are retained: a later identical prefix hits the cache.
        self.assertEqual(stats["cached_blocks"], 2)

    def test_cancel_frees_for_next_request(self):
        alloc = PagedKVCacheAllocator(page_size=4, num_pages=4)
        alloc.create_request("a")
        alloc.append_tokens("a", list(range(16)))  # fills every page
        alloc.cancel_request("a")
        alloc.create_request("b")
        alloc.append_tokens("b", list(range(16)))  # all cache hits, no OOM
        self.assertEqual(alloc.stats()["cache_hits"], 4)

    def test_double_release_raises(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=4)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2])
        alloc.finish_request("a")
        with self.assertRaises(KeyError):
            alloc.finish_request("a")


class TestForkAndCopyOnWrite(unittest.TestCase):
    def test_fork_shares_pages_and_cow_on_write(self):
        alloc = PagedKVCacheAllocator(page_size=4, num_pages=16)
        alloc.create_request("parent")
        alloc.append_tokens("parent", [1, 2, 3, 4, 5, 6])
        full_page, tail_page = alloc.requests["parent"].block_table

        alloc.fork("parent", "child")
        child = alloc.requests["child"]
        self.assertEqual(child.block_table, [full_page, tail_page])
        self.assertEqual(alloc.pages[full_page].ref_count, 2)
        self.assertEqual(alloc.pages[tail_page].ref_count, 2)

        # Child writes into the shared partial tail -> copy-on-write.
        alloc.append_tokens("child", [7, 8])
        self.assertEqual(alloc.stats()["cow_copies"], 1)
        new_tail = child.block_table[-1]
        self.assertNotEqual(new_tail, tail_page)
        self.assertEqual(alloc.pages[new_tail].tokens, (5, 6, 7, 8))
        # Parent untouched and now sole owner of the old tail.
        self.assertEqual(alloc.pages[tail_page].tokens, (5, 6))
        self.assertEqual(alloc.pages[tail_page].ref_count, 1)

        # Parent keeps writing in place: no extra COW.
        alloc.append_tokens("parent", [9])
        self.assertEqual(alloc.stats()["cow_copies"], 1)
        self.assertEqual(alloc.pages[tail_page].tokens, (5, 6, 9))

    def test_fork_cancel_keeps_parent_pages(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=8)
        alloc.create_request("p")
        alloc.append_tokens("p", [1, 2, 3, 4])
        alloc.fork("p", "c")
        alloc.cancel_request("c")
        for pid in alloc.requests["p"].block_table:
            self.assertEqual(alloc.pages[pid].ref_count, 1)
        alloc.append_tokens("p", [5, 6])  # still works
        self.assertEqual(alloc.requests["p"].num_tokens, 6)


class TestLRUEviction(unittest.TestCase):
    def test_evicts_least_recently_used_cached_page(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=6)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4])  # pages p0=[1,2], p1=[3,4]
        alloc.finish_request("a")  # both cached & free, p0 is LRU

        # Reuse the [1,2] prefix: p0 pinned, p1 becomes the LRU victim.
        alloc.create_request("b")
        alloc.append_tokens("b", [1, 2])
        victim_hash = alloc._hash_block(
            alloc.pages[alloc.requests["b"].block_table[0]].block_hash, (3, 4)
        )
        self.assertIn(victim_hash, alloc.hash_to_page)

        # Allocate past every never-used page so the cached victim is evicted.
        alloc.append_tokens("b", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        stats = alloc.stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertNotIn(victim_hash, alloc.hash_to_page)
        # The pinned prefix page survived eviction.
        self.assertEqual(alloc.pages[alloc.requests["b"].block_table[0]].tokens, (1, 2))

    def test_active_pages_never_evicted(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=3)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4])
        alloc.finish_request("a")

        alloc.create_request("b")
        alloc.append_tokens("b", [1, 2, 3, 4])  # reuses both cached pages
        alloc.create_request("c")
        alloc.append_tokens("c", [7, 8])  # takes the last free page
        self.assertEqual(alloc.stats()["evictions"], 0)
        with self.assertRaises(OutOfMemoryError):
            alloc.append_tokens("c", [9, 10])
        # Every page is still owned by an active request.
        for page in alloc.pages:
            self.assertGreater(page.ref_count, 0)


class TestOutOfMemory(unittest.TestCase):
    def test_oom_when_all_pages_pinned(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=3)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4, 5, 6])
        with self.assertRaises(OutOfMemoryError):
            alloc.append_tokens("a", [7])
        self.assertEqual(alloc.used_pages(), 3)

    def test_oom_on_fork_then_recover_after_cancel(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=3)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4])
        alloc.create_request("b")
        alloc.append_tokens("b", [7, 8])  # pool is now full
        alloc.fork("a", "c")  # fork itself is cheap: pages are shared
        with self.assertRaises(OutOfMemoryError):
            alloc.append_tokens("c", [5])
        alloc.cancel_request("b")  # frees one page
        alloc.append_tokens("c", [5])
        self.assertEqual(alloc.requests["c"].num_tokens, 5)


class TestSnapshotRestore(unittest.TestCase):
    def test_page_table_recovery(self):
        alloc = PagedKVCacheAllocator(page_size=2, num_pages=8)
        alloc.create_request("a")
        alloc.append_tokens("a", [1, 2, 3, 4, 5])
        alloc.fork("a", "b")
        alloc.append_tokens("b", [6])
        snap = alloc.snapshot()

        # Mutate after the snapshot: more appends, a cancel, a new request.
        alloc.append_tokens("a", [7, 8, 9, 10])
        alloc.cancel_request("b")
        alloc.create_request("c")
        alloc.append_tokens("c", list(range(6)))
        self.assertNotEqual(alloc.snapshot(), snap)

        restored = PagedKVCacheAllocator.restore(snap)
        self.assertEqual(restored.snapshot(), snap)

        # The restored allocator keeps working: refs, cache and COW intact.
        restored.append_tokens("a", [7, 8])
        self.assertEqual(restored.requests["a"].num_tokens, 7)
        restored.cancel_request("b")
        for pid in restored.requests["a"].block_table:
            self.assertGreaterEqual(restored.pages[pid].ref_count, 1)
        restored.create_request("d")
        restored.append_tokens("d", [1, 2, 3, 4])  # prefix cache still valid
        self.assertGreater(restored.stats()["cache_hits"], snap["counters"]["cache_hits"])


class TestMetrics(unittest.TestCase):
    def test_peak_reuse_fragmentation_and_reclaim(self):
        alloc = PagedKVCacheAllocator(page_size=4, num_pages=16)
        alloc.create_request("a")
        alloc.append_tokens("a", list(range(10)))  # 3 pages, peak so far
        alloc.create_request("b")
        alloc.append_tokens("b", list(range(100, 106)))  # 2 pages
        self.assertEqual(alloc.stats()["peak_used_pages"], 5)

        alloc.finish_request("a")  # 3 pages reclaimed, 2 full blocks cached
        alloc.create_request("c")
        alloc.append_tokens("c", list(range(8)))  # 2 cache hits

        stats = alloc.stats()
        self.assertEqual(stats["reclaimed_pages"], 3)
        self.assertEqual(stats["cache_hits"], 2)
        lookups = stats["cache_hits"] + stats["cache_misses"]
        self.assertAlmostEqual(stats["reuse_rate"], stats["cache_hits"] / lookups)
        # b's tail wastes 2 slots, c uses full pages only.
        self.assertEqual(stats["wasted_token_slots"], 2)
        self.assertAlmostEqual(
            stats["fragmentation"],
            stats["wasted_token_slots"] / (stats["used_pages"] * 4),
        )


if __name__ == "__main__":
    unittest.main()
