"""Paged KV cache allocator.

Implements a vLLM-style paged allocator:

* Logical block table per request mapping logical blocks to physical pages.
* Content-hash based prefix cache so identical token blocks share one page.
* Reference counting per physical page; pages referenced by active requests
  are never evicted.
* Copy-on-write when a request writes to a page shared by a fork or a
  cached prefix.
* LRU eviction restricted to unreferenced cached pages.
* Snapshot / restore of the full page-table state for crash recovery.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field


class OutOfMemoryError(RuntimeError):
    """Raised when no physical page can be allocated or evicted."""


@dataclass
class Page:
    """One physical page holding up to ``page_size`` tokens."""

    page_id: int
    ref_count: int = 0
    block_hash: str | None = None
    tokens: tuple = ()


@dataclass
class Request:
    """A sequence request owning a logical block table."""

    request_id: str
    block_table: list = field(default_factory=list)
    num_tokens: int = 0


class PagedKVCacheAllocator:
    def __init__(self, page_size: int, num_pages: int):
        if page_size <= 0 or num_pages <= 0:
            raise ValueError("page_size and num_pages must be positive")
        self.page_size = page_size
        self.num_pages = num_pages
        self.pages = [Page(page_id=i) for i in range(num_pages)]
        # LRU-ordered free list: key = page_id. Front = least recently used.
        self.free_pages: OrderedDict = OrderedDict(
            (i, None) for i in range(num_pages)
        )
        # Prefix cache: content hash -> page_id. Entries with ref_count == 0
        # are evictable; entries with ref_count > 0 are pinned by requests.
        self.hash_to_page: dict = {}
        self.requests: dict = {}

        self._peak_used = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
        self._cow_copies = 0
        self._reclaimed_pages = 0

    # ------------------------------------------------------------------ #
    # hashing
    # ------------------------------------------------------------------ #
    def _hash_block(self, parent_hash: str | None, tokens: tuple) -> str:
        h = hashlib.sha1()
        h.update((parent_hash or "").encode())
        h.update(repr(tuple(tokens)).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    # page allocation / eviction
    # ------------------------------------------------------------------ #
    def _allocate_page(self) -> Page:
        if not self.free_pages:
            raise OutOfMemoryError(
                f"no free physical pages (total={self.num_pages}); "
                "all pages are pinned by active requests"
            )
        pid, _ = self.free_pages.popitem(last=False)
        page = self.pages[pid]
        if page.block_hash is not None:
            # Evict an unreferenced cached block from the prefix cache.
            del self.hash_to_page[page.block_hash]
            self._evictions += 1
        page.ref_count = 1
        page.tokens = ()
        page.block_hash = None
        used = self.num_pages - len(self.free_pages)
        self._peak_used = max(self._peak_used, used)
        return page

    def _release_page(self, page: Page) -> None:
        page.ref_count -= 1
        if page.ref_count < 0:
            raise RuntimeError(f"page {page.page_id} ref_count underflow")
        if page.ref_count == 0:
            # Keep block_hash so the page stays in the prefix cache and can
            # be reused until it is evicted. MRU position in the free list.
            self.free_pages[page.page_id] = None
            self._reclaimed_pages += 1

    # ------------------------------------------------------------------ #
    # request lifecycle
    # ------------------------------------------------------------------ #
    def create_request(self, request_id: str) -> Request:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id: {request_id}")
        req = Request(request_id=request_id)
        self.requests[request_id] = req
        return req

    def _get_request(self, request_id: str) -> Request:
        try:
            return self.requests[request_id]
        except KeyError:
            raise KeyError(f"unknown request: {request_id}") from None

    def append_tokens(self, request_id: str, tokens) -> None:
        req = self._get_request(request_id)
        tokens = list(tokens)
        pos, n = 0, len(tokens)
        while pos < n:
            if req.block_table:
                last = self.pages[req.block_table[-1]]
                if len(last.tokens) < self.page_size:
                    if last.ref_count > 1:
                        last = self._copy_on_write(req, len(req.block_table) - 1)
                    take = min(self.page_size - len(last.tokens), n - pos)
                    last.tokens = last.tokens + tuple(tokens[pos : pos + take])
                    pos += take
                    req.num_tokens += take
                    if len(last.tokens) == self.page_size:
                        self._register_full_block(req, len(req.block_table) - 1)
                    continue
            consumed = self._append_new_block(req, tokens[pos:])
            pos += consumed
            req.num_tokens += consumed

    def _append_new_block(self, req: Request, remaining: list) -> int:
        parent_hash = (
            self.pages[req.block_table[-1]].block_hash if req.block_table else None
        )
        if len(remaining) >= self.page_size:
            chunk = tuple(remaining[: self.page_size])
            block_hash = self._hash_block(parent_hash, chunk)
            pid = self.hash_to_page.get(block_hash)
            if pid is not None:
                page = self.pages[pid]
                if page.ref_count == 0:
                    del self.free_pages[pid]
                page.ref_count += 1
                self._cache_hits += 1
                req.block_table.append(pid)
            else:
                self._cache_misses += 1
                page = self._allocate_page()
                page.tokens = chunk
                page.block_hash = block_hash
                self.hash_to_page[block_hash] = page.page_id
                req.block_table.append(page.page_id)
            return self.page_size
        page = self._allocate_page()
        page.tokens = tuple(remaining)
        req.block_table.append(page.page_id)
        return len(remaining)

    def _register_full_block(self, req: Request, index: int) -> None:
        """Hash a block that just became full; dedup against the cache."""
        page = self.pages[req.block_table[index]]
        parent_hash = (
            self.pages[req.block_table[index - 1]].block_hash if index > 0 else None
        )
        block_hash = self._hash_block(parent_hash, page.tokens)
        existing = self.hash_to_page.get(block_hash)
        if existing is not None and existing != page.page_id:
            # Identical content already cached: merge onto the cached page.
            other = self.pages[existing]
            if other.ref_count == 0:
                del self.free_pages[existing]
            other.ref_count += 1
            req.block_table[index] = existing
            page.block_hash = None
            self._release_page(page)
            self._cache_hits += 1
            return
        page.block_hash = block_hash
        self.hash_to_page[block_hash] = page.page_id

    def _copy_on_write(self, req: Request, index: int) -> Page:
        old = self.pages[req.block_table[index]]
        new = self._allocate_page()
        new.tokens = old.tokens
        req.block_table[index] = new.page_id
        self._release_page(old)
        self._cow_copies += 1
        return new

    def fork(self, parent_id: str, child_id: str) -> Request:
        parent = self._get_request(parent_id)
        child = self.create_request(child_id)
        child.block_table = list(parent.block_table)
        child.num_tokens = parent.num_tokens
        for pid in child.block_table:
            self.pages[pid].ref_count += 1
        return child

    def finish_request(self, request_id: str) -> None:
        self._free_request(request_id)

    def cancel_request(self, request_id: str) -> None:
        self._free_request(request_id)

    def _free_request(self, request_id: str) -> None:
        req = self._get_request(request_id)
        for pid in req.block_table:
            self._release_page(self.pages[pid])
        del self.requests[request_id]

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def used_pages(self) -> int:
        return self.num_pages - len(self.free_pages)

    def stats(self) -> dict:
        used = self.used_pages()
        # Internal fragmentation: unused token slots in the partially filled
        # tail page of every active request.
        wasted_slots = 0
        for req in self.requests.values():
            if req.block_table:
                tail = self.pages[req.block_table[-1]]
                wasted_slots += self.page_size - len(tail.tokens)
        lookups = self._cache_hits + self._cache_misses
        return {
            "num_pages": self.num_pages,
            "used_pages": used,
            "free_pages": len(self.free_pages),
            "peak_used_pages": self._peak_used,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "reuse_rate": (self._cache_hits / lookups) if lookups else 0.0,
            "evictions": self._evictions,
            "cow_copies": self._cow_copies,
            "reclaimed_pages": self._reclaimed_pages,
            "cached_blocks": len(self.hash_to_page),
            "wasted_token_slots": wasted_slots,
            "fragmentation": (
                wasted_slots / (used * self.page_size) if used else 0.0
            ),
        }

    # ------------------------------------------------------------------ #
    # snapshot / recovery
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Serialize the full allocator state (page tables included)."""
        return {
            "page_size": self.page_size,
            "num_pages": self.num_pages,
            "pages": [
                {
                    "ref_count": p.ref_count,
                    "block_hash": p.block_hash,
                    "tokens": list(p.tokens),
                }
                for p in self.pages
            ],
            "free_pages": list(self.free_pages.keys()),
            "requests": {
                rid: {
                    "block_table": list(req.block_table),
                    "num_tokens": req.num_tokens,
                }
                for rid, req in self.requests.items()
            },
            "counters": {
                "peak_used": self._peak_used,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "evictions": self._evictions,
                "cow_copies": self._cow_copies,
                "reclaimed_pages": self._reclaimed_pages,
            },
        }

    @classmethod
    def restore(cls, snap: dict) -> "PagedKVCacheAllocator":
        """Rebuild an allocator from :meth:`snapshot` output."""
        alloc = cls(snap["page_size"], snap["num_pages"])
        for i, state in enumerate(snap["pages"]):
            page = alloc.pages[i]
            page.ref_count = state["ref_count"]
            page.block_hash = state["block_hash"]
            page.tokens = tuple(state["tokens"])
            if page.block_hash is not None:
                alloc.hash_to_page[page.block_hash] = i
        alloc.free_pages = OrderedDict((pid, None) for pid in snap["free_pages"])
        for rid, rstate in snap["requests"].items():
            req = Request(
                request_id=rid,
                block_table=list(rstate["block_table"]),
                num_tokens=rstate["num_tokens"],
            )
            alloc.requests[rid] = req
        counters = snap["counters"]
        alloc._peak_used = counters["peak_used"]
        alloc._cache_hits = counters["cache_hits"]
        alloc._cache_misses = counters["cache_misses"]
        alloc._evictions = counters["evictions"]
        alloc._cow_copies = counters["cow_copies"]
        alloc._reclaimed_pages = counters["reclaimed_pages"]
        return alloc
