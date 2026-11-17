# paged-kv-cache-allocator

Python 分页 KV Cache 分配器，处理前缀共享、写时复制、淘汰和并发请求取消。

LLM 推理服务中，若每个请求都复制完整 KV Cache，多个请求共享长 system
prompt 时显存会迅速占满。本分配器以 vLLM PagedAttention 的方式管理显存：
逻辑 block table + 物理页 + 引用计数 + 内容哈希前缀缓存，相同前缀只存一份。

## 快速开始

```python
from kv_cache import PagedKVCacheAllocator, OutOfMemoryError

alloc = PagedKVCacheAllocator(page_size=16, num_pages=64)

alloc.create_request("req-1")
alloc.append_tokens("req-1", system_prompt)   # 满页自动哈希入缓存
alloc.create_request("req-2")
alloc.append_tokens("req-2", system_prompt)   # 命中前缀缓存，零新分配

child = alloc.fork("req-1", "req-1-beam")     # 共享全部页，ref+1
alloc.append_tokens("req-1-beam", [42])       # 写共享页触发 copy-on-write

alloc.cancel_request("req-2")                 # 取消：立即释放引用
print(alloc.stats())                          # 峰值页数/复用率/碎片/回收

snap = alloc.snapshot()                       # 崩溃恢复
alloc = PagedKVCacheAllocator.restore(snap)
```

运行场景演示（输出峰值页数、复用率、碎片与回收结果）：

```bash
python3 demo.py
```

运行测试：

```bash
python3 -m unittest
```

## 页生命周期

```
            allocate                    写满并哈希
  FREE ───────────────► ALLOCATED ───────────────► CACHED (pinned)
   ▲                      │   ▲                    ref>0, 在哈希表中
   │                      │   │ fork/前缀命中          │
   │                      │   └──────────────────────┘
   │                      │ ref+1
   │                 SHARED (ref>1)
   │                      │ 写入共享页
   │                      ▼
   │              COPY-ON-WRITE：复制新页，旧页 ref-1
   │                      │
   │      finish/cancel  │ ref-1
   └──────────────────────┴──────────┐
      ref=0 回到空闲链表(MRU 端)      │
      保留 block_hash 仍可被前缀命中  │
                                     ▼
              新分配从空闲链表 LRU 端取页 ──► EVICTED：
              若页带 hash 则从哈希表删除，页内容作废
```

各状态说明：

- **FREE**：在空闲链表中，按 LRU 排序。带 `block_hash` 的空闲页同时留在
  前缀缓存里，可被相同内容的 block 直接命中复用。
- **ALLOCATED**：被某请求独占（`ref_count == 1`），可原地追加 token。
- **CACHED (pinned)**：写满的页按「父块哈希 + token 内容」计算
  `block_hash` 注册进哈希表；只要 `ref_count > 0` 就不可淘汰。
- **SHARED**：`fork` 或前缀命中使 `ref_count > 1`。向共享页写入时触发
  copy-on-write：分配新页、拷贝内容、旧页引用减一，页表改指新页。
- **释放**：`finish_request` / `cancel_request` 对页表每页 `ref-1`；
  归零的页回到空闲链表 MRU 端，哈希保留，因此取消请求后其前缀仍可被
  后续请求复用。
- **EVICTED**：空闲页被新分配从 LRU 端取出时，若带哈希则从前缀缓存
  删除。被活跃请求引用的页不在空闲链表中，永远不会被淘汰；无页可分配
  时抛出 `OutOfMemoryError`。

## 指标

`stats()` 返回：

| 字段 | 含义 |
| --- | --- |
| `peak_used_pages` | 运行期间物理页占用峰值 |
| `reuse_rate` | 前缀缓存命中率（hits / (hits + misses)） |
| `fragmentation` | 内部碎片：活跃请求尾页空闲 token 槽 / 已用页总槽位 |
| `reclaimed_pages` | 累计回收（引用归零）的页数 |
| `evictions` / `cow_copies` | LRU 淘汰次数 / 写时复制次数 |

## 测试覆盖

`tests/test_allocator.py` 覆盖：长前缀共享与尾块去重、请求取消/结束的
引用释放、fork 共享与 copy-on-write、LRU 淘汰顺序、活跃页不可淘汰、
内存不足（OOM）及取消后恢复、页表快照/恢复（snapshot/restore 往返一致
且恢复后可继续服务）、指标统计。
