# DocMind 架构说明

本文只记录「为什么这么设计」，实现细节看代码。

## 1. 分层

```
api/        HTTP 边界：参数校验、鉴权、响应模型、SSE
services/   业务编排：入库流水线、索引注册表
rag/        纯逻辑：切片、向量、检索、Prompt、模型适配
db/models   持久化：6 张表
```

约束：`rag/` 不 import `api/`，`services/` 不 import `fastapi`。这样检索逻辑可以脱离 HTTP 单独跑（`scripts/eval_retrieval.py` 就是这么做的），评测结果不会被接口层噪声污染。

## 2. 数据模型

| 表 | 关键字段 | 说明 |
| --- | --- | --- |
| `users` | `tenant_id`, `email`, `password_hash` | 注册即开租户，1 人 1 租户 |
| `knowledge_bases` | `tenant_id`, `name` | `(tenant_id, name)` 唯一 |
| `documents` | `kb_id`, `checksum`, `chunk_count` | `checksum` 做幂等去重 |
| `chunks` | `doc_id`, `ordinal`, `text`, `embedding` | 向量以 JSON 存，可读可迁移 |
| `chat_sessions` | `kb_id`, `user_id` | 标题取首问前 40 字 |
| `messages` | `session_id`, `role`, `citations`, `latency_ms` | 引用结果随消息落库 |

索引：`chunks(kb_id, doc_id, ordinal)` 支撑按库顺序重建；`documents(kb_id)`、`users(email)` 唯一索引。

## 3. 入库流水线

```
上传 → 大小校验 → 解码(utf-8 → gb18030) → normalize → 切句 → 装箱切片
     → 批量向量化 → 事务写入 document + chunks → 索引版本自然失效
```

- **幂等**：`sha256(raw)` 命中同库已有文档直接 409，避免重复切片挤占 top_k；
- **原子**：document 与 chunks 同一事务写入，`flush` 一次拿 id，避免逐条 flush；
- **不阻塞事件循环**：解析+向量化在线程池执行。

## 4. 检索

```
query → tokenize（ASCII 词 + 中文二元组）
      ├─ BM25（k1=1.5, b=0.75，Lucene 风格 IDF）
      └─ 向量（哈希 / OpenAI embedding）
            ↓
        RRF 融合：score = Σ w_i / (60 + rank_i)
            ↓
        top_k + 各路排名（返回给前端展示「为什么命中」）
```

为什么 RRF 而不是加权分数：两路分数量纲完全不同（BM25 0~30，余弦 -1~1），归一化又会引入新的假设。RRF 只吃排名，工程上最稳。

权重默认 0.35 是实测结论，见 README 的扫描表。

## 5. 索引与并发

- `IndexRegistry` 以 `kb_id` 为键缓存 `(version, retriever)`，版本 = `(COUNT(chunks), MAX(chunk.id))`；
- 读路径无锁（命中直接返回），重建时持 `RLock`，最坏情况是同一知识库并发重建两次，结果一致；
- 缓存上限 128 个库，超出按插入顺序淘汰（够用即可，不引入 LRU 依赖）。

## 6. 流式问答

```
POST /chat
  → 检索（同步，毫秒级）
  → 写 user message，提交事务
  → StreamingResponse:
        event: meta    携带 citations（前端先渲染出处骨架）
        event: token   逐段文本
        event: done    字数、耗时
  → 生成结束后用独立 Session 写 assistant message
```

用独立 Session 落库的原因：流式响应期间请求级 Session 可能已随依赖退出而关闭，复用它会偶发 `Session is closed`。

## 7. 可观测性

- `X-Request-ID`：入站沿用，缺失则生成 12 位 hex，日志与错误体都带；
- 指标：`docmind_http_requests_total{method,path,status}`、`docmind_http_request_duration_seconds`（histogram）、`docmind_retrieval_duration`、`docmind_chat_chars_total` 等；
- 路由标签用 `route.path` 模板（`/api/v1/kbs/{kb_id}`）而不是真实路径，避免标签基数爆炸。

## 8. 容量估算（单实例）

| 项目 | 量级 | 说明 |
| --- | --- | --- |
| 内存/千切片 | ~40 MB | 384 维 float + 原文，Python 对象开销大 |
| 建索引耗时 | ~1.5 s / 千切片 | 纯 Python BM25 统计 + 余弦矩阵构建 |
| 检索延迟 | 30~60 ms | 千切片内暴力检索；上万切片需换 ANN |

结论：单实例舒适区是「几十个知识库、万级切片」。再往上要换 pgvector/FAISS，并考虑把索引放到独立服务。
