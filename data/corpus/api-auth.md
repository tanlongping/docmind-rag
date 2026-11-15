# 开放平台鉴权说明

## 获取 access_token

调用 `POST /oauth/token`，请求体包含 `client_id`、`client_secret`、`grant_type=client_credentials`。返回的 `access_token` 默认有效期 7200 秒，过期后需要重新获取。

## 请求头格式

所有业务接口都需要在请求头带上 `Authorization: Bearer <access_token>`。缺少该头会返回 401，token 过期返回 401 且响应体 code 为 `token_expired`。

## 调用频率限制

默认配额为每个应用每分钟 600 次请求，超出后返回 429，响应头 `Retry-After` 给出建议重试秒数。配额可按应用维度申请提升。

## 错误码约定

- 40001：参数缺失或格式错误
- 40101：签名校验失败
- 42901：触发限流
- 50001：服务内部错误，可重试

## 幂等性

写接口支持 `Idempotency-Key` 请求头，同一个 key 在 24 小时内重复提交只会生效一次，适合支付、下单类操作。
