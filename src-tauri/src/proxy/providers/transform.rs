//! 格式转换模块
//!
//! 实现 Anthropic ↔ OpenAI 格式转换，用于 OpenRouter 支持
//! 参考: anthropic-proxy-rs

use super::inline_think::split_leading_think_block;
use crate::proxy::{
    error::ProxyError,
    json_canonical::canonical_json_string,
    tool_media::{
        chat_media_part_from_tool_part, flush_pending_chat_tool_media, plan_chat_tool_output_media,
        queue_chat_tool_output_media, ToolMediaScope,
    },
};
use serde_json::{json, Value};

const ANTHROPIC_BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

/// Strip only a leading Claude Code attribution line from system text.
///
/// Claude Code can send dynamic `x-anthropic-billing-header` metadata at the
/// start of `system`. If forwarded into OpenAI Chat messages or Responses
/// `instructions`, the rotating `cch=` value changes the prompt prefix on every
/// request and prevents prefix cache reuse (#2350). Later occurrences are kept
/// to avoid deleting user-authored prompt text.
pub(crate) fn strip_leading_anthropic_billing_header(text: &str) -> &str {
    if !text.starts_with(ANTHROPIC_BILLING_HEADER_PREFIX) {
        return text;
    }

    let Some(line_end) = text
        .as_bytes()
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')
    else {
        return "";
    };

    let bytes = text.as_bytes();
    let mut rest_start = line_end + 1;
    if bytes[line_end] == b'\r' && bytes.get(line_end + 1) == Some(&b'\n') {
        rest_start += 1;
    }

    let rest = &text[rest_start..];
    if let Some(stripped) = rest.strip_prefix("\r\n") {
        stripped
    } else if let Some(stripped) = rest.strip_prefix('\n') {
        stripped
    } else if let Some(stripped) = rest.strip_prefix('\r') {
        stripped
    } else {
        rest
    }
}

/// Detect OpenAI o-series reasoning models (o1, o3, o4-mini, etc.)
/// These models require `max_completion_tokens` instead of `max_tokens`.
pub fn is_openai_o_series(model: &str) -> bool {
    model.len() > 1
        && model.starts_with('o')
        && model.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit())
}

/// Claude Code 选定的思考强度档位（`/effort` 与 `output_config.effort` 的取值）。
///
/// 转换层只做「同一强度 → 目标协议的表达」这一件事，不按模型能力增删档位：强度是
/// 用户意图，上游认不认由上游裁决，代理不猜。因此 Chat / Responses / Gemini 三条
/// 转换路径共用同一份解析结果，各自只负责自己的编码方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingIntensity {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ThinkingIntensity {
    /// 解析 Claude Code 的档位名。未知取值返回 `None`（宁可不注入也不猜语义）。
    fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// 档位名。Claude Code 的取值与 OpenAI 同名，两边共用一套字符串。
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// OpenAI Chat `reasoning_effort` 与 Responses `reasoning.effort` 的取值。
    pub fn openai_effort(self) -> &'static str {
        self.as_str()
    }

    /// Gemini `thinkingConfig.thinkingLevel` 的取值。
    ///
    /// Gemini 的枚举最高只到 `high`（minimal / low / medium / high），没有 xhigh
    /// 与 max：两者都落到它的最高档，强度不会被降到 high 以下。
    pub fn gemini_thinking_level(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High | Self::XHigh | Self::Max => "high",
        }
    }
}

/// 把客户端给出的档位名编码成 OpenAI `reasoning_effort` / `reasoning.effort` 取值。
///
/// OpenAI 两条路径的档位名与客户端同名，原样透传，因此这张表同时是各客户端档位
/// 词汇的并集：Claude Code 用 low…max，Codex 另有 `minimal` 与 `ultra`。不做任何
/// 按模型能力的降级——上游认不认由上游裁决。
///
/// `none` / `off` / `disabled` 表示关闭推理，返回 `None`；未知取值同样返回 `None`
/// （宁可不注入也不猜语义）。
pub fn openai_effort_value(effort: &str) -> Option<&'static str> {
    match effort.trim().to_ascii_lowercase().as_str() {
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" => Some("max"),
        "ultra" => Some("ultra"),
        _ => None,
    }
}

/// 把客户端档位名编码成 Anthropic `output_config.effort` 的取值。
///
/// Anthropic 的档位枚举是 low / medium / high / xhigh / max：`minimal` 落到最低档
/// low，`ultra`（Codex 扩展档）落到最高档 max。这属于**协议枚举**层面的对齐，不是
/// 按模型能力降级——这两个词 Anthropic 协议里根本没有，原样发只会 400。
pub fn anthropic_effort_value(effort: &str) -> Option<&'static str> {
    match effort.trim().to_ascii_lowercase().as_str() {
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" | "ultra" => Some("max"),
        _ => None,
    }
}

/// 解析请求中 Claude Code 表达的思考强度。
///
/// 优先级：
/// 1. 显式 `output_config.effort` —— Claude Code 的 `/effort` 直接写在这里，档位
///    原样透传（含 `max` 与 `xhigh`）。未知取值忽略。
///    `thinking: disabled` 不影响它：用户关掉思考时 Claude Code 仍会照发 effort，
///    这里按用户选定的强度原样带走。
/// 2. 旧式 `thinking`：`adaptive` → `xhigh`（自适应即最高强度）；`enabled` 按
///    `budget_tokens` 折算（<4 000 low / <16 000 medium / 否则 high，缺 budget 记
///    high）；`disabled` 或缺失 → 无强度。
///
/// 结果与目标模型无关：调用方直接把强度编码成各自协议的字段。
pub fn resolve_thinking_intensity(body: &Value) -> Option<ThinkingIntensity> {
    if let Some(effort) = body
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
    {
        return ThinkingIntensity::parse(effort);
    }

    let thinking = body.get("thinking")?;
    match thinking.get("type").and_then(Value::as_str) {
        Some("adaptive") => Some(ThinkingIntensity::XHigh),
        Some("enabled") => {
            let budget = thinking.get("budget_tokens").and_then(Value::as_u64);
            Some(match budget {
                Some(b) if b < 4_000 => ThinkingIntensity::Low,
                Some(b) if b < 16_000 => ThinkingIntensity::Medium,
                // enabled 但没给 budget：按强推理处理
                Some(_) | None => ThinkingIntensity::High,
            })
        }
        _ => None,
    }
}

/// Anthropic 对 5.x 模型（Fable / Mythos / Opus / Sonnet）省略 `thinking` 时仍按
/// adaptive 思考（Fable 与 Opus 5.5 关不掉思考，Claude Code 干脆不发）。请求映射到
/// 别家模型后这个默认语义就丢了，[`resolve_thinking_intensity`] 会读成「没有强度」
/// ——把客户端的隐含意图显式化，强度才会被带到上游协议里。
pub fn make_default_adaptive_thinking_explicit(body: &mut Value, client_model: Option<&str>) {
    let Some(client_model) = client_model else {
        return;
    };
    if body
        .get("thinking")
        .is_some_and(|thinking| !thinking.is_null())
        || !crate::proxy::thinking_optimizer::adaptive_thinking_is_default(client_model)
    {
        return;
    }
    if let Some(obj) = body.as_object_mut() {
        obj.insert("thinking".to_string(), json!({"type": "adaptive"}));
    }
}

/// Anthropic 请求 → OpenAI Chat Completions 请求
///
/// 转换工具库 API：当前无生产调用方（连通性检查不再发真实请求，曾是其唯一 crate 内
/// 消费者），但保留其转换逻辑与下方测试套件，供代理转换路径复用 / 未来接线。
#[allow(dead_code)]
pub fn anthropic_to_openai(body: Value) -> Result<Value, ProxyError> {
    anthropic_to_openai_with_reasoning_content(body, false)
}

/// Anthropic 请求 → OpenAI Chat Completions 请求
///
/// `preserve_reasoning_content` 仅用于明确需要 DeepSeek/MiMo
/// `reasoning_content` 兼容字段的 provider。默认转换保持通用 OpenAI-compatible
/// 请求体，避免向严格后端发送未知字段。
pub fn anthropic_to_openai_with_reasoning_content(
    body: Value,
    preserve_reasoning_content: bool,
) -> Result<Value, ProxyError> {
    let mut result = json!({});

    // NOTE: 模型映射由上游统一处理（proxy::model_mapper），格式转换层只做结构转换。
    if let Some(model) = body.get("model").and_then(|m| m.as_str()) {
        result["model"] = json!(model);
    }

    let mut messages = Vec::new();

    // 处理 system prompt
    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            let text = strip_leading_anthropic_billing_header(text);
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        } else if let Some(arr) = system.as_array() {
            // 顶层 system 数组合并为一条 system 消息（跨轮字节稳定，不影响前缀缓存）
            let mut parts = Vec::new();
            for msg in arr {
                if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
                    let text = strip_leading_anthropic_billing_header(text);
                    if text.is_empty() {
                        continue;
                    }
                    parts.push(text.to_string());
                }
            }
            if !parts.is_empty() {
                messages.push(json!({"role": "system", "content": parts.join("\n")}));
            }
        }
    }

    // 转换 messages
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");
            let converted = convert_message_to_openai(role, content, preserve_reasoning_content)?;
            messages.extend(converted);
        }
    }

    result["messages"] = json!(messages);

    // 转换参数 — o-series 模型需要 max_completion_tokens
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("");
    if let Some(v) = body.get("max_tokens") {
        if is_openai_o_series(model) {
            result["max_completion_tokens"] = v.clone();
        } else {
            result["max_tokens"] = v.clone();
        }
    }
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }
    if let Some(v) = body.get("stop_sequences") {
        result["stop"] = v.clone();
    }
    if let Some(v) = body.get("stream") {
        result["stream"] = v.clone();
    }

    // Map Anthropic thinking → OpenAI reasoning_effort（强度是用户意图，上游认不认
    // 由上游裁决）
    if let Some(intensity) = resolve_thinking_intensity(&body) {
        result["reasoning_effort"] = json!(intensity.openai_effort());
    }

    // 转换 tools (过滤 BatchTool)
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|v| v.as_str()) != Some("BatchTool"))
            .map(|t| {
                let mut function = json!({
                    "name": t.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                });
                // 缺失的 description 省略而非输出 null：hosted 工具（web_search 等）
                // 与未填描述的自定义/MCP 工具都不带该字段，严格上游收到 null 会
                // 拒绝整个请求（400 "expected string, received null"）。
                if let Some(description) = t.get("description").filter(|d| !d.is_null()) {
                    function["description"] = description.clone();
                }
                function["parameters"] =
                    clean_schema(t.get("input_schema").cloned().unwrap_or(json!({})));
                json!({"type": "function", "function": function})
            })
            .collect();

        if !openai_tools.is_empty() {
            result["tools"] = json!(openai_tools);
        }
    }

    if let Some(v) = body.get("tool_choice") {
        result["tool_choice"] = map_tool_choice_to_chat(v);
    }

    Ok(result)
}

/// 为 OpenAI Chat Completions 流式请求注入 `stream_options.include_usage`。
///
/// OpenAI 兼容上游在流式下默认不在 SSE 里返回 usage，必须显式声明 include_usage
/// 才会在末尾吐 usage chunk。缺这一注入会导致流式请求的 token/成本/缓存全部漏记
/// （input/output/cache 全为 0）。保留客户端可能透传的其它 stream_options 字段，
/// 仅补 include_usage；非流式请求不动。
///
/// 由 Claude→openai_chat（claude.rs）与 Codex Responses→Chat（transform_codex_chat.rs）
/// 两条转换路径共用，确保两个客户端方向行为一致。
pub(crate) fn inject_openai_stream_include_usage(result: &mut Value) {
    let is_stream = result
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !is_stream {
        return;
    }
    match result.get_mut("stream_options") {
        Some(Value::Object(opts)) => {
            opts.insert("include_usage".to_string(), json!(true));
        }
        _ => {
            result["stream_options"] = json!({ "include_usage": true });
        }
    }
}

/// Translate an Anthropic `tool_choice` into the OpenAI Chat Completions form.
///
/// Anthropic forms:
///   "auto" / "any" / "none"           (string enum)
///   {"type": "auto" | "any" | "none"}
///   {"type": "tool", "name": "<X>"}
///
/// OpenAI Chat forms:
///   "auto" / "none" / "required"      (note: no "any" — use "required")
///   {"type": "function", "function": {"name": "<X>"}}
///
/// The Responses API uses a flatter `{"type":"function","name":"X"}` selector,
/// so it has a sibling `map_tool_choice_to_responses` in `transform_responses.rs`.
/// Keep the two in sync.
fn map_tool_choice_to_chat(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(s) => match s.as_str() {
            "any" => json!("required"),
            _ => json!(s),
        },
        Value::Object(obj) => match obj.get("type").and_then(|t| t.as_str()) {
            Some("any") => json!("required"),
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some("tool") => {
                let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("");
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            }
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

/// 转换单条消息到 OpenAI 格式（可能产生多条消息）
fn convert_message_to_openai(
    role: &str,
    content: Option<&Value>,
    preserve_reasoning_content: bool,
) -> Result<Vec<Value>, ProxyError> {
    let mut result = Vec::new();

    let content = match content {
        Some(c) => c,
        None => {
            result.push(json!({"role": role, "content": null}));
            return Ok(result);
        }
    };

    // 字符串内容
    if let Some(text) = content.as_str() {
        result.push(json!({"role": role, "content": text}));
        return Ok(result);
    }

    // 数组内容（多模态/工具调用）
    if let Some(blocks) = content.as_array() {
        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut pending_tool_media = Vec::new();
        // reasoning_parts: 仅在兼容 DeepSeek/MiMo thinking tool-call 路径时
        // 生成 reasoning_content，通用 OpenAI-compatible 路径不发送该非标准字段。
        let mut reasoning_parts = Vec::new();

        for block in blocks {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        content_parts.push(json!({"type": "text", "text": text}));
                    }
                }
                "image" => {
                    if let Some(image) =
                        chat_media_part_from_tool_part(block, ToolMediaScope::ImagesOnly)
                    {
                        content_parts.push(image);
                    }
                }
                "tool_use" => {
                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": canonical_json_string(&input)
                        }
                    }));
                }
                "tool_result" => {
                    // tool_result 变成单独的 tool role 消息
                    let tool_use_id = block
                        .get("tool_use_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("");
                    let content_val = block.get("content");
                    let media_plan = content_val.cloned().and_then(plan_chat_tool_output_media);
                    let content_str = if let Some(media_plan) = media_plan {
                        queue_chat_tool_output_media(
                            &mut pending_tool_media,
                            tool_use_id,
                            media_plan.media_parts,
                        );
                        media_plan.tool_content
                    } else {
                        // Keep the no-media representation exactly equal to
                        // the legacy converter for prompt-cache stability.
                        match content_val {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => canonical_json_string(v),
                            None => String::new(),
                        }
                    };
                    result.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content_str
                    }));
                }
                "thinking" => {
                    // 提取 thinking 内容，后续可作为 reasoning_content 传给需要它的上游。
                    if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                        if !thinking.is_empty() {
                            reasoning_parts.push(thinking.to_string());
                        }
                    }
                }
                "redacted_thinking" if preserve_reasoning_content => {
                    // Claude Code encrypts historical thinking into redacted_thinking blocks.
                    // MiMo/DeepSeek require non-empty reasoning_content on assistant tool-call
                    // messages, so inject a minimal placeholder when the real content is
                    // unavailable. Skip when preserve_reasoning_content is off (generic
                    // OpenAI-compatible path).
                    reasoning_parts.push("[redacted thinking]".to_string());
                }
                _ => {}
            }
        }

        // Chat tool messages cannot carry image parts. Keep parallel tool
        // results adjacent, then present all extracted media in one user turn
        // before any ordinary message content from the same Anthropic turn.
        flush_pending_chat_tool_media(&mut result, &mut pending_tool_media);

        // 添加带内容和/或工具调用的消息
        if !content_parts.is_empty() || !tool_calls.is_empty() {
            let mut msg = json!({"role": role});

            // 内容处理
            if content_parts.is_empty() {
                msg["content"] = Value::Null;
            } else if content_parts.len() == 1 {
                // 单 text block 简化为纯字符串
                if let Some(text) = content_parts[0].get("text") {
                    msg["content"] = text.clone();
                } else {
                    msg["content"] = json!(content_parts);
                }
            } else {
                msg["content"] = json!(content_parts);
            }

            // 工具调用
            if !tool_calls.is_empty() {
                msg["tool_calls"] = json!(tool_calls);
            }

            if preserve_reasoning_content && role == "assistant" && !tool_calls.is_empty() {
                let reasoning_content = if reasoning_parts.is_empty() {
                    "tool call".to_string()
                } else {
                    reasoning_parts.join("\n")
                };
                msg["reasoning_content"] = json!(reasoning_content);
            }

            result.push(msg);
        }

        return Ok(result);
    }

    // 其他情况直接透传
    result.push(json!({"role": role, "content": content}));
    Ok(result)
}

/// 清理工具参数的 JSON schema，并为根 schema 补齐 OpenAI 要求的 object 类型。
pub fn clean_schema(schema: Value) -> Value {
    clean_schema_inner(schema, true)
}

fn clean_schema_inner(mut schema: Value, is_root: bool) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        let missing_type = is_root && !obj.contains_key("type");
        if missing_type {
            obj.insert("type".to_string(), json!("object"));
        }
        if missing_type && !obj.contains_key("properties") {
            obj.insert("properties".to_string(), json!({}));
        }

        // 移除 "format": "uri"
        if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
            obj.remove("format");
        }

        // 递归清理嵌套 schema
        if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
            for (_, value) in properties.iter_mut() {
                *value = clean_schema_inner(value.clone(), false);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            *items = clean_schema_inner(items.clone(), false);
        }
    }
    schema
}

/// OpenAI 响应 → Anthropic 响应
pub fn openai_to_anthropic(body: Value) -> Result<Value, ProxyError> {
    let choices = body
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ProxyError::TransformError("No choices in response".to_string()))?;

    let choice = choices
        .first()
        .ok_or_else(|| ProxyError::TransformError("Empty choices array".to_string()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| ProxyError::TransformError("No message in choice".to_string()))?;

    let mut content = Vec::new();
    let mut has_tool_use = false;

    // DeepSeek provider 会把思考内容放在 message.reasoning_content。
    if let Some(reasoning_content) = message.get("reasoning_content").and_then(|r| r.as_str()) {
        if !reasoning_content.is_empty() {
            content.push(json!({"type": "thinking", "thinking": reasoning_content}));
        }
    }

    // 文本/拒绝内容。正文开头内联的 <think>/<thinking> 块（DeepSeek 系、MiniMax M3 等
    // Chat 兼容上游）拆成 thinking 块，与流式路径一致；只看第一段文本。
    let mut leading_text = true;
    let mut push_text = |content: &mut Vec<Value>, text: &str| {
        let split = if leading_text {
            leading_text = false;
            split_leading_think_block(text)
        } else {
            None
        };
        let Some((thinking, answer)) = split else {
            content.push(json!({"type": "text", "text": text}));
            return;
        };
        if !thinking.is_empty() {
            content.push(json!({"type": "thinking", "thinking": thinking}));
        }
        if !answer.is_empty() {
            content.push(json!({"type": "text", "text": answer}));
        }
    };
    if let Some(msg_content) = message.get("content") {
        if let Some(text) = msg_content.as_str() {
            if !text.is_empty() {
                push_text(&mut content, text);
            }
        } else if let Some(parts) = msg_content.as_array() {
            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                push_text(&mut content, text);
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(refusal) = part.get("refusal").and_then(|r| r.as_str()) {
                            if !refusal.is_empty() {
                                content.push(json!({"type": "text", "text": refusal}));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Some providers put refusal at message-level.
    if let Some(refusal) = message.get("refusal").and_then(|r| r.as_str()) {
        if !refusal.is_empty() {
            content.push(json!({"type": "text", "text": refusal}));
        }
    }

    // 工具调用（tool_calls）
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        if !tool_calls.is_empty() {
            has_tool_use = true;
        }
        for tc in tool_calls {
            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
            let empty_obj = json!({});
            let func = tc.get("function").unwrap_or(&empty_obj);
            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args_str = func
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }));
        }
    }
    // 兼容旧格式（function_call）
    if !has_tool_use {
        if let Some(function_call) = message.get("function_call") {
            let id = function_call
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("");
            let name = function_call
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let has_arguments = function_call.get("arguments").is_some();

            let input = match function_call.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                Some(v @ Value::Object(_)) | Some(v @ Value::Array(_)) => v.clone(),
                _ => json!({}),
            };

            if !name.is_empty() || has_arguments {
                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
                has_tool_use = true;
            }
        }
    }

    // 映射 finish_reason → stop_reason
    let stop_reason = choice
        .get("finish_reason")
        .and_then(|r| r.as_str())
        .map(|r| match r {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" | "function_call" => "tool_use",
            "content_filter" => "end_turn",
            other => {
                log::warn!(
                    "[Claude/OpenAI] Unknown finish_reason in non-streaming response: {other}"
                );
                "end_turn"
            }
        })
        .or(if has_tool_use { Some("tool_use") } else { None });

    // usage — map cache tokens from OpenAI format to Anthropic format
    let usage = body.get("usage").cloned().unwrap_or(json!({}));
    // OpenAI prompt_tokens 含缓存命中，Anthropic input_tokens 不含 → 减去 cache_read 与
    // cache_creation，使 input 成为 fresh input。本路径以 app_type="claude" 记账（calculator
    // 不再扣减），若不减则缓存会被计入 input 与各 cache 桶两次。三桶互斥，恒等：
    // input + cache_read + cache_creation == prompt_tokens（inclusive 上游）。
    // 与流式 build_anthropic_usage_json (#2774) 及 transform_gemini 的 saturating_sub 对称。
    // 最终 cache_read/cache_creation：直传字段优先于 OpenAI nested details。
    // 每环要求「存在且非 0」：显式 0 按未上报处理继续向后找（#8041：部分中转把
    // 靠前字段硬编码为桩 0，真值只在低顺位字段），DeepSeek 文档化命中字段末位兜底。
    fn provided_nonzero(value: Option<&Value>) -> Option<u64> {
        value.and_then(Value::as_u64).filter(|tokens| *tokens > 0)
    }
    let cached = provided_nonzero(usage.get("cache_read_input_tokens"))
        .or_else(|| provided_nonzero(usage.pointer("/prompt_tokens_details/cached_tokens")))
        .or_else(|| provided_nonzero(usage.get("prompt_cache_hit_tokens")))
        .unwrap_or(0);
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cache_write_tokens")
                .or_else(|| usage.pointer("/input_tokens_details/cache_write_tokens"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .saturating_sub(cached)
        .saturating_sub(cache_creation) as u32;
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let mut usage_json = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    });

    if cached > 0 {
        usage_json["cache_read_input_tokens"] = json!(cached);
    }
    if cache_creation > 0 {
        usage_json["cache_creation_input_tokens"] = json!(cache_creation);
    }

    let result = json!({
        "id": body.get("id").and_then(|i| i.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model").and_then(|m| m.as_str()).unwrap_or(""),
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage_json
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_to_openai_simple() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["model"], "claude-3-opus");
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_anthropic_to_openai_with_system() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "You are a helpful assistant."
        );
        assert_eq!(result["messages"][1]["role"], "user");
    }

    #[test]
    fn test_anthropic_to_openai_strips_leading_billing_header_from_system_string() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": "x-anthropic-billing-header: cc_version=2.1.119.47e; cc_entrypoint=sdk-cli; cch=a7754;\n\nYou are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "You are a helpful assistant."
        );
        assert_eq!(result["messages"][1]["role"], "user");
    }

    #[test]
    fn test_anthropic_to_openai_strips_billing_header_from_system_array_parts() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.119.47e; cc_entrypoint=sdk-cli; cch=a7754;\n"},
                {"type": "text", "text": "Stable prompt"}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(result["messages"][0]["content"], "Stable prompt");
        assert_eq!(result["messages"][1]["role"], "user");
    }

    #[test]
    fn test_anthropic_to_openai_preserves_prompt_after_billing_header_in_same_part() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.119.47e; cc_entrypoint=sdk-cli; cch=a7754;\n\nStable prompt part 1"},
                {"type": "text", "text": "Stable prompt part 2"}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "Stable prompt part 1\nStable prompt part 2"
        );
        assert_eq!(result["messages"][1]["role"], "user");
    }

    #[test]
    fn test_anthropic_to_openai_keeps_non_leading_billing_header_text() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": "Keep this literal:\nx-anthropic-billing-header: example",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "Keep this literal:\nx-anthropic-billing-header: example"
        );
    }

    #[test]
    fn test_anthropic_to_openai_with_tools() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["tools"][0]["type"], "function");
        assert_eq!(result["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            result["tools"][0]["function"]["parameters"]["type"],
            json!("object")
        );
        assert_eq!(
            result["tools"][0]["function"]["parameters"]["properties"]["location"]["type"],
            json!("string")
        );
    }

    #[test]
    fn test_anthropic_to_openai_defaults_missing_tool_schema_type() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"properties": {"location": {"type": "string"}}}
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let parameters = &result["tools"][0]["function"]["parameters"];
        assert_eq!(parameters["type"], json!("object"));
        assert_eq!(
            parameters["properties"]["location"]["type"],
            json!("string")
        );
    }

    #[test]
    fn test_anthropic_to_openai_defaults_empty_tool_schema() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Do work"}],
            "tools": [{"name": "do_work", "input_schema": {}}]
        });

        let result = anthropic_to_openai(input).unwrap();
        let parameters = &result["tools"][0]["function"]["parameters"];
        assert_eq!(parameters, &json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn test_clean_schema_only_defaults_root_to_object() {
        let schema = json!({
            "properties": {
                "nullable_value": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "list": {
                    "items": {"type": "string"}
                }
            }
        });

        let result = clean_schema(schema);
        assert_eq!(result["type"], json!("object"));
        assert_eq!(
            result["properties"]["nullable_value"],
            json!({"anyOf": [{"type": "string"}, {"type": "null"}]})
        );
        assert_eq!(
            result["properties"]["list"],
            json!({"items": {"type": "string"}})
        );
    }

    #[test]
    fn test_anthropic_to_openai_strips_cache_control_from_merged_system() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"].as_array().unwrap().len(), 2);
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "You are Claude Code.\nBe concise."
        );
        assert!(result["messages"][0].get("cache_control").is_none());
        assert_eq!(result["messages"][1]["role"], "user");
    }

    #[test]
    fn test_anthropic_to_openai_strips_cache_control_from_mixed_system() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be concise."}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "You are Claude Code.\nBe concise."
        );
        assert!(result["messages"][0].get("cache_control").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_preserves_mid_conversation_system_in_place() {
        // Claude Code 会在对话中间注入 system 消息（如 <total_tokens>），
        // 必须保持原位，不合并不上提，否则破坏前缀缓存。
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": "You are Claude Code.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "system", "content": "<total_tokens>14963538 tokens left</total_tokens>"},
                {"role": "user", "content": "Continue"}
            ]
        });

        let result = anthropic_to_openai(input).unwrap();
        let messages = result["messages"].as_array().unwrap();

        // 顶层 system 在最前面
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are Claude Code.");

        // 中途 system 保持原位（第 3 条，index=3），不被合并或上提
        assert_eq!(messages[3]["role"], "system");
        assert_eq!(
            messages[3]["content"],
            "<total_tokens>14963538 tokens left</total_tokens>"
        );

        // 总共 5 条消息，没有合并
        assert_eq!(messages.len(), 5);
    }

    #[test]
    fn test_anthropic_to_openai_strips_cache_control_from_conflicting_system() {
        let input = json!({
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"],
            "You are Claude Code.\nBe concise."
        );
        assert!(result["messages"][0].get("cache_control").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_tool_use() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert!(msg.get("tool_calls").is_some());
        assert_eq!(msg["tool_calls"][0]["id"], "call_123");
        assert!(msg.get("reasoning_content").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_tool_use_preserves_reasoning_content() {
        let input = json!({
            "model": "deepseek-v4-flash",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should call the tool."},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }]
        });

        let result = anthropic_to_openai_with_reasoning_content(input, true).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["reasoning_content"], "I should call the tool.");
        assert!(msg.get("tool_calls").is_some());
        assert_eq!(msg["tool_calls"][0]["id"], "call_123");
    }

    #[test]
    fn test_anthropic_to_openai_tool_use_injects_placeholder_reasoning_content_when_missing() {
        let input = json!({
            "model": "deepseek-v4-flash",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }]
        });

        let result = anthropic_to_openai_with_reasoning_content(input, true).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["reasoning_content"], "tool call");
        assert!(msg.get("tool_calls").is_some());
        assert_eq!(msg["tool_calls"][0]["id"], "call_123");
    }

    #[test]
    fn test_anthropic_to_openai_omits_missing_tool_description() {
        let input = json!({
            "model": "claude-opus-5",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "Bash", "description": "Run a bash command",
                 "input_schema": {"type": "object"}},
                {"name": "NoDesc",
                 "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
            ]
        });

        let result = anthropic_to_openai_with_reasoning_content(input, false).unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        // 带 description 的工具原样保留
        assert_eq!(
            tools[0]["function"]["description"],
            json!("Run a bash command")
        );
        // 缺 description 的自定义工具与 hosted 工具：省略字段，而不是序列化成 null
        assert!(tools[1]["function"].get("description").is_none());
        assert!(tools[2]["function"].get("description").is_none());
        assert!(tools[1]["function"].get("parameters").is_some());
    }

    #[test]
    fn test_anthropic_to_openai_tool_use_uses_redacted_thinking_placeholder() {
        let input = json!({
            "model": "mimo-v2.5-pro",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }]
        });

        let result = anthropic_to_openai_with_reasoning_content(input, true).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["reasoning_content"], "[redacted thinking]");
        assert_eq!(msg["tool_calls"][0]["id"], "call_123");
    }

    #[test]
    fn test_anthropic_to_openai_does_not_emit_reasoning_content_by_default() {
        let input = json!({
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should call the tool."},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert!(msg.get("tool_calls").is_some());
        assert!(msg.get("reasoning_content").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_skips_thinking_only_message() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "No visible content yet."}
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_anthropic_to_openai_tool_result() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 25°C"}
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let msg = &result["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_123");
        assert_eq!(msg["content"], "Sunny, 25°C");
    }

    #[test]
    fn test_anthropic_to_openai_no_media_tool_results_keep_legacy_representation() {
        let raw_json_string = "{ \"status\": \"ok\", \"count\": 2 }";
        let input = json!({
            "model": "claude-3-opus",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_string",
                        "content": raw_json_string
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_array",
                        "content": [{"type": "text", "text": "plain"}]
                    }
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let messages = result["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"], raw_json_string);
        assert_eq!(
            messages[1]["content"],
            canonical_json_string(&json!([{"type": "text", "text": "plain"}]))
        );
    }

    #[test]
    fn test_anthropic_to_openai_moves_tool_result_image_to_user_message() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call_image",
                    "content": [
                        {"type": "text", "text": "caption"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "CLAUDE_CHAT_IMAGE_SENTINEL"
                            },
                            "cache_control": {"type": "ephemeral"},
                            "prompt_cache_breakpoint": true
                        }
                    ]
                }]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let messages = result["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_image");
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("tool result media moved"));
        assert!(!messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("CLAUDE_CHAT_IMAGE_SENTINEL"));
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[1]["content"][0]["text"],
            "[cc-switch: media output of tool call call_image]"
        );
        assert_eq!(messages[1]["content"][1]["type"], "image_url");
        assert!(messages[1]["content"][1].get("cache_control").is_none());
        assert!(messages[1]["content"][1]
            .get("prompt_cache_breakpoint")
            .is_none());
        assert_eq!(
            messages[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,CLAUDE_CHAT_IMAGE_SENTINEL"
        );
    }

    #[test]
    fn test_anthropic_to_openai_batches_parallel_tool_result_media() {
        let input = json!({
            "model": "claude-3-opus",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "ONE"}
                        }]
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_2",
                        "content": [{
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "TWO"}
                        }]
                    }
                ]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        let messages = result["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn test_anthropic_to_openai_maps_remote_image_source() {
        let input = json!({
            "model": "claude-3-opus",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png"
                    },
                    "cache_control": {"type": "ephemeral"},
                    "prompt_cache_breakpoint": true
                }]
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(
            result["messages"][0]["content"][0]["image_url"]["url"],
            "https://example.com/image.png"
        );
        assert!(result["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
        assert!(result["messages"][0]["content"][0]
            .get("prompt_cache_breakpoint")
            .is_none());
    }

    #[test]
    fn test_openai_to_anthropic_simple() {
        let input = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["id"], "chatcmpl-123");
        assert_eq!(result["type"], "message");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 5);
    }

    fn chat_response_with_content(content: Value) -> Value {
        json!({
            "id": "chatcmpl-think",
            "object": "chat.completion",
            "model": "deepseek-v4.1-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    }

    #[test]
    fn test_openai_to_anthropic_splits_leading_inline_think_block() {
        // #7722：上游把思考内联进 content 开头，两种标签都要拆成 thinking 块
        for content in [
            "<thinking>tool call</thinking>\n日志很大。",
            "<think>tool call</think>\n\n日志很大。",
        ] {
            let result = openai_to_anthropic(chat_response_with_content(json!(content))).unwrap();
            assert_eq!(
                result["content"],
                json!([
                    {"type": "thinking", "thinking": "tool call"},
                    {"type": "text", "text": "日志很大。"}
                ]),
                "content {content:?}"
            );
        }
    }

    #[test]
    fn test_openai_to_anthropic_splits_inline_think_in_first_text_part_only() {
        let result = openai_to_anthropic(chat_response_with_content(json!([
            {"type": "text", "text": "<think>plan</think>answer"},
            {"type": "text", "text": "<think>kept</think>"}
        ])))
        .unwrap();
        assert_eq!(
            result["content"],
            json!([
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "answer"},
                {"type": "text", "text": "<think>kept</think>"}
            ])
        );
    }

    #[test]
    fn test_openai_to_anthropic_keeps_text_that_is_not_a_leading_think_block() {
        // 未闭合的块、正文中途出现的标签都原样保留
        for content in ["<think>unclosed", "see the <thinking>x</thinking> tag"] {
            let result = openai_to_anthropic(chat_response_with_content(json!(content))).unwrap();
            assert_eq!(
                result["content"],
                json!([{"type": "text", "text": content}]),
                "content {content:?}"
            );
        }
    }

    #[test]
    fn test_openai_to_anthropic_preserves_id_for_usage_dedup() {
        let input = json!({
            "id": "chatcmpl-claude-compatible",
            "object": "chat.completion",
            "model": "claude-sonnet-4-5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let result = openai_to_anthropic(input).unwrap();
        let usage = crate::proxy::usage::parser::TokenUsage::from_claude_response(&result)
            .expect("converted Anthropic response should parse usage");

        assert_eq!(
            usage.message_id.as_deref(),
            Some("chatcmpl-claude-compatible")
        );
        assert_eq!(
            usage.dedup_request_id(None),
            "session:chatcmpl-claude-compatible"
        );
    }

    #[test]
    fn test_openai_to_anthropic_with_tool_calls() {
        let input = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["id"], "call_123");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["content"][0]["input"]["location"], "Tokyo");
        assert_eq!(result["stop_reason"], "tool_use");
    }

    #[test]
    fn test_deepseek_reasoning_content_round_trips_for_tool_calls() {
        let upstream_response = json!({
            "id": "chatcmpl-deepseek",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Need the current date before calling weather.",
                    "content": "Let me check the date first.",
                    "tool_calls": [{
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let anthropic_response = openai_to_anthropic(upstream_response).unwrap();
        assert_eq!(anthropic_response["content"][0]["type"], "thinking");
        assert_eq!(
            anthropic_response["content"][0]["thinking"],
            "Need the current date before calling weather."
        );
        assert_eq!(anthropic_response["content"][1]["type"], "text");
        assert_eq!(anthropic_response["content"][2]["type"], "tool_use");
        assert_eq!(anthropic_response["content"][2]["id"], "call_date");

        let follow_up_request = json!({
            "model": "deepseek-v4-flash",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": anthropic_response["content"].clone()
            }]
        });
        let replayed = anthropic_to_openai_with_reasoning_content(follow_up_request, true).unwrap();
        let msg = &replayed["messages"][0];

        assert_eq!(
            msg["reasoning_content"],
            "Need the current date before calling weather."
        );
        assert_eq!(msg["tool_calls"][0]["id"], "call_date");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_date");
    }

    #[test]
    fn test_model_passthrough() {
        // 格式转换层只做结构转换，模型映射由上游 proxy::model_mapper 处理
        let input = json!({
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["model"], "gpt-4o");
    }

    #[test]
    fn test_anthropic_to_openai_does_not_inject_prompt_cache_key() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert!(result.get("prompt_cache_key").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_strips_all_cache_control() {
        let input = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
                ]
            }],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        });

        let result = anthropic_to_openai(input).unwrap();
        // System message: no cache_control
        assert!(result["messages"][0].get("cache_control").is_none());
        // User message: content simplified to string (no cache_control → flat string)
        assert_eq!(result["messages"][1]["content"], "Hello");
        // Tool: no cache_control
        assert!(result["tools"][0].get("cache_control").is_none());
    }

    /// 精确复现 Issue #3805 报告的 400 错误场景:
    /// GLM/Qwen 等严格校验模型拒绝 cache_control 和 content 数组格式
    #[test]
    fn test_regression_gh3805_no_cache_control_leak_to_openai() {
        let input = json!({
            "model": "glm-5.1",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
                ]}
            ],
            "tools": [{
                "name": "search",
                "description": "Search the web",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        });

        let result = anthropic_to_openai(input).unwrap();

        // 验证: messages 中不存在 cache_control
        for (i, msg) in result["messages"].as_array().unwrap().iter().enumerate() {
            assert!(
                msg.get("cache_control").is_none(),
                "messages[{i}] must not have cache_control"
            );
        }

        // 验证: content 中没有 cache_control
        for (i, msg) in result["messages"].as_array().unwrap().iter().enumerate() {
            if let Some(content) = msg.get("content") {
                assert!(
                    !content.is_array()
                        || content
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|part| part.get("cache_control").is_none()),
                    "messages[{i}] content parts must not have cache_control"
                );
            }
        }

        // 验证: system content 为纯字符串格式（不是数组）
        let sys_msg = &result["messages"][0];
        assert_eq!(sys_msg["role"], "system");
        assert!(
            sys_msg["content"].is_string(),
            "system content must be string, got: {}",
            sys_msg["content"]
        );

        // 验证: user content 为纯字符串格式（不是数组）
        let user_msg = &result["messages"][1];
        assert_eq!(user_msg["role"], "user");
        assert!(
            user_msg["content"].is_string(),
            "user content must be string, got: {}",
            user_msg["content"]
        );

        // 验证: tools 中不存在 cache_control
        if let Some(tools) = result["tools"].as_array() {
            for (i, tool) in tools.iter().enumerate() {
                assert!(
                    tool.get("cache_control").is_none(),
                    "tools[{i}] must not have cache_control"
                );
            }
        }
    }

    #[test]
    fn test_openai_to_anthropic_with_cache_tokens() {
        let input = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {
                    "cached_tokens": 80
                }
            }
        });

        let result = openai_to_anthropic(input).unwrap();
        // prompt_tokens(100) 含 cached(80)，转换后 input 应为 fresh = 100 - 80 = 20
        assert_eq!(result["usage"]["input_tokens"], 20);
        assert_eq!(result["usage"]["output_tokens"], 50);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 80);
    }

    #[test]
    fn test_openai_to_anthropic_with_direct_cache_fields() {
        let input = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 20
            }
        });

        let result = openai_to_anthropic(input).unwrap();
        // cache_read(60)+cache_creation(20) 均从 prompt(100) 扣除，fresh = 100 - 60 - 20 = 20
        // 守恒：input(20) + cache_read(60) + cache_creation(20) == prompt(100)
        assert_eq!(result["usage"]["input_tokens"], 20);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 60);
        assert_eq!(result["usage"]["cache_creation_input_tokens"], 20);
    }

    #[test]
    fn test_openai_to_anthropic_clamps_input_when_cache_exceeds_prompt() {
        // prompt(100) < cache_read(60)+cache_creation(50)=110：saturating 钳到 0，防下溢。
        // 钉桩：阻止未来把 saturating_sub 误改成普通减法(debug panic / release wrap)。
        let input = json!({
            "id": "chatcmpl-uf",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 50
            }
        });
        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["usage"]["input_tokens"], 0);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 60);
        assert_eq!(result["usage"]["cache_creation_input_tokens"], 50);
    }

    #[test]
    fn test_openai_to_anthropic_stub_zero_cache_read_falls_through_to_deepseek_hit_tokens() {
        // #8041 报告者 payload：显式 0 的 cache_read_input_tokens 按「未上报」处理
        // 继续向后找（与 usage/parser.rs、流式 extract_cache_read_tokens 同规则），
        // 真值在末位兜底 prompt_cache_hit_tokens；fresh input = 6650 - 6400 = 250。
        let input = json!({
            "id": "chatcmpl-stub0",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 6650,
                "completion_tokens": 16,
                "cache_read_input_tokens": 0,
                "prompt_cache_hit_tokens": 6400
            }
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["usage"]["cache_read_input_tokens"], 6400);
        assert_eq!(result["usage"]["input_tokens"], 250);
    }

    #[test]
    fn test_openai_to_anthropic_finish_reason_content_filter_maps_end_turn() {
        let input = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Blocked"},
                "finish_reason": "content_filter"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["stop_reason"], "end_turn");
    }

    #[test]
    fn test_openai_to_anthropic_with_legacy_function_call() {
        let input = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Tokyo\"}"
                    }
                },
                "finish_reason": "function_call"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["content"][0]["input"]["location"], "Tokyo");
        assert_eq!(result["stop_reason"], "tool_use");
    }

    #[test]
    fn test_openai_to_anthropic_with_content_parts_and_refusal() {
        let input = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "refusal", "refusal": "I can't do that"}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let result = openai_to_anthropic(input).unwrap();
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello");
        assert_eq!(result["content"][1]["type"], "text");
        assert_eq!(result["content"][1]["text"], "I can't do that");
    }

    #[test]
    fn test_is_openai_o_series() {
        assert!(is_openai_o_series("o1"));
        assert!(is_openai_o_series("o1-preview"));
        assert!(is_openai_o_series("o1-mini"));
        assert!(is_openai_o_series("o3"));
        assert!(is_openai_o_series("o3-mini"));
        assert!(is_openai_o_series("o4-mini"));
        assert!(!is_openai_o_series("gpt-4o"));
        assert!(!is_openai_o_series("openai-gpt"));
        assert!(!is_openai_o_series("o"));
        assert!(!is_openai_o_series(""));
    }

    // ── resolve_thinking_intensity unit tests ──

    #[test]
    fn test_output_config_effort_passes_through_verbatim() {
        for (effort, expected) in [
            ("low", ThinkingIntensity::Low),
            ("medium", ThinkingIntensity::Medium),
            ("high", ThinkingIntensity::High),
            ("xhigh", ThinkingIntensity::XHigh),
            ("max", ThinkingIntensity::Max),
        ] {
            let body = json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort}
            });
            assert_eq!(resolve_thinking_intensity(&body), Some(expected), "{effort}");
        }
    }

    #[test]
    fn test_output_config_max_is_not_downgraded_by_model() {
        // 强度是用户意图：目标模型认不认 max 与代理无关，一律原样带走。
        for model in ["gpt-5.4", "gpt-5-pro", "gpt-4o", "gemini-2.5-pro", "glm-5.2"] {
            let body = json!({
                "model": model,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"}
            });
            assert_eq!(
                resolve_thinking_intensity(&body),
                Some(ThinkingIntensity::Max),
                "{model}"
            );
        }
    }

    #[test]
    fn test_output_config_effort_survives_thinking_disabled() {
        // Claude Code 关掉思考时仍照发 effort（Workflow 子代理会同时发
        // disabled + effort）；按用户选定的强度原样透传，不做降档改写。
        let body = json!({
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "max"}
        });
        assert_eq!(
            resolve_thinking_intensity(&body),
            Some(ThinkingIntensity::Max)
        );
    }

    #[test]
    fn test_output_config_effort_wins_over_thinking() {
        let body = json!({
            "output_config": {"effort": "low"},
            "thinking": {"type": "adaptive"}
        });
        assert_eq!(resolve_thinking_intensity(&body), Some(ThinkingIntensity::Low));
    }

    #[test]
    fn test_output_config_unknown_value_yields_none() {
        let body = json!({"output_config": {"effort": "turbo"}});
        assert_eq!(resolve_thinking_intensity(&body), None);
    }

    #[test]
    fn test_thinking_budget_thresholds() {
        for (budget, expected) in [
            (1024u64, ThinkingIntensity::Low),
            (3999, ThinkingIntensity::Low),
            (4000, ThinkingIntensity::Medium),
            (8000, ThinkingIntensity::Medium),
            (15999, ThinkingIntensity::Medium),
            (16000, ThinkingIntensity::High),
            (32000, ThinkingIntensity::High),
        ] {
            let body = json!({"thinking": {"type": "enabled", "budget_tokens": budget}});
            assert_eq!(resolve_thinking_intensity(&body), Some(expected), "{budget}");
        }
        // enabled 但没给 budget：按强推理处理
        let no_budget = json!({"thinking": {"type": "enabled"}});
        assert_eq!(
            resolve_thinking_intensity(&no_budget),
            Some(ThinkingIntensity::High)
        );
    }

    #[test]
    fn test_thinking_adaptive_maps_xhigh() {
        let body = json!({"thinking": {"type": "adaptive"}});
        assert_eq!(
            resolve_thinking_intensity(&body),
            Some(ThinkingIntensity::XHigh)
        );
    }

    #[test]
    fn test_thinking_off_has_no_intensity() {
        for body in [
            json!({"thinking": {"type": "disabled"}}),
            json!({"messages": [{"role": "user", "content": "Hello"}]}),
        ] {
            assert_eq!(resolve_thinking_intensity(&body), None);
        }
    }

    #[test]
    fn test_openai_effort_keeps_every_tier() {
        assert_eq!(ThinkingIntensity::Low.openai_effort(), "low");
        assert_eq!(ThinkingIntensity::Medium.openai_effort(), "medium");
        assert_eq!(ThinkingIntensity::High.openai_effort(), "high");
        assert_eq!(ThinkingIntensity::XHigh.openai_effort(), "xhigh");
        assert_eq!(ThinkingIntensity::Max.openai_effort(), "max");
    }

    #[test]
    fn test_gemini_thinking_level_clamps_to_its_top_tier() {
        // Gemini 的 thinkingLevel 枚举没有 xhigh / max，两者都落到它的最高档 high。
        assert_eq!(ThinkingIntensity::Low.gemini_thinking_level(), "low");
        assert_eq!(ThinkingIntensity::Medium.gemini_thinking_level(), "medium");
        assert_eq!(ThinkingIntensity::High.gemini_thinking_level(), "high");
        assert_eq!(ThinkingIntensity::XHigh.gemini_thinking_level(), "high");
        assert_eq!(ThinkingIntensity::Max.gemini_thinking_level(), "high");
    }

    #[test]
    fn test_default_adaptive_client_model_keeps_effort_after_mapping() {
        // Claude Code omits `thinking` for Fable when thinking is "off" (Fable
        // cannot turn it off); after mapping to another vendor the effort must survive.
        for client_model in [
            "claude-fable-5-1",
            "claude-sonnet-5",
            "claude-mythos-5",
            "claude-opus-5",
            "claude-opus-5-5",
        ] {
            let mut body = json!({"model": "gpt-6-sol", "output_config": {"effort": "high"}});
            make_default_adaptive_thinking_explicit(&mut body, Some(client_model));
            assert_eq!(body["thinking"]["type"], "adaptive", "{client_model}");
            assert_eq!(
                resolve_thinking_intensity(&body),
                Some(ThinkingIntensity::High),
                "{client_model}"
            );
        }
    }

    #[test]
    fn test_default_adaptive_helper_leaves_other_requests_alone() {
        // Opus 4.8 does not think by default: `thinking` stays omitted.
        let mut omitted = json!({"model": "gpt-6-sol", "output_config": {"effort": "high"}});
        make_default_adaptive_thinking_explicit(&mut omitted, Some("claude-opus-4-8"));
        assert!(omitted.get("thinking").is_none());
        // 显式 effort 本身就是强度，省略 thinking 不影响解析结果。
        assert_eq!(
            resolve_thinking_intensity(&omitted),
            Some(ThinkingIntensity::High)
        );

        // An explicit `disabled` (Claude Code's Sonnet 5 "off") is kept.
        let mut disabled = json!({
            "model": "gpt-6-sol",
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"}
        });
        make_default_adaptive_thinking_explicit(&mut disabled, Some("claude-sonnet-5"));
        assert_eq!(disabled["thinking"]["type"], "disabled");

        let mut unknown_client = json!({"model": "gpt-6-sol"});
        make_default_adaptive_thinking_explicit(&mut unknown_client, None);
        assert!(unknown_client.get("thinking").is_none());

        let mut null_thinking = json!({"model": "gpt-6-sol", "thinking": null});
        make_default_adaptive_thinking_explicit(&mut null_thinking, Some("claude-fable-5-1"));
        assert_eq!(null_thinking["thinking"]["type"], "adaptive");
    }

    // ── Integration: anthropic_to_openai carries the intensity verbatim ──

    #[test]
    fn test_anthropic_to_openai_forwards_effort_for_any_model() {
        // 不再按模型白名单门控：非 o 系/GPT-5+ 的上游同样收到用户选定的强度。
        for model in ["gpt-4o", "glm-5.2", "kimi-k3", "deepseek-v4-pro"] {
            let input = json!({
                "model": model,
                "max_tokens": 1024,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"},
                "messages": [{"role": "user", "content": "Hello"}]
            });
            let result = anthropic_to_openai(input).unwrap();
            assert_eq!(result["reasoning_effort"], "max", "{model}");
        }
    }

    #[test]
    fn test_anthropic_to_openai_forwards_xhigh() {
        let input = json!({
            "model": "glm-5.2",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["reasoning_effort"], "xhigh");
    }

    #[test]
    fn test_reasoning_model_thinking_enabled_small_budget() {
        let input = json!({
            "model": "o3",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["reasoning_effort"], "low");
    }

    #[test]
    fn test_reasoning_model_thinking_adaptive() {
        let input = json!({
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["reasoning_effort"], "xhigh");
    }

    #[test]
    fn test_reasoning_model_no_thinking_no_effort() {
        let input = json!({
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert!(result.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_o_series_max_completion_tokens() {
        for model in &["o1", "o3-mini", "o4-mini"] {
            let input = json!({
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "Hello"}]
            });

            let result = anthropic_to_openai(input).unwrap();
            assert!(
                result.get("max_tokens").is_none(),
                "{model} should not have max_tokens"
            );
            assert_eq!(
                result["max_completion_tokens"], 4096,
                "{model} should use max_completion_tokens"
            );
        }
    }

    #[test]
    fn test_anthropic_to_openai_non_o_series_keeps_max_tokens() {
        let input = json!({
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_openai(input).unwrap();
        assert_eq!(result["max_tokens"], 1024);
        assert!(result.get("max_completion_tokens").is_none());
    }

    fn run_tool_choice(value: Value) -> Value {
        let input = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "name": "search",
                "description": "search the web",
                "input_schema": {"type": "object", "properties": {}}
            }],
            "tool_choice": value,
        });
        anthropic_to_openai(input).unwrap()["tool_choice"].clone()
    }

    #[test]
    fn tool_choice_string_any_maps_to_required() {
        assert_eq!(run_tool_choice(json!("any")), json!("required"));
    }

    #[test]
    fn tool_choice_string_auto_and_none_pass_through() {
        assert_eq!(run_tool_choice(json!("auto")), json!("auto"));
        assert_eq!(run_tool_choice(json!("none")), json!("none"));
    }

    #[test]
    fn tool_choice_object_any_maps_to_required() {
        assert_eq!(run_tool_choice(json!({"type": "any"})), json!("required"));
    }

    #[test]
    fn tool_choice_object_auto_and_none_collapse_to_string() {
        assert_eq!(run_tool_choice(json!({"type": "auto"})), json!("auto"));
        assert_eq!(run_tool_choice(json!({"type": "none"})), json!("none"));
    }

    #[test]
    fn tool_choice_forced_tool_maps_to_nested_function_selector() {
        // Anthropic {"type":"tool","name":"X"} must become OpenAI Chat
        // {"type":"function","function":{"name":"X"}} — the *nested* form, not
        // the flat Responses-API form.
        assert_eq!(
            run_tool_choice(json!({"type": "tool", "name": "search"})),
            json!({"type": "function", "function": {"name": "search"}}),
        );
    }
}
