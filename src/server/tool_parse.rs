use crate::server::ollama_types::{Tool, ToolCall, ToolCallFunction};

/// Result of attempting to parse tool calls from model output.
#[derive(Debug)]
pub enum ToolParseResult {
    /// No tool calls detected — plain text response.
    Text(String),
    /// One or more tool calls extracted.
    ToolCalls {
        /// Text content before the tool call marker (may be empty).
        content: String,
        /// Parsed tool calls.
        calls: Vec<ToolCall>,
    },
}

/// Parse tool calls from raw model output text.
///
/// Tries model-specific formats in order, then falls back to raw JSON
/// scanning. The `tools` slice is used to validate that parsed tool
/// names match declared tools.
///
/// Supported formats (confirmed with real models):
/// - Qwen 2.5 JSON: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
/// - Qwen 3.5 XML: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
/// - Raw JSON scanning: JSON objects with `name` + `arguments`/`parameters` keys
///   (handles Llama 3.1's special-token-wrapped JSON and plain JSON output)
pub fn parse_tool_calls(output: &str, tools: &[Tool]) -> ToolParseResult {
    let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();

    // Strip special tokens that models may emit around tool calls
    let cleaned = strip_special_tokens(output);

    if let Some(result) = try_qwen(&cleaned, &tool_names) {
        return result;
    }
    if let Some(result) = try_raw_json(&cleaned, &tool_names) {
        return result;
    }

    ToolParseResult::Text(output.to_string())
}

/// Strip common special tokens that models emit around tool calls.
fn strip_special_tokens(output: &str) -> String {
    let tokens = [
        "<|eot_id|>",
        "<|end_header_id|>",
        "<|start_header_id|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    let mut result = output.to_string();
    for token in &tokens {
        result = result.replace(token, "");
    }
    result
}

/// Qwen: `<tool_call>...</tool_call>` blocks.
/// Supports two formats:
/// 1. JSON (Qwen 2.5): `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
/// 2. XML (Qwen 3.5): `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
fn try_qwen(output: &str, tool_names: &[&str]) -> Option<ToolParseResult> {
    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";

    if !output.contains(open_tag) {
        return None;
    }

    let mut calls = Vec::new();
    let content_end = output.find(open_tag).unwrap_or(0);
    let content = output[..content_end].trim().to_string();

    let mut search_from = 0;
    while let Some(start) = output[search_from..].find(open_tag) {
        let abs_start = search_from + start + open_tag.len();
        if let Some(end) = output[abs_start..].find(close_tag) {
            let inner = output[abs_start..abs_start + end].trim();
            // Try JSON format first
            if let Some(mut parsed) = parse_json_tool_calls(inner, tool_names) {
                calls.append(&mut parsed);
            }
            // Try XML format (Qwen 3.5): <function=name><parameter=key>value</parameter></function>
            else if let Some(call) = parse_qwen35_xml_tool_call(inner, tool_names) {
                calls.push(call);
            }
            search_from = abs_start + end + close_tag.len();
        } else {
            // No closing tag — try to parse remaining as JSON or XML
            let inner = output[abs_start..].trim();
            let inner = inner.split("<|im_end|>").next().unwrap_or(inner).trim();
            if let Some(mut parsed) = parse_json_tool_calls(inner, tool_names) {
                calls.append(&mut parsed);
            } else if let Some(call) = parse_qwen35_xml_tool_call(inner, tool_names) {
                calls.push(call);
            }
            break;
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(ToolParseResult::ToolCalls { content, calls })
    }
}

/// Parse Qwen 3.5 XML-style tool call:
/// `<function=name><parameter=key>value</parameter>...</function>`
fn parse_qwen35_xml_tool_call(inner: &str, tool_names: &[&str]) -> Option<ToolCall> {
    let func_prefix = "<function=";
    let func_start = inner.find(func_prefix)?;
    let after_prefix = &inner[func_start + func_prefix.len()..];
    let name_end = after_prefix.find('>')?;
    let name = after_prefix[..name_end].trim();

    if !tool_names.is_empty() && !tool_names.contains(&name) {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    let param_prefix = "<parameter=";
    let param_close = "</parameter>";
    let mut search = after_prefix;

    while let Some(p_start) = search.find(param_prefix) {
        let after_p = &search[p_start + param_prefix.len()..];
        if let Some(p_name_end) = after_p.find('>') {
            let param_name = after_p[..p_name_end].trim();
            let after_name = &after_p[p_name_end + 1..];
            let value_end = after_name.find(param_close).unwrap_or(after_name.len());
            let param_value = after_name[..value_end].trim();

            let json_value = serde_json::from_str(param_value)
                .unwrap_or(serde_json::Value::String(param_value.to_string()));
            arguments.insert(param_name.to_string(), json_value);

            search = &after_name[value_end..];
        } else {
            break;
        }
    }

    Some(ToolCall {
        id: None,
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: serde_json::Value::Object(arguments),
        },
    })
}

/// Raw JSON fallback: find JSON objects in the output that look like tool calls.
/// Handles both "entire output is JSON" and "JSON embedded in special tokens"
/// (e.g. Llama 3.1's `<|eot_id|>{"name":...}<|end_header_id|>`).
fn try_raw_json(output: &str, tool_names: &[&str]) -> Option<ToolParseResult> {
    let trimmed = output.trim();

    // Try entire output as JSON first
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(call) = extract_tool_call(&v, tool_names) {
                return Some(ToolParseResult::ToolCalls {
                    content: String::new(),
                    calls: vec![call],
                });
            }
        }
    }

    // Scan for JSON objects containing "name" — handles models that emit
    // special tokens around tool call JSON
    let mut calls = Vec::new();
    let mut content_end = None;
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(json_str) = extract_json_str(&trimmed[i..]) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(call) = extract_tool_call(&v, tool_names) {
                        if content_end.is_none() {
                            content_end = Some(i);
                        }
                        calls.push(call);
                        i += json_str.len();
                        continue;
                    }
                }
            }
        }
        i += 1;
    }

    if calls.is_empty() {
        return None;
    }

    let content = if let Some(end) = content_end {
        strip_special_tokens(&trimmed[..end]).trim().to_string()
    } else {
        String::new()
    };

    // Deduplicate (some models repeat the same call)
    let mut seen = std::collections::HashSet::new();
    calls.retain(|c| {
        let key = format!("{}:{}", c.function.name, c.function.arguments);
        seen.insert(key)
    });

    Some(ToolParseResult::ToolCalls { content, calls })
}

/// Extract the shortest valid JSON object string starting at position 0.
/// Tracks quoted strings to avoid counting braces inside string literals.
fn extract_json_str(s: &str) -> Option<&str> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse one or more tool calls from a JSON string.
/// Handles both single objects and arrays.
fn parse_json_tool_calls(json_str: &str, tool_names: &[&str]) -> Option<Vec<ToolCall>> {
    let trimmed = json_str.trim();

    // Try as array first
    if trimmed.starts_with('[') {
        let arr: Vec<serde_json::Value> = serde_json::from_str(trimmed).ok()?;
        let calls: Vec<ToolCall> = arr
            .into_iter()
            .filter_map(|v| extract_tool_call(&v, tool_names))
            .collect();
        if calls.is_empty() {
            return None;
        }
        return Some(calls);
    }

    // Try as single object
    if trimmed.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(trimmed).ok()?;
        let call = extract_tool_call(&v, tool_names)?;
        return Some(vec![call]);
    }

    None
}

/// Extract a ToolCall from a JSON value.
/// Supports both `{"name": ..., "arguments": ...}` and `{"name": ..., "parameters": ...}`.
fn extract_tool_call(v: &serde_json::Value, tool_names: &[&str]) -> Option<ToolCall> {
    let obj = v.as_object()?;
    let name = obj.get("name").and_then(|n| n.as_str())?;

    // Validate against declared tools (skip validation if no tools declared)
    if !tool_names.is_empty() && !tool_names.contains(&name) {
        return None;
    }

    let arguments = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));

    Some(ToolCall {
        id: None,
        function: ToolCallFunction {
            name: name.to_string(),
            arguments,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ollama_types::{Tool, ToolFunction};

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "get_weather".into(),
                description: "Get weather".into(),
                parameters: serde_json::json!({}),
            },
        }
    }

    fn search_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "search".into(),
                description: "Search".into(),
                parameters: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn test_plain_text() {
        let tools = vec![weather_tool()];
        match parse_tool_calls("Hello, how can I help?", &tools) {
            ToolParseResult::Text(t) => assert_eq!(t, "Hello, how can I help?"),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn test_qwen_format() {
        let tools = vec![weather_tool()];
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>";
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
                assert_eq!(calls[0].function.arguments["location"], "Paris");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_qwen_multiple_calls() {
        let tools = vec![weather_tool(), search_tool()];
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"restaurants\"}}\n</tool_call>";
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].function.name, "get_weather");
                assert_eq!(calls[1].function.name, "search");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_qwen35_xml_format() {
        let tools = vec![weather_tool()];
        let output = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
                assert_eq!(calls[0].function.arguments["location"], "Paris");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_qwen35_xml_with_thinking() {
        let tools = vec![weather_tool()];
        let output = "I need to check the weather.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>";
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { content, calls } => {
                assert!(content.contains("check the weather"));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
                assert_eq!(calls[0].function.arguments["location"], "Paris");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_raw_json_fallback() {
        let tools = vec![weather_tool()];
        let output = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_raw_json_unknown_tool_ignored() {
        let tools = vec![weather_tool()];
        let output = r#"{"name": "unknown_tool", "arguments": {"x": 1}}"#;
        match parse_tool_calls(output, &tools) {
            ToolParseResult::Text(_) => {}
            _ => panic!("Expected Text for unknown tool"),
        }
    }

    #[test]
    fn test_raw_json_with_braces_in_string() {
        let tools = vec![weather_tool()];
        let output = r#"{"name": "get_weather", "arguments": {"location": "Paris {France}"}}"#;
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.arguments["location"], "Paris {France}");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parameters_key_alias() {
        let tools = vec![weather_tool()];
        let output = r#"{"name": "get_weather", "parameters": {"location": "Paris"}}"#;
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls[0].function.arguments["location"], "Paris");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_llama31_special_token_wrapped_json() {
        let tools = vec![weather_tool()];
        let output = r#"<|eot_id|>{ "name": "get_weather", "parameters": { "location": "Paris" } }<|end_header_id|>

<|start_header_id|>system<|end_header_id|>

<|eot_id|>{ "name": "get_weather", "parameters": { "location": "Paris" } }<|end_header_id|> is the function call"#;
        match parse_tool_calls(output, &tools) {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1); // Deduplicated
                assert_eq!(calls[0].function.name, "get_weather");
                assert_eq!(calls[0].function.arguments["location"], "Paris");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }
}
