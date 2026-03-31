use minijinja::{Environment, Value};

use crate::server::ollama_types::{ChatMessage, Tool};

const CHATML_TEMPLATE: &str = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

/// Jinja2-based chat template engine.
///
/// Uses the model's embedded chat template when available,
/// falling back to ChatML format. The template is compiled once
/// at construction and reused for every render call.
pub struct ChatTemplateEngine {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateEngine {
    /// Create a new engine from the model's embedded template string.
    /// Falls back to ChatML if `template` is None.
    pub fn new(template: Option<&str>) -> Self {
        Self::with_tokens(template, "", "")
    }

    /// Create a new engine with explicit BOS/EOS token strings.
    pub fn with_tokens(template: Option<&str>, bos_token: &str, eos_token: &str) -> Self {
        let source = template.unwrap_or(CHATML_TEMPLATE).to_string();

        let mut env = Environment::new();

        env.set_unknown_method_callback(python_method_shim);

        // Register raise_exception as a global function (some templates use it)
        env.add_function("raise_exception", raise_exception);

        if let Err(e) = env.add_template_owned("chat".to_string(), source) {
            tracing::warn!("Failed to compile chat template, using ChatML fallback: {e}");
            let _ = env.add_template_owned("chat".to_string(), CHATML_TEMPLATE.to_string());
        }

        Self {
            env,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
        }
    }

    /// Render a prompt from messages with optional tool definitions.
    pub fn render_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {e}"))?;

        let msg_values: Vec<Value> = messages.iter().map(message_to_value).collect();

        // Build context map — single construction, tools added conditionally
        let mut ctx = std::collections::BTreeMap::<&str, Value>::new();
        ctx.insert("messages", Value::from(msg_values));
        ctx.insert("add_generation_prompt", Value::from(add_generation_prompt));
        ctx.insert("bos_token", Value::from(self.bos_token.as_str()));
        ctx.insert("eos_token", Value::from(self.eos_token.as_str()));

        if let Some(tools) = tools {
            if !tools.is_empty() {
                let tool_values: Vec<Value> = tools.iter().map(tool_to_value).collect();
                ctx.insert("tools", Value::from(tool_values));
            }
        }

        let rendered = tmpl
            .render(Value::from(ctx))
            .map_err(|e| anyhow::anyhow!("Failed to render chat template: {e}"))?;

        Ok(rendered)
    }
}

/// Shim for Python methods commonly used in HuggingFace chat templates.
/// Covers string methods (startswith, strip, split, replace) and
/// dict methods (items, keys, values, get).
fn python_method_shim(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, minijinja::Error> {
    // String methods
    if let Some(s) = value.as_str() {
        match method {
            "startswith" => {
                let prefix = args.first().and_then(|a| a.as_str()).unwrap_or("");
                return Ok(Value::from(s.starts_with(prefix)));
            }
            "endswith" => {
                let suffix = args.first().and_then(|a| a.as_str()).unwrap_or("");
                return Ok(Value::from(s.ends_with(suffix)));
            }
            "strip" => return Ok(Value::from(s.trim())),
            "lstrip" => return Ok(Value::from(s.trim_start())),
            "rstrip" => return Ok(Value::from(s.trim_end())),
            "split" => {
                let sep = args.first().and_then(|a| a.as_str()).unwrap_or(" ");
                let parts: Vec<Value> = s.split(sep).map(Value::from).collect();
                return Ok(Value::from(parts));
            }
            "replace" => {
                let old = args.first().and_then(|a| a.as_str()).unwrap_or("");
                let new = args.get(1).and_then(|a| a.as_str()).unwrap_or("");
                return Ok(Value::from(s.replace(old, new)));
            }
            _ => {}
        }
    }

    // Dict/object methods
    match method {
        "items" => {
            if let Ok(iter) = value.try_iter() {
                let items: Vec<Value> = iter
                    .filter_map(|k| {
                        let v = value.get_item(&k).ok()?;
                        Some(Value::from(vec![k, v]))
                    })
                    .collect();
                return Ok(Value::from(items));
            }
            Ok(Value::from(Vec::<Value>::new()))
        }
        "keys" => {
            if let Ok(iter) = value.try_iter() {
                return Ok(Value::from(iter.collect::<Vec<Value>>()));
            }
            Ok(Value::from(Vec::<Value>::new()))
        }
        "values" => {
            if let Ok(iter) = value.try_iter() {
                let vals: Vec<Value> = iter
                    .filter_map(|k| value.get_item(&k).ok())
                    .collect();
                return Ok(Value::from(vals));
            }
            Ok(Value::from(Vec::<Value>::new()))
        }
        "get" => {
            let key = args.first().cloned().unwrap_or(Value::UNDEFINED);
            let default = args.get(1).cloned().unwrap_or(Value::UNDEFINED);
            match value.get_item(&key) {
                Ok(v) if !v.is_undefined() => Ok(v),
                _ => Ok(default),
            }
        }
        _ => Err(minijinja::Error::new(
            minijinja::ErrorKind::UnknownMethod,
            format!("unknown method: {method}"),
        )),
    }
}

fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

fn message_to_value(msg: &ChatMessage) -> Value {
    let mut map = std::collections::BTreeMap::new();
    map.insert("role".to_string(), Value::from(msg.role.as_str()));
    map.insert(
        "content".to_string(),
        Value::from(msg.content.as_deref().unwrap_or("")),
    );
    if let Some(ref tool_calls) = msg.tool_calls {
        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                let func = minijinja::context! {
                    name => tc.function.name.as_str(),
                    arguments => Value::from_serialize(&tc.function.arguments),
                };
                minijinja::context! { function => func }
            })
            .collect();
        map.insert("tool_calls".to_string(), Value::from(calls));
    }
    Value::from(map)
}

fn tool_to_value(tool: &Tool) -> Value {
    let params = Value::from_serialize(&tool.function.parameters);
    let function = minijinja::context! {
        name => tool.function.name.as_str(),
        description => tool.function.description.as_str(),
        parameters => params,
    };
    minijinja::context! {
        type => tool.tool_type.as_str(),
        function => function,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ollama_types::*;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    fn system_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "get_weather".into(),
                description: "Get weather for a location".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }),
            },
        }
    }

    #[test]
    fn test_chatml_fallback_basic() {
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![user_msg("Hello")];
        let result = engine.render_prompt(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_fallback_multi_turn() {
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![
            system_msg("You are helpful."),
            user_msg("Hi"),
            assistant_msg("Hello!"),
            user_msg("How are you?"),
        ];
        let result = engine.render_prompt(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_no_generation_prompt() {
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![user_msg("Hello")];
        let result = engine.render_prompt(&messages, None, false).unwrap();
        assert!(!result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_with_tools_still_renders() {
        // ChatML fallback doesn't use tools in template, but shouldn't error
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![user_msg("What's the weather?")];
        let tools = vec![weather_tool()];
        let result = engine.render_prompt(&messages, Some(&tools), true).unwrap();
        assert!(result.contains("What's the weather?"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_custom_template_with_tools() {
        // A simple template that renders tools
        let template = r#"{% if tools is defined %}Tools: {% for tool in tools %}{{ tool.function.name }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"#;

        let engine = ChatTemplateEngine::new(Some(template));
        let messages = vec![user_msg("What's the weather?")];
        let tools = vec![weather_tool()];
        let result = engine.render_prompt(&messages, Some(&tools), true).unwrap();
        assert!(result.contains("Tools: get_weather"));
        assert!(result.contains("user: What's the weather?"));
        assert!(result.ends_with("assistant: "));
    }

    #[test]
    fn test_tool_message_role() {
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![
            user_msg("What's the weather in Paris?"),
            ChatMessage {
                role: "assistant".into(),
                content: Some(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: None,
                    function: ToolCallFunction {
                        name: "get_weather".into(),
                        arguments: serde_json::json!({"location": "Paris"}),
                    },
                }]),
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("Sunny, 22°C".into()),
                tool_calls: None,
            },
        ];
        let result = engine.render_prompt(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>tool\nSunny, 22°C<|im_end|>"));
    }

    #[test]
    fn test_empty_content_message() {
        let engine = ChatTemplateEngine::new(None);
        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: None,
        }];
        let result = engine.render_prompt(&messages, None, false).unwrap();
        assert!(result.contains("<|im_start|>assistant\n<|im_end|>"));
    }

    #[test]
    fn test_invalid_template_falls_back_to_chatml() {
        // Invalid Jinja2 syntax should fall back to ChatML
        let engine = ChatTemplateEngine::new(Some("{% invalid syntax {{{}}}"));
        let messages = vec![user_msg("Hello")];
        let result = engine.render_prompt(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_bos_eos_tokens() {
        let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";
        let engine = ChatTemplateEngine::with_tokens(Some(template), "<s>", "</s>");
        let messages = vec![user_msg("Hello")];
        let result = engine.render_prompt(&messages, None, false).unwrap();
        assert_eq!(result, "<s>Hello</s>");
    }
}
