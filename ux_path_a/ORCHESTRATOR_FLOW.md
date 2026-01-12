# Orchestrator Flow - How It Works After User Submits a Command

## Overview

The orchestrator (`ux_path_a/backend/core/orchestrator.py`) is the brain of the chat system. It coordinates between the LLM (OpenAI), tools (data sources, analysis), and the database to generate intelligent responses.

## Complete Flow Diagram

```
User submits message
    ↓
[API Endpoint: /api/chat/messages]
    ↓
1. Save user message to database
    ↓
2. Load conversation history (last 10 messages)
    ↓
[Orchestrator.process_message()]
    ↓
3. Build message context:
   - System prompt (rules, safety, tool instructions)
   - Last 10 messages from history
   - Current user message
    ↓
4. Get available tools from ToolRegistry
   - get_symbol_data
   - get_bars
   - analyze_trend
   - calculate_indicators
    ↓
5. FIRST LLM CALL (with function calling enabled)
   - OpenAI decides: answer directly OR call tools
    ↓
   ┌─────────────────┬─────────────────┐
   │                 │                 │
   │  No Tool Calls  │  Tool Calls     │
   │                 │                 │
   └────────┬────────┴────────┬────────┘
            │                 │
            │                 ↓
            │         6. Execute Tools
            │            - Parse tool name & args
            │            - Call ToolRegistry.execute_tool()
            │            - Tool connects to Trading Platform
            │            - Returns real market data
            │                 │
            │                 ↓
            │         7. Add tool results to conversation
            │            - Add assistant message (tool calls)
            │            - Add function messages (tool results)
            │                 │
            │                 ↓
            │         8. SECOND LLM CALL
            │            - LLM receives tool results
            │            - Generates final response using real data
            │                 │
            └─────────────────┴─────────┐
                                        ↓
                            9. Return response:
                               - content (final answer)
                               - tool_calls (what tools were used)
                               - tool_results (tool outputs)
                               - token_usage (cost tracking)
                                        ↓
[Back to API Endpoint]
    ↓
10. Save assistant message to database
    - Store response content
    - Store tool_calls (JSON)
    - Store tool_results (JSON)
    - Store token_usage (JSON)
    ↓
11. Update session
    - Update timestamp
    - Track total tokens used
    ↓
12. Return response to frontend
```

## Step-by-Step Breakdown

### Step 1: User Message Arrives
**Location:** `ux_path_a/backend/api/chat.py` → `send_message()`

```python
# User sends: "What is the current price of SPY?"
message = ChatMessageRequest(content="What is the current price of SPY?")
```

### Step 2: Save User Message
```python
user_db_msg = DBChatMessage(
    session_id=session_id,
    role="user",
    content=message.content,
)
db.add(user_db_msg)
db.commit()
```

### Step 3: Load Conversation History
```python
# Gets last 10 messages from database for context
conversation_history = [
    {"role": "user", "content": "...", ...},
    {"role": "assistant", "content": "...", ...},
    # ... up to 10 messages
]
```

### Step 4: Call Orchestrator
```python
response = await orchestrator.process_message(
    message="What is the current price of SPY?",
    session_id="...",
    conversation_history=conversation_history,
)
```

### Step 5: Orchestrator Builds Context
**Location:** `orchestrator.py` → `process_message()`

```python
messages = [
    {"role": "system", "content": system_prompt},  # Rules & safety
    # ... last 10 messages from history
    {"role": "user", "content": "What is the current price of SPY?"}
]
```

**System Prompt includes:**
- Education-only framing
- Mandatory tool usage (INV-LLM-02)
- Risk disclosures
- No data fabrication rules

### Step 6: Get Tool Definitions
```python
tools = self.tool_registry.get_function_definitions()
# Returns OpenAI function calling format:
# [
#   {
#     "name": "get_symbol_data",
#     "description": "Get current market data for a symbol",
#     "parameters": {...}
#   },
#   ...
# ]
```

### Step 7: FIRST LLM CALL
```python
response = self.client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    functions=tools,  # Available tools
    function_call="auto",  # LLM decides if tools needed
)
```

**What happens:**
- LLM analyzes the question
- Sees it needs current price data
- Decides to call `get_symbol_data` tool
- Returns tool call request (not final answer)

### Step 8: Detect Tool Calls
```python
message_response = response.choices[0].message

# Check if LLM wants to call tools
if message_response.tool_calls:
    # LLM wants to use tools!
    function_calls = message_response.tool_calls
```

**Example tool call:**
```python
{
    "function": {
        "name": "get_symbol_data",
        "arguments": '{"symbol": "SPY"}'
    }
}
```

### Step 9: Execute Tools
**Location:** `orchestrator.py` → `_execute_tools()`

```python
for func_call in function_calls:
    tool_name = func_call.function.name  # "get_symbol_data"
    tool_args = json.loads(func_call.function.arguments)  # {"symbol": "SPY"}
    
    # Execute via registry
    result = await self.tool_registry.execute_tool(
        tool_name="get_symbol_data",
        arguments={"symbol": "SPY"},
        session_id=session_id,
    )
```

**Tool Execution Flow:**
1. Registry finds the tool
2. Tool connects to Trading Copilot Platform
3. Fetches real market data (not fabricated!)
4. Returns data:
   ```python
   {
       "symbol": "SPY",
       "price": 485.23,
       "volume": 1234567,
       ...
   }
   ```

### Step 10: Add Tool Results to Conversation
```python
# Add assistant message (tool call request)
messages.append({
    "role": "assistant",
    "content": "",
    "tool_calls": [...]
})

# Add tool results
for tool_result in tool_results:
    messages.append({
        "role": "function",
        "name": "get_symbol_data",
        "content": '{"symbol": "SPY", "price": 485.23, ...}'
    })
```

### Step 11: SECOND LLM CALL
```python
final_response = self.client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,  # Now includes tool results
    # No functions parameter - just generate answer
)
```

**What happens:**
- LLM sees the tool results
- Generates natural language response using real data
- Includes risk disclaimers
- Formats as markdown

**Example response:**
```
Based on the current market data for SPY:

**Current Price:** $485.23**

*This tool is for educational purposes only. Not financial advice.*
```

### Step 12: Combine Token Usage
```python
# If tools were used, combine tokens from both calls
total_tokens = response.usage.total_tokens + final_response.usage.total_tokens
```

### Step 13: Return Response
```python
return {
    "content": "Based on the current market data...",
    "tool_calls": [...],  # What tools were called
    "tool_results": [...],  # Tool outputs
    "token_usage": {
        "prompt_tokens": 1500,
        "completion_tokens": 200,
        "total_tokens": 1700
    }
}
```

### Step 14: Save to Database
**Back in API endpoint:**
```python
assistant_db_msg = DBChatMessage(
    session_id=session_id,
    role="assistant",
    content=response["content"],
    tool_calls=response["tool_calls"],  # Stored as JSON
    tool_results=response["tool_results"],  # Stored as JSON
    token_usage=response["token_usage"],  # Stored as JSON
)
db.add(assistant_db_msg)
db.commit()
```

### Step 15: Return to Frontend
```python
return ChatResponse(
    message=assistant_msg,
    session_id=session_id,
    token_usage=response["token_usage"],
)
```

## Key Components

### 1. ChatOrchestrator Class
- **Purpose:** Coordinates LLM and tools
- **Key Methods:**
  - `process_message()` - Main entry point
  - `_execute_tools()` - Executes tool calls
  - `_register_tools()` - Loads available tools

### 2. ToolRegistry
- **Purpose:** Manages available tools
- **Key Methods:**
  - `register()` - Add a tool
  - `execute_tool()` - Run a tool
  - `get_function_definitions()` - Get OpenAI format

### 3. Tools
- **Data Tools:** `get_symbol_data`, `get_bars`
- **Analysis Tools:** `analyze_trend`, `calculate_indicators`
- **All tools:** Connect to Trading Copilot Platform, return real data

## Important Invariants Enforced

1. **INV-LLM-01:** No data fabrication
   - All data comes from tools
   - LLM cannot make up prices/numbers

2. **INV-LLM-02:** All analytics from tools
   - LLM must call tools before answering
   - Cannot provide market data without tools

3. **INV-SAFE-02:** Education-only
   - System prompt enforces this
   - Responses include disclaimers

4. **INV-LLM-03:** Token budget tracking
   - Token usage tracked per session
   - Stored in database

## Example: "What is SPY's current price?"

1. User: "What is SPY's current price?"
2. Orchestrator builds context with system prompt
3. First LLM call: LLM decides to call `get_symbol_data(symbol="SPY")`
4. Tool executes: Fetches real data from Trading Platform
5. Tool returns: `{"symbol": "SPY", "price": 485.23, ...}`
6. Second LLM call: LLM generates response using tool data
7. Response: "The current price of SPY is $485.23..."
8. Saved to database with tool calls and results
9. Returned to frontend

## Two-Call Pattern

**Why two LLM calls?**

1. **First call:** LLM decides what tools to use
2. **Tool execution:** Get real data
3. **Second call:** LLM generates answer using real data

This ensures:
- LLM never fabricates data
- All numbers come from tools
- Responses are based on real market data

## Error Handling

- **Tool execution fails:** Error message added to results
- **LLM call fails:** Exception raised, caught by API
- **Database save fails:** Transaction rolled back

## Token Tracking

Tokens are tracked for:
- Cost monitoring (INV-LLM-03)
- Budget enforcement
- Audit logging

Both LLM calls are counted when tools are used.
