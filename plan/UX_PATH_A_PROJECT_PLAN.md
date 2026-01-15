# UX Path A Project Plan — Web Chat App (Smart Trading Copilot)

## Executive Summary

UX Path A is a standalone, ChatGPT-style smart web application with a custom UI that provides conversational market analysis and strategy reasoning. This document defines the execution plan, architecture, and implementation checklist.

## 1. Project Definition

### 1.1 Core Characteristics
- **Type**: Standalone web application
- **UI Style**: ChatGPT-style conversational interface
- **LLM Access**: OpenAI APIs
- **Backend**: Python FastAPI (integrates with existing Trading Copilot Platform)
- **Frontend**: React/Next.js
- **Purpose**: Power-user analysis, correctness, extensibility

### 1.2 Non-Goals (Explicit)
- ❌ Execute trades (INV-SAFE-01)
- ❌ Store broker credentials (INV-SAFE-01)
- ❌ Provide personalized investment advice (INV-SAFE-02)
- ❌ Guarantee outcomes or profits (INV-STRAT-03)
- ❌ Embed strategy logic in UI or prompts (INV-ARCH-01)

## 2. Architectural Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (User)                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Web Chat UI (React/Next.js)                         │
│  - Conversational Interface                                 │
│  - Message History                                          │
│  - Chart Embeddings                                         │
│  - Auth UI                                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/WebSocket
                            ↓
┌─────────────────────────────────────────────────────────────┐
│      App Backend (FastAPI - Chat Orchestrator)              │
│  - Session Management                                       │
│  - LLM Orchestration                                        │
│  - Tool Routing                                             │
│  - State Management                                         │
│  - Audit Logging                                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              OpenAI API (LLM Reasoning)                    │
│  - GPT-4/GPT-4 Turbo                                        │
│  - Function Calling                                         │
│  - System Prompts                                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│    Trading Copilot Platform (Internal Tool Calls)           │
│  - Data Engine (Market Data)                                │
│  - Strategy Analysis                                        │
│  - Backtesting Engine                                       │
│  - Portfolio Analysis                                       │
│  - Indicator Calculations                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Architectural Rules

1. **UI is Stateless** (INV-STATE-01)
   - UI does not contain financial logic
   - All calculations happen in backend
   - UI only displays tool outputs

2. **Tool Outputs are Authoritative** (INV-LLM-02)
   - All numeric outputs originate from tools
   - LLM cannot fabricate prices or indicators (INV-LLM-01)
   - Tool outputs are immutable context

3. **Backend is Single Source of Truth**
   - All analytics from platform tools
   - No logic duplication in UI
   - Platform invariants enforced server-side

4. **Modular Design** (INV-ARCH-02)
   - UI can be replaced without logic changes
   - Clear separation of concerns
   - Tool layer is independent

## 3. Functional Scope (MVP)

### 3.1 Core Capabilities

| # | Capability | Description | Priority |
|---|-----------|------------|----------|
| 1 | Symbol Analysis | Analyze any symbol with price, volume, indicators | P0 |
| 2 | Trend Regime | Detect and explain trend regimes (UP/DOWN/RANGE) | P0 |
| 3 | Volatility Regime | Identify volatility conditions | P0 |
| 4 | Directional Bias | Provide directional market bias | P0 |
| 5 | Confidence Score | Quantify analysis confidence | P0 |
| 6 | Explainability | Natural language explanations | P0 |
| 7 | Indicator-level Reasoning | Explain individual indicators | P0 |
| 8 | Regime Interpretation | Interpret market regimes | P0 |
| 9 | Scenario Analysis | Conservative/Base/Aggressive scenarios | P1 |
| 10 | Risk Framing | Explicit risk disclosure | P0 |
| 11 | Conversational Iteration | Follow-up questions with context | P0 |
| 12 | Backtesting | Historical strategy evaluation | P1 |
| 13 | Portfolio CSV Import | Import portfolio for analysis | P2 |
| 14 | Exposure Analysis | Read-only portfolio risk analysis | P2 |

### 3.2 LLM Orchestration Model

**LLM Responsibilities:**
- Interpret user intent
- Decide which tools to call
- Generate human-readable explanations
- Maintain conversation context

**Prohibited LLM Behaviors:**
- ❌ Fabricating prices or indicators (INV-LLM-01)
- ❌ Overriding tool outputs
- ❌ Making execution recommendations
- ❌ Providing personalized advice (INV-SAFE-02)

**Orchestration Strategy:**
- Server-side orchestration for determinism
- Tool outputs injected as immutable context
- Function calling for tool selection
- System prompts enforce invariants

## 4. Implementation Phases

### Phase A1: Skeleton (Foundation)
**Goal**: Basic chat interface with LLM integration

**Components:**
- [ ] Next.js project setup
- [ ] Basic chat UI (message list, input)
- [ ] FastAPI backend setup
- [ ] OpenAI API integration
- [ ] Basic LLM call loop
- [ ] Session management
- [ ] Auth system (basic)
- [ ] Paper portfolio management via chat

**Deliverables:**
- Working chat interface
- Can send messages and receive LLM responses
- Session persistence
- Basic auth

### Phase A2: Tool Integration
**Goal**: Connect to Trading Copilot Platform tools

**Components:**
- [ ] Tool registry system
- [ ] Data source tools (get_bars, get_indicators)
- [ ] Chart generation tools
- [ ] Strategy analysis tools
- [ ] Indicator calculation tools
- [ ] Simulated trades tool
- [ ] Backtesting tool integration
- [ ] Chart display in UI
- [ ] Tool output formatting

**Deliverables:**
- All core capabilities functional
- Charts embedded in chat
- Tool outputs properly formatted
- LLM can call tools and explain results

### Phase A3: Guardrails & Caching
**Goal**: Production-ready safety and performance

**Components:**
- [ ] Token budget enforcement (INV-LLM-03)
- [ ] Rate limiting
- [ ] Caching layer (daily regimes)
- [ ] Feature gating (INV-CROSS-01)
- [ ] High-volatility warnings
- [ ] Leverage warnings
- [ ] Error handling (INV-ERR-01)
- [ ] Partial results handling

**Deliverables:**
- Cost controls active
- Safety warnings displayed
- Caching reduces API calls
- Graceful error handling

### Phase A4: Backtest & Portfolio (Optional)
**Goal**: Advanced features

**Components:**
- [ ] CSV portfolio import
- [ ] Portfolio analysis tools
- [ ] Exposure calculation
- [ ] Risk analysis
- [ ] Backtest comparison views

**Deliverables:**
- Portfolio import working
- Risk analysis displayed
- Backtest comparisons available

## 5. Technical Stack

### 5.1 Frontend
- **Framework**: Next.js 14+ (App Router)
- **UI Library**: React 18+
- **Styling**: Tailwind CSS
- **State Management**: React Context / Zustand
- **Charts**: Plotly.js (matches backend)
- **HTTP Client**: Fetch API / Axios
- **WebSocket**: For real-time updates (optional)

### 5.2 Backend
- **Framework**: FastAPI
- **Python Version**: 3.10+
- **LLM Client**: OpenAI Python SDK
- **Database**: SQLite (sessions, audit logs) or PostgreSQL
- **Caching**: Redis (optional) or in-memory
- **Auth**: JWT tokens
- **Validation**: Pydantic models

### 5.3 Integration
- **Platform Tools**: Existing `core/` modules
- **Data Engine**: `core/data/`
- **Strategies**: `core/strategy/`
- **Backtesting**: `core/backtest/`
- **Visualization**: `core/visualization/`

## 6. System Prompt Strategy

### 6.1 Requirements
- Single, versioned system prompt
- Stored in backend, not frontend
- References platform invariants implicitly
- Enforces education-only framing (INV-SAFE-02)
- Mandates tool usage before conclusions (INV-LLM-02)
- Includes risk disclosure language (INV-SAFE-03)

### 6.2 Prompt Structure
```
You are a financial analysis assistant for educational purposes only.

RULES:
1. You MUST use tools to get all market data - never fabricate numbers
2. All analysis must be educational, not personalized advice
3. Always include risk disclosures
4. Explain your reasoning using tool outputs
5. If data is unavailable, say so clearly

TOOLS AVAILABLE:
- get_symbol_data: Get price, volume, indicators
- analyze_trend: Detect trend regimes
- calculate_indicators: Compute technical indicators
- run_backtest: Evaluate strategies historically
- analyze_portfolio: Portfolio risk analysis

Remember: This is for educational purposes only. Not financial advice.
```

## 7. State & Memory Model

### 7.1 Session State
- Conversation ID (UUID)
- User ID
- Explicit user inputs (messages)
- Referenced symbols/timeframes
- Referenced strategies
- Tool call history
- Timestamps

### 7.2 Persistence Rules
- No hidden memory (INV-STATE-02)
- All cached analytics keyed per INV-STATE-03
- Session data stored in database
- Audit logs for all interactions

### 7.3 Context Management
- Follow-up questions reuse context explicitly (INV-STATE-02)
- Conversation history maintained per session
- Tool outputs cached per symbol/timeframe
- Regime data cached daily

## 8. Guardrails & Limits

### 8.1 Cost Controls
- Per-session LLM token budgets (INV-LLM-03)
- Rate limiting per user
- Cached daily regimes reused
- Tool call limits

### 8.2 Safety Controls
- Feature gating enforced server-side (INV-CROSS-01)
- High-volatility warnings mandatory
- Leverage warnings mandatory
- Education-only disclaimers

### 8.3 Failure Handling
- Fail closed on missing data (INV-ERR-01)
- Partial results allowed with explanation
- Graceful degradation
- Clear error messages

## 9. Auditability

### 9.1 Required Logs
For every interaction, backend must log:
- User/session ID
- Prompt version
- Tools called
- Tool inputs and outputs
- Timestamps
- Strategy versions
- LLM responses

### 9.2 Compliance
- Satisfies INV-AUDIT-01 (audit trail)
- Satisfies INV-AUDIT-02 (reproducibility)
- All logs stored securely
- Queryable audit system

## 10. Acceptance Criteria

UX Path A is considered complete when:

- ✅ All user-visible analytics originate from tools
- ✅ Same inputs yield same outputs (INV-DATA-01)
- ✅ UI can be replaced without logic changes (INV-ARCH-02)
- ✅ No invariant violations observed
- ✅ All MVP capabilities functional
- ✅ Guardrails active
- ✅ Audit logging working
- ✅ Documentation complete

## 11. File Structure

```
ux_path_a/
├── frontend/                 # Next.js application
│   ├── app/                 # Next.js App Router
│   │   ├── (auth)/          # Auth pages
│   │   ├── chat/           # Chat interface
│   │   └── api/             # API routes (proxies)
│   ├── components/          # React components
│   │   ├── chat/           # Chat components
│   │   ├── charts/         # Chart components
│   │   └── ui/             # UI components
│   ├── lib/                 # Utilities
│   ├── hooks/              # React hooks
│   └── types/              # TypeScript types
│
├── backend/                 # FastAPI application
│   ├── api/                 # API routes
│   │   ├── chat.py         # Chat endpoints
│   │   ├── auth.py         # Auth endpoints
│   │   └── tools.py        # Tool endpoints
│   ├── core/                # Core logic
│   │   ├── orchestrator.py # LLM orchestrator
│   │   ├── tools/          # Tool implementations
│   │   ├── prompts/        # System prompts
│   │   └── session.py      # Session management
│   ├── models/             # Pydantic models
│   ├── db/                 # Database models
│   └── utils/              # Utilities
│
├── shared/                  # Shared types/constants
│   └── types.py            # TypeScript/Python types
│
└── tests/                   # Tests
    ├── frontend/
    └── backend/
```

## 12. Next Steps

1. Create project structure
2. Set up Next.js frontend
3. Set up FastAPI backend
4. Implement Phase A1 components
5. Integrate with existing platform
6. Implement tool system
7. Add guardrails
8. Testing and refinement
