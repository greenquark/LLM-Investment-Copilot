# UX Path A Quick Start Guide

## Overview

UX Path A is now fully implemented with:
- ✅ Database integration (SQLAlchemy models, migrations)
- ✅ Tool implementations (data sources, analysis)
- ✅ Frontend (Next.js with chat UI)
- ✅ Guardrails (token budgets, safety controls)
- ✅ End-to-end tests

## Setup Instructions

### 1. Backend Setup

```bash
cd ux_path_a/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
# Set OPENAI_API_KEY=your-key-here

# Run database migrations (optional - tables auto-create on startup)
# alembic upgrade head

# Start server
uvicorn main:app --reload
```

Backend will run on `http://localhost:8000`

### 2. Frontend Setup

```bash
cd ux_path_a/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on `http://localhost:3000`

### 3. Testing

```bash
cd ux_path_a/backend
pytest ../tests/test_end_to_end.py -v
```

## Architecture

```
Frontend (Next.js) → Backend (FastAPI) → OpenAI API → Trading Copilot Platform Tools
```

### Key Components

**Backend:**
- `main.py` - FastAPI application
- `core/orchestrator.py` - LLM orchestration
- `core/tools/` - Tool implementations
- `core/guardrails.py` - Safety and cost controls
- `core/models.py` - Database models
- `api/chat.py` - Chat endpoints
- `api/auth.py` - Authentication

**Frontend:**
- `app/page.tsx` - Main page
- `components/chat/` - Chat UI components
- `components/auth/` - Authentication UI
- `lib/api.ts` - API client
- `hooks/useAuth.ts` - Auth hook

## Features Implemented

### ✅ Core Capabilities
- Symbol analysis via `get_symbol_data` tool
- Trend analysis via `analyze_trend` tool
- Technical indicators via `calculate_indicators` tool
- Historical data via `get_bars` tool

### ✅ Safety & Compliance
- Token budget tracking (INV-LLM-03)
- Audit logging (INV-AUDIT-01, INV-AUDIT-02)
- Safety warnings (volatility, leverage)
- Risk disclosures
- Education-only framing

### ✅ User Experience
- ChatGPT-style chat interface
- Session management
- Message history
- Authentication
- Responsive design

## Usage Example

1. **Start backend and frontend**
2. **Open browser** to `http://localhost:3000`
3. **Login** (any credentials work for MVP)
4. **Start chatting:**
   - "What is the current price of AAPL?"
   - "Analyze the trend for TSLA"
   - "Calculate RSI for SPY"

## Tool Examples

The LLM can call these tools automatically:

- `get_symbol_data(symbol="AAPL")` - Get current market data
- `get_bars(symbol="AAPL", start_date="2024-01-01")` - Get historical bars
- `analyze_trend(symbol="TSLA")` - Analyze trend regime
- `calculate_indicators(symbol="SPY", indicators=["RSI", "MA"])` - Calculate indicators

## Database Schema

- **users** - User accounts
- **chat_sessions** - Chat sessions
- **chat_messages** - Message history
- **audit_logs** - Audit trail (INV-AUDIT-01)
- **token_budgets** - Token usage tracking (INV-LLM-03)

## Configuration

Edit `ux_path_a/backend/.env`:
```env
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7
MAX_TOKENS_PER_SESSION=100000
DATABASE_URL=sqlite:///./ux_path_a.db
```

**Note:** All settings can be configured via environment variables. The `.env` file is automatically loaded. See `ux_path_a/backend/.env.example` for all available options.

## Next Steps

1. Add chart generation tool
2. Add backtesting tool
3. Implement caching
4. Add rate limiting
5. Enhance error handling
6. Add portfolio import (Phase A4)

## Troubleshooting

**Backend won't start:**
- Check OpenAI API key is set
- Check database file permissions
- Check port 8000 is available

**Frontend won't start:**
- Run `npm install` first
- Check Node.js version (18+)
- Check port 3000 is available

**Tools not working:**
- Ensure `config/env.backtest.yaml` exists
- Check MarketData.app API token is configured
- Check symbol is valid

## Architecture Compliance

All platform invariants are enforced:
- ✅ INV-LLM-01: No data fabrication
- ✅ INV-LLM-02: All data from tools
- ✅ INV-LLM-03: Token budgets enforced
- ✅ INV-SAFE-01: No trade execution
- ✅ INV-SAFE-02: Education-only
- ✅ INV-SAFE-03: Risk disclosures
- ✅ INV-AUDIT-01: Audit logging
- ✅ INV-AUDIT-02: Reproducibility
- ✅ INV-STATE-01: UI is stateless
- ✅ INV-ARCH-01: No logic in UI
