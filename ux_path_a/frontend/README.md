# UX Path A Frontend

Next.js frontend for the Smart Trading Copilot web chat application.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment:**
   Create `.env.local`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```

Frontend will run on `http://localhost:3000`

## Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/              # Next.js App Router
│   ├── page.tsx      # Main page
│   └── layout.tsx    # Root layout
├── components/       # React components
│   ├── chat/        # Chat UI components
│   └── auth/        # Authentication UI
├── lib/             # Utilities
│   └── api.ts       # API client
├── hooks/           # React hooks
│   └── useAuth.ts   # Authentication hook
└── public/          # Static assets
```

## Features

- ChatGPT-style conversational interface
- Session management
- Message history
- Authentication
- Responsive design with dark mode support
- Markdown rendering for LLM responses
