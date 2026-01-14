# Local Development Setup

## Quick Start

The backend uses **absolute imports** (`from ux_path_a.backend.*`) which require the project root to be in `PYTHONPATH`.

### Easy Way (Recommended)

**Windows:**
```powershell
cd ux_path_a/backend
.\run_local.ps1
```

**Linux/Mac:**
```bash
cd ux_path_a/backend
chmod +x run_local.sh
./run_local.sh
```

These scripts automatically set `PYTHONPATH` and start the server.

### Manual Way

**Windows PowerShell:**
```powershell
# From project root
cd C:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
cd ux_path_a/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Linux/Mac:**
```bash
# From project root
cd /path/to/LLM-Investment-Copilot
export PYTHONPATH=$PWD:$PYTHONPATH
cd ux_path_a/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Why This Is Needed

The backend uses absolute imports like:
```python
from ux_path_a.backend.api import chat
from ux_path_a.backend.backend_core.models import ChatSession
```

For Python to find `ux_path_a`, the project root must be in `PYTHONPATH`. This is:
- ✅ **Automatic in Railway** (Dockerfile sets `PYTHONPATH=/app`)
- ⚠️ **Manual in local dev** (hence the scripts above)

## Alternative: Run from Project Root

You can also run from the project root using Python's `-m` flag:

```bash
# From project root
python -m uvicorn ux_path_a.backend.main:app --reload --host 0.0.0.0 --port 8000
```

This works because Python automatically adds the current directory to `sys.path` when using `-m`.

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'ux_path_a'`**
- Solution: Make sure `PYTHONPATH` includes the project root
- Use the provided scripts (`run_local.ps1` or `run_local.sh`)
- Or set `PYTHONPATH` manually as shown above

**Error: `Multiple classes found for path "ChatMessage"`**
- Solution: Clear Python cache: `Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force`
- Restart the server
