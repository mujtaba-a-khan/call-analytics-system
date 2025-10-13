---
name: Bug report
about: Create a report to help me improve Call-Analytics-System"
title: "[BUG] <short summary>"
labels: bug
assignees: mujtaba-a-khan

---

**Describe the bug**  
A clear and concise description of what the bug is.

**To Reproduce**  
Steps to reproduce the behavior (be specific):
1. …
2. …
3. …
4. See error

**Expected behavior**  
A clear and concise description of what you expected to happen.

**How did you run the app?** (check all that apply)
- [ ] Streamlit (`streamlit run app.py`)
- [ ] CLI script
- [ ] Docker
- [ ] Jenkins CI
- [ ] Other: …

**Environment**
- OS: macOS / Linux / Windows (version: ___)
- Python: `python --version` → ___
- Virtual env: venv / conda / system
- Installation: `pip install -e .[ci]` / `pip install -r requirements.txt` / other
- FFmpeg: `ffmpeg -version` → ___
- GPU: model ___  • CUDA ___  • `nvidia-smi` ok? yes/no
- Torch: `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"` → ___
- Streamlit: `streamlit version` → ___

**Project-specific details**
- Audio input: sample rate ___ kHz • file type ___ • duration ___
- STT backend & model: (e.g., Whisper tiny/base/small/medium/large) → ___
- Embedding / Vector DB: (e.g., SentenceTransformers ____, ChromaDB/FAISS/Pinecone) → ___
- LLM/provider & model: (e.g., Ollama/OpenAI/Watsonx) → ___
- Config in use: path(s) under `configs/` (attach sanitized snippets)
- Running commit SHA: `git rev-parse --short HEAD` → ___
- Jenkins/Docker (if used): link to failing build / image tag → ___

**Relevant config (sanitized)**  
Paste only the parts of `.env` and `configs/*.yaml` that matter. **Remove keys/tokens.**

```yaml
# example
OPENAI_API_KEY: "***redacted***"
model: "whisper-small"
```

**Logs / Traceback**
```
<error output here>
```

**Screenshots / recordings**  
If applicable, add screenshots or a short screen recording.

**Minimal audio/sample (optional but very helpful)**  
Link a 10–30s audio file that reproduces the issue (remove sensitive data).

**Additional context**  
Add any other context about the problem here.

**Checklist**
- [ ] I searched existing issues/discussions.
- [ ] I updated to the latest `main` or release and can still reproduce.
- [ ] I can reproduce with a minimal example (small audio + default config).
- [ ] Secrets/API keys are **not** included in this report.
