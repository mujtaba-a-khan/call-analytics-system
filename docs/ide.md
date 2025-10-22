# IDE Setup - Visual Studio Code

## Overview

This document describes the VS Code setup. The project was developed using Visual Studio Code with Python 3.11 on macOS. This guide provides the configuration for Python development, testing, and debugging.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Required Extensions](#required-extensions)
- [Recommended Extensions](#recommended-extensions)
- [Workspace Configuration](#workspace-configuration)
- [Debugging Configuration](#debugging-configuration)
- [Testing Setup](#testing-setup)
- [Git Integration](#git-integration)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)
- [Development Best Practices](#development-best-practices)
- [My Development Workflow](#my-development-workflow)
- [Summary](#summary)

---

## Prerequisites

Before setting up VS Code, ensure you have:

- **Visual Studio Code** 1.80 or higher
- **Python 3.11** or higher
- **Git** for version control
- **FFmpeg** for audio processing

**Verify installations** (Terminal/zsh):

```bash
code --version
python --version  # Should show 3.11.x or higher
git --version
ffmpeg -version
```

*Derived from `README.md`, lines 33-89 (Quick Start)*

---

## Initial Setup

### 1. Open Project

```bash
cd call-analytics-system
code .
```

### 2. Trust Workspace

When prompted, click **"Trust"** to enable full VS Code features.

### 3. Create Virtual Environment

*From: `README.md`, lines 69-80*

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS Terminal
```

### 4. Install Dependencies

*From: `pyproject.toml`, lines 74-86*

```bash
# Install main dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with documentation tools
pip install -e ".[docs]"
```

### 5. Select Python Interpreter

1. Press `Cmd+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python`

---

## Required Extensions

### Python Development

**1. Python** (Microsoft)
- Extension ID: `ms-python.python`
- Provides IntelliSense, linting, debugging

```bash
code --install-extension ms-python.python
```

**2. Pylance** (Microsoft)
- Extension ID: `ms-python.vscode-pylance`
- Fast language server with type checking

```bash
code --install-extension ms-python.vscode-pylance
```

**3. Python Debugger** (Microsoft)
- Extension ID: `ms-python.debugpy`
- Debugging support

```bash
code --install-extension ms-python.debugpy
```

### Code Quality

**4. Ruff** (Astral Software)
- Extension ID: `charliermarsh.ruff`
- Fast Python linter and formatter

*From: `pyproject.toml`, lines 144-173*

```bash
code --install-extension charliermarsh.ruff
```

**5. Black Formatter** (Microsoft)
- Extension ID: `ms-python.black-formatter`
- Python code formatter

*From: `pyproject.toml`, lines 130-143*

```bash
code --install-extension ms-python.black-formatter
```

**6. Mypy Type Checker** (Microsoft)
- Extension ID: `ms-python.mypy-type-checker`
- Static type checking

*From: `pyproject.toml`, lines 175-221*

```bash
code --install-extension ms-python.mypy-type-checker
```

### Install All Required Extensions

```bash
code --install-extension ms-python.python && \
code --install-extension ms-python.vscode-pylance && \
code --install-extension ms-python.debugpy && \
code --install-extension charliermarsh.ruff && \
code --install-extension ms-python.black-formatter && \
code --install-extension ms-python.mypy-type-checker
```

---

## Recommended Extensions

### General Development

**1. GitLens** (GitKraken)
- Extension ID: `eamodio.gitlens`
- Enhanced Git capabilities

```bash
code --install-extension eamodio.gitlens
```

**2. Git Graph** (mhutchie)
- Extension ID: `mhutchie.git-graph`
- Visualize repository history

```bash
code --install-extension mhutchie.git-graph
```

**3. Todo Tree** (Gruntfuggly)
- Extension ID: `gruntfuggly.todo-tree`
- Track TODO, FIXME comments

```bash
code --install-extension gruntfuggly.todo-tree
```

### File Support

**4. TOML Language Support** (be5invis)
- Extension ID: `be5invis.toml`
- Syntax highlighting for config files

*Project uses TOML extensively: `config/*.toml`, `pyproject.toml`*

```bash
code --install-extension be5invis.toml
```

**5. Markdown All in One** (Yu Zhang)
- Extension ID: `yzhang.markdown-all-in-one`
- Enhanced markdown editing

*For documentation files: `docs/*.md`, `README.md`*

```bash
code --install-extension yzhang.markdown-all-in-one
```

**6. Rainbow CSV** (mechatroner)
- Extension ID: `mechatroner.rainbow-csv`
- CSV file visualization

*Useful for viewing test data and call records*

```bash
code --install-extension mechatroner.rainbow-csv
```

### Documentation

**7. autoDocstring** (Nils Werner)
- Extension ID: `njpwerner.autodocstring`
- Generate Python docstrings

```bash
code --install-extension njpwerner.autodocstring
```

---

## Workspace Configuration

### Create `.vscode/settings.json`

These settings match the project's configuration in `pyproject.toml`:

```json
{
  // Python Configuration
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  
  // Formatting
  "editor.formatOnSave": true,
  "editor.formatOnPaste": false,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  
  // Black Configuration (from pyproject.toml lines 130-143)
  "black-formatter.args": [
    "--line-length=100"
  ],
  
  // Ruff Configuration (from pyproject.toml lines 144-173)
  "ruff.lint.args": [
    "--line-length=100"
  ],
  "ruff.organizeImports": true,
  
  // Mypy Configuration (from pyproject.toml lines 175-221)
  "mypy-type-checker.args": [
    "--python-version=3.11",
    "--warn-return-any",
    "--warn-unused-configs"
  ],
  
  // Testing (from pyproject.toml lines 222-243)
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests",
    "-ra",
    "--strict-markers",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--tb=short",
    "--maxfail=1"
  ],
  
  // File Associations
  "files.associations": {
    "*.toml": "toml",
    "Jenkinsfile": "groovy"
  },
  
  // File Exclusions
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true,
    "**/*.egg-info": true
  },
  
  // Editor Settings
  "editor.rulers": [100],
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  
  // Git
  "git.autofetch": true,
  "git.confirmSync": false
}
```

### Create `.vscode/launch.json`

Debug configurations for the project:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Streamlit: Launch UI",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": [
        "run",
        "src/ui/app.py",
        "--server.port=8501",
        "--server.headless=true"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "CLI: Main Command",
      "type": "debugpy",
      "request": "launch",
      "module": "src.cli",
      "args": ["--help"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Script: Setup Environment",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/setup_environment.py",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Script: Rebuild Index",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/rebuild_index.py",
      "args": ["--help"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Pytest: Current Test",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v",
        "-s"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

### Create `.vscode/tasks.json`

Suggested VS Code tasks that mirror common project scripts:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Streamlit UI",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "streamlit", "run", "src/ui/app.py"],
      "problemMatcher": [],
      "group": {
        "kind": "build",
        "isDefault": false
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "pytest", "tests/", "-v"],
      "problemMatcher": [],
      "group": {
        "kind": "test",
        "isDefault": true
      }
    },
    {
      "label": "Lint with Ruff",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "ruff", "check", "src/", "scripts/"],
      "problemMatcher": [],
      "group": "none"
    },
    {
      "label": "Format with Black",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "black", "src/", "scripts/"],
      "problemMatcher": [],
      "group": "none"
    },
    {
      "label": "Type Check with Mypy",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "mypy", "src/"],
      "problemMatcher": [],
      "group": "none"
    },
    {
      "label": "Setup Environment",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["scripts/setup_environment.py"],
      "problemMatcher": [],
      "group": "none"
    },
    {
      "label": "Download Models",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["scripts/download_models.py"],
      "problemMatcher": [],
      "group": "none"
    },
    {
      "label": "Rebuild Index",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["scripts/rebuild_index.py"],
      "problemMatcher": [],
      "group": "none"
    }
  ]
}
```

---

## Debugging Configuration

### Using Breakpoints

1. **Set Breakpoint**: Click left gutter or press `F9`
2. **Conditional Breakpoint**: Right-click → Edit Breakpoint
3. **Start Debugging**: Press `F5` or click Run button

### Launch Configurations

**Debug the Streamlit UI**:
- Select "Streamlit: Launch UI" in Run and Debug panel
- Press `F5`
- App launches at http://localhost:8501

**Debug CLI Commands** (*from `src/cli.py`*):
- Select "CLI: Main Command"
- Modify `args` in `launch.json` for specific commands
- Example: `["ui", "--port", "8502"]`

**Debug Scripts** (*from `scripts/` directory*):
- Select appropriate script configuration
- Breakpoints work in setup, download, or rebuild scripts

**Debug Tests** (*from `tests/` directory*):
- Select "Pytest: Current Test"
- Opens current test file for debugging

---

## Testing Setup

### Configuration

*From: `pyproject.toml`, lines 222-243*

Tests are configured with pytest and coverage:
- Test directory: `tests/`
- Coverage target: `src/`
- Output formats: HTML, XML, terminal

### Running Tests in VS Code

**Method 1: Test Explorer**

1. Click **Testing** icon in Activity Bar (beaker icon)
2. Tests auto-discover from `tests/` directory
3. Click play button to run individual or all tests
4. View results inline

**Method 2: Terminal**

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_aggregations.py -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

**Method 3: Tasks**

- Press `Cmd+Shift+P`
- Type "Tasks: Run Task"
- Select "Run Tests"

### Test Files in Repository

*From workspace structure:*
- `tests/test_aggregations.py` - Data aggregation tests
- `tests/test_text_processing.py` - Text utility tests

---

## Git Integration

### Built-in Git Features

VS Code has excellent Git support out of the box:

**Source Control Panel** (`Cmd+Shift+G`):
- View changes
- Stage/unstage files
- Commit with messages
- Push/pull/sync

**GitLens Extension**:
- Inline blame annotations
- File/line history
- Repository explorer
- Comparison views

### Common Git Workflows

**Creating Feature Branch**:
```bash
# In terminal
git checkout -b feature/new-feature
```

**Committing Changes**:
1. Open Source Control panel (`Cmd+Shift+G`)
2. Stage files by clicking `+`
3. Enter commit message
4. Click checkmark or press `Cmd+Enter`

**Viewing History**:
- Click Git Graph in status bar
- Right-click files → "Open Timeline"
- Use GitLens hovers for inline history

---

## Keyboard Shortcuts

Common keyboard shortcuts used during development.

### Essential Navigation

| Action              | Shortcut          |
| ------------------- | ----------------- |
| Command Palette     | `Cmd+Shift+P`    |
| Quick Open File     | `Cmd+P`          |
| Go to Definition    | `F12`             |
| Go Back             | `Cmd+Option+Left`        |
| Peek Definition     | `Option+F12`         |
| Find References     | `Shift+F12`       |

### Editing

| Action              | Shortcut          |
| ------------------- | ----------------- |
| Multi-Cursor        | `Option+Cmd+Down/Up`|
| Select Next Occurrence | `Cmd+D`       |
| Move Line Up/Down   | `Option+Up/Down`     |
| Copy Line           | `Shift+Option+Up/Down` |
| Delete Line         | `Cmd+Shift+K`    |
| Comment Line        | `Cmd+/`          |
| Format Document     | `Shift+Option+F`     |
| Rename Symbol       | `F2`              |

### Search & Terminal

| Action              | Shortcut          |
| ------------------- | ----------------- |
| Find in Files       | `Cmd+Shift+F`    |
| Replace in Files    | `Cmd+Shift+H`    |
| Toggle Terminal     | `Cmd+J`      |
| Toggle Sidebar      | `Cmd+B`          |
| Source Control      | `Cmd+Shift+G`    |

### Debugging

| Action              | Shortcut          |
| ------------------- | ----------------- |
| Start Debugging     | `F5`              |
| Toggle Breakpoint   | `F9`              |
| Step Over           | `F10`             |
| Step Into           | `F11`             |
| Step Out            | `Shift+F11`       |
| Stop Debugging      | `Shift+F5`        |

---

## Troubleshooting

### Python Interpreter Not Found

**Problem**: VS Code can't find virtual environment

**Solution**:

```bash
# Delete and recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Reload VS Code
# Cmd+Shift+P → "Developer: Reload Window"
# Select interpreter: Cmd+Shift+P → "Python: Select Interpreter"
```

### Extensions Not Working

**Problem**: Ruff/Black/Mypy not running

**Solution**:

1. Verify extensions installed: View → Extensions
2. Check `.vscode/settings.json` exists
3. Reload window: `Cmd+Shift+P` → "Developer: Reload Window"
4. Check Output panel: View → Output → Select extension

### Tests Not Discovered

**Problem**: Test Explorer shows no tests

**Solution**:

```bash
# Verify pytest installed
pip show pytest

# Check test file naming (must be test_*.py or *_test.py)
# Check test function naming (must start with test_)

# Clear cache and reload
# Cmd+Shift+P → "Python: Clear Cache and Reload Window"
```

### Formatting Not Working on Save

**Problem**: Black doesn't format on save

**Solution**:

Verify settings in `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "python.testing.pytestArgs": [
    "tests",
    "-ra",
    "--strict-markers",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--tb=short",
    "--maxfail=1"
  ]
}
```

## Development Best Practices

Key practices followed during development:

1. **Format on Save**:
   - Black automatically formats code on save
   - Maintains consistent code style
   - Line length set to 100 characters

2. **Linting**:
   - Ruff provides real-time feedback
   - Catches common errors and code smells
   - Configured for Python 3.11 standards

3. **Type Checking**:
   - Mypy ensures type safety
   - Helps catch bugs before runtime
   - Improves code documentation

4. **Testing**:
   - Pytest integrated in Test Explorer
   - Run tests frequently during development
   - Coverage reports help identify untested code

5. **Git Integration**:
   - GitLens shows inline blame and history
   - Frequent small commits with clear messages
   - Review changes before committing

6. **Debugging**:
   - Use breakpoints instead of print statements
   - Step through code to understand flow
   - Inspect variables in Debug Console
   
---
## My Development Workflow

Here's how I typically worked on this project:

### Daily Development Loop

1. **Morning Setup**:
   ```bash
   # Open project
   cd call-analytics-system
   
   # Activate environment
   source .venv/bin/activate
   ```

2. **Making Changes**:
   - Use `Cmd+P` to quickly open files
   - Edit code with auto-complete and type hints
   - Format on save handles Black formatting
   - Ruff catches issues as I type

3. **Testing Changes**:
   - Launch Streamlit UI via `Cmd+Shift+P` → "Tasks: Run Task" → "Run Streamlit UI"
   - Test features in browser
   - Use `F12` to jump to definitions when debugging
   - Add breakpoints with `F9` and debug with `F5`

4. **Before Committing**:
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Or use Cmd+Shift+P → "Tasks: Run Task" → "Run Tests"
   ```

5. **Committing**:
   - `Cmd+Shift+G` to open Source Control
   - Review changes (click files to see diffs)
   - Stage files with `+` icon
   - Write commit message
   - `Cmd+Enter` to commit

### Shortcuts I Used Most

Based on my actual usage, these are the top shortcuts that saved me the most time:

**Top 10 Most Used**:
1. `Cmd+P` - Quick file open (used hundreds of times daily)
2. `Cmd+Shift+P` - Command palette (second most used)
3. `F12` - Go to definition (essential for navigating code)
4. `Cmd+D` - Select next occurrence (for quick edits)
5. `Cmd+/` - Toggle comment (when testing code)
6. `Cmd+J` - Toggle terminal (constantly switching)
7. `Option+Up/Down` - Move lines (for code organization)
8. `Cmd+Shift+F` - Search in files (finding patterns)
9. `F5` - Start debugging (troubleshooting issues)
10. `Cmd+Shift+G` - Git panel (committing frequently)

## Summary

With this setup complete, you have:

- ✅ Automatic code formatting with Black (line length 100)
- ✅ Linting with Ruff configured for Python 3.11
- ✅ Type checking with Mypy
- ✅ Integrated debugging for scripts, UI, and tests
- ✅ Test discovery and execution with pytest
- ✅ Git integration with history and blame
- ✅ Project-specific settings matching `pyproject.toml`
- ✅ Tasks for common operations
