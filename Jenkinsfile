pipeline {
  agent {
    docker {
      image 'python:3.13-slim'
      args '-u root'
    }
  }

  options { timestamps(); ansiColor('xterm') }

  environment {
    PIP_CACHE_DIR = "${WORKSPACE}/.pip-cache"
    PYTHONDONTWRITEBYTECODE = '1'
  }

  stages {
    stage('System deps') {
      steps {
        sh '''
          set -euxo pipefail
          apt-get update
          apt-get install -y --no-install-recommends \
            build-essential \
            gfortran \
            python3-dev \
            ffmpeg \
            git \
            graphviz
          rm -rf /var/lib/apt/lists/*
        '''
      }
    }

    stage('Install') {
      steps {
        sh '''
          set -euxo pipefail
          python -m venv .venv
          . .venv/bin/activate
          python -m pip install --upgrade pip setuptools wheel
          pip install -e ".[dev,test,docs]"
        '''
      }
    }

    stage('Environment Setup') {
      steps {
        sh '''
          set -euxo pipefail
          . .venv/bin/activate
          python scripts/setup_environment.py --base-dir "$(pwd)" --skip-packages --skip-sample-data
        '''
      }
    }

    stage('Lint') {
      steps {
        sh '. .venv/bin/activate && ruff check src scripts'
        sh '. .venv/bin/activate && black --check src scripts'
        sh '. .venv/bin/activate && mypy src'
      }
    }

    stage('Tests') {
      steps {
        sh '''
          set -euxo pipefail
          . .venv/bin/activate
          mkdir -p build/reports
          if [ ! -d tests ]; then
            echo "Tests directory not found; add tests under ./tests to keep CI healthy." >&2
            exit 1
          fi

          pytest -m "not slow" -v \
            --junitxml=build/reports/junit.xml \
            --cov=src --cov-report=xml:build/reports/coverage.xml
        '''
      }
    }

    stage('Docs (Sphinx)') {
      steps {
        sh '''
          set -euxo pipefail
          . .venv/bin/activate
          if [ -d docs ]; then
            python -m sphinx.ext.apidoc -o docs/api src || true
            python -m sphinx.cmd.build -b html docs docs/_build/html
          fi
        '''
      }
    }

    stage('Publish reports') {
      steps {
        junit 'build/reports/junit.xml'
        script {
          def coverageReport = 'build/reports/coverage.xml'
          if (fileExists(coverageReport)) {
            recordCoverage(tools: [[parser: 'COBERTURA', pattern: coverageReport]])
          } else {
            echo "Coverage report ${coverageReport} not found; skipping coverage publish."
          }
        }
        archiveArtifacts artifacts: 'docs/_build/html/**', allowEmptyArchive: true
      }
    }
  }

  post {
    success { echo 'CI passed ✅' }
    always  { cleanWs() }
  }
}
