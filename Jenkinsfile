pipeline {
  agent any

  tools {
    maven 'MAVEN_3'
    ant 'ANT_1_10'
  }

  options { timestamps(); ansiColor('xterm') }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Prepare Python') {
      steps {
        sh '''set -e
if python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "ensurepip available; skipping system package install."
else
  echo "ensurepip missing; attempting to install python3-venv..."
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
    else
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
    fi
    python3 -c "import ensurepip" >/dev/null 2>&1 || { echo "ensurepip still unavailable after installation."; exit 1; }
  else
    echo "apt-get not available. Please install python3-venv manually on this agent."
    exit 1
  fi
fi
'''
      }
    }

    stage('Ant: Clean + Setup') {
      steps {
        sh 'ant -noinput -buildfile build.xml clean'
        sh 'ant -noinput -buildfile build.xml setup'
      }
    }

    stage('Ant: Lint + Test + Docs + Wheel') {
      steps {
        sh 'ant -noinput -buildfile build.xml lint'
        sh 'ant -noinput -buildfile build.xml test'
        sh 'ant -noinput -buildfile build.xml docs'
        sh 'ant -noinput -buildfile build.xml wheel'
      }
    }

    stage('Maven: Verify (runs Ant inside) & Package ZIP') {
      steps {
        sh 'mvn -B -V -q verify'
        sh 'mvn -B -q package'
      }
    }
  }

  post {
    always {
      junit 'test-reports/*.xml'
      archiveArtifacts artifacts: 'dist/*.whl, docs/_build/html/**, artifacts/*.zip', allowEmptyArchive: true
      cleanWs()
    }
    success {
      echo 'CI passed ✅'
    }
  }
}
