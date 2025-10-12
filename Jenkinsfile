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
REQUIRED_PACKAGES="python3-venv python3-dev build-essential"
ensure_ensurepip() {
  python3 -c "import ensurepip" >/dev/null 2>&1
}

needs_install=false
if ! ensure_ensurepip; then
  needs_install=true
fi

if ! command -v gcc >/dev/null 2>&1; then
  needs_install=true
fi

if [ "$needs_install" = true ]; then
  echo "Installing required system packages: $REQUIRED_PACKAGES"
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $REQUIRED_PACKAGES
    else
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y $REQUIRED_PACKAGES
    fi
  else
    echo "apt-get not available. Please install: $REQUIRED_PACKAGES"
    exit 1
  fi
else
  echo "Required Python tooling and compiler already present."
fi

ensure_ensurepip || { echo "ensurepip still unavailable after attempted installation."; exit 1; }
command -v gcc >/dev/null 2>&1 || { echo "gcc still unavailable after attempted installation."; exit 1; }
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
