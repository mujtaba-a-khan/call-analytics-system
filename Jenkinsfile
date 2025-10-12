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
