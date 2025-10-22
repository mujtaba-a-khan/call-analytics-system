# Domain-Specific Languages

## Jenkins Pipeline DSL
I scripted my CI/CD flow with the Jenkins pipeline DSL so each stage lives in one declarative block. The snippet below from `Jenkinsfile:1-83` shows how I describe tools, stages, and the hand-off between Ant and Maven.

```groovy
pipeline {
  agent any

  tools {
    maven 'MAVEN_3'
    ant 'ANT_1_10'
  }

  stages {
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
}
```

## Ant Build DSL
My `build.xml` (`build.xml:1-61`) leans on Ant’s XML DSL to chain the Python tooling. I split the workflow into reusable targets so Jenkins and my local shell can call the same entry points.

```xml
<project name="call-analytics-system" default="ci" basedir=".">
  <target name="lint">
    <exec executable="${venv.bin}/ruff" failonerror="true">
      <arg value="check"/><arg value="src"/><arg value="scripts"/>
    </exec>
    <exec executable="${venv.bin}/black" failonerror="true">
      <arg value="--check"/><arg value="src"/><arg value="scripts"/>
    </exec>
    <exec executable="${venv.bin}/mypy" failonerror="true">
      <arg value="src"/>
    </exec>
  </target>

  <target name="ci" depends="clean,setup,lint,test,docs,wheel"/>
</project>
```

## Maven POM DSL
I also documented the release packaging steps in `pom.xml:1-57`. Maven’s DSL lets me call Ant again under `verify` and then zip release assets with the Exec plugin.

```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-antrun-plugin</artifactId>
  <version>3.1.0</version>
  <executions>
    <execution>
      <id>run-ant-ci</id>
      <phase>verify</phase>
      <goals>
        <goal>run</goal>
      </goals>
      <configuration>
        <target>
          <ant antfile="build.xml" target="ci"/>
        </target>
      </configuration>
    </execution>
  </executions>
</plugin>
```

## Regex Rule DSL
I lean on custom regex patterns to shape analytics behavior. The configuration in `config/rules.toml:190-196` lists the patterns that feed into the labeling engine.

```toml
[custom_rules]
urgent_complaint = "(?i)(extremely|very|really)\\s+(unhappy|disappointed|frustrated)"
vip_customer = "(?i)(premium|vip|gold|platinum)\\s+(member|customer|client)"
technical_escalation = "(?i)(critical|severe|major)\\s+(bug|issue|problem)"
```

Those strings get compiled in `src/core/labeling_engine.py:59-75`, so I can override outcomes when a transcript matches one of the patterns.

```python
def _compile_custom_patterns(self) -> dict[str, re.Pattern]:
    patterns = {}
    custom_rules = self.rules.get("custom_rules", {})

    for name, pattern_str in custom_rules.items():
        try:
            patterns[name] = re.compile(pattern_str, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Invalid regex pattern for {name}: {e}")

    return patterns
```

## Natural-Language Query Patterns
To interpret ad-hoc questions I wrote another regex-powered mini language in `src/analysis/query_interpreter.py:106-148`. This mapping makes phrases like “last 7 days” or “compare outcomes” resolve into concrete filters.

```python
patterns = {
    "last_n_days": re.compile(r"last\s+(\d+)\s+days?", re.IGNORECASE),
    "call_type": re.compile(r"(inquiry|support|complaint|billing|sales)", re.IGNORECASE),
    "average": re.compile(r"\b(average|avg|mean)\b", re.IGNORECASE),
    "between": re.compile(r"between\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE),
    "compare": re.compile(r"\b(compare|versus|vs)\b", re.IGNORECASE),
}
```

I reuse the same idea in my text helpers (`src/utils/text_processing.py:265-334`) to pull entities like emails or amounts from raw transcripts whenever I need quick analytics inputs.

```python
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
money_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?"
```
