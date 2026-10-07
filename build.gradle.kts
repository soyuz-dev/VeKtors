plugins {
    kotlin("jvm") version "2.4.0"
}

group = "org.soyuz"
version = "0.2-alpha"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
}

kotlin {
    jvmToolchain(25)
}

tasks.test {
    useJUnitPlatform()
}