plugins {
    kotlin("jvm") version "2.4.0"
}

group = "org.soyuz"
version = "0.1-alpha"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(25)
}

tasks.test {
    useJUnitPlatform()
}