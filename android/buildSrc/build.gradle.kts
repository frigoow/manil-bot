// The desktop app's native build (CMakeJniLibrary, findCmake, findJniInclude)
// is cardbot-framework's: its sources compiled in here from the submodule.
plugins { `kotlin-dsl` }
repositories { mavenCentral() }
sourceSets { main { kotlin.srcDir("../../cardbot-framework/build-logic") } }
