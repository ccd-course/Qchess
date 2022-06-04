import jpype
import jpype.imports


def init_jvm():
    jpype.startJVM()
    jpype.addClassPath("../lib/backend-2.0.1.jar")
