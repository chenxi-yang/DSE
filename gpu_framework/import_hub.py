import constants
import importlib

# a protocol for adding benchmarks
if constants.benchmark_name == "thermostat":
    import benchmarks.thermostat as tm
    importlib.reload(tm)
    from benchmarks.thermostat import *
elif constants.benchmark_name == "mountain_car":
    import benchmarks.mountain_car as mc
    importlib.reload(mc)
    from benchmarks.mountain_car import *
elif constants.benchmark_name == "unsmooth_1":
    import benchmarks.unsmooth_1 as us
    importlib.reload(us)
    from benchmarks.unsmooth_1 import *
elif constants.benchmark_name == "unsmooth_1_a":
    import benchmarks.unsmooth_1_a as usa
    importlib.reload(usa)
    from benchmarks.unsmooth_1_a import *
elif constants.benchmark_name == "unsmooth_1_b":
    import benchmarks.unsmooth_1_b as usa
    importlib.reload(usa)
    from benchmarks.unsmooth_1_b import *
elif constants.benchmark_name == "unsmooth_1_c":
    import benchmarks.unsmooth_1_c as usc
    importlib.reload(usc)
    from benchmarks.unsmooth_1_c import *
elif constants.benchmark_name == "unsmooth_2_separate":
    import benchmarks.unsmooth_2_separate as uss
    importlib.reload(uss)
    from benchmarks.unsmooth_2_separate import *
elif constants.benchmark_name == "unsmooth_2_overall":
    import benchmarks.unsmooth_2_overall as uso
    importlib.reload(uso)
    from benchmarks.unsmooth_2_overall import *
elif constants.benchmark_name == "path_explosion":
    import benchmarks.path_explosion as pe
    importlib.reload(pe)
    from benchmarks.path_explosion import *
elif constants.benchmark_name == "path_explosion_2":
    import benchmarks.path_explosion_2 as pe2
    importlib.reload(pe2)
    from benchmarks.path_explosion_2 import *
elif constants.benchmark_name in ["pattern1_a", "pattern1_b"]:
    import benchmarks.pattern1 as p1
    importlib.reload(p1)
    from benchmarks.pattern1 import *
elif constants.benchmark_name == "pattern2":
    import benchmarks.pattern2 as p2
    importlib.reload(p2)
    from benchmarks.pattern2 import *
elif constants.benchmark_name in ["pattern3_a", "pattern3_b"]:
    import benchmarks.pattern3 as p3
    importlib.reload(p3)
    from benchmarks.pattern3 import *
elif constants.benchmark_name in ["pattern31_a", "pattern31_b"]:
    import benchmarks.pattern31 as p31
    importlib.reload(p31)
    from benchmarks.pattern31 import *
elif constants.benchmark_name in ["pattern5_a", "pattern5_b"]:
    import benchmarks.pattern5 as p5
    importlib.reload(p5)
    from benchmarks.pattern5 import *
elif constants.benchmark_name == "pattern6":
    import benchmarks.pattern6 as p6
    importlib.reload(p6)
    from benchmarks.pattern6 import *
elif constants.benchmark_name == "pattern7":
    import benchmarks.pattern7 as p7
    importlib.reload(p7)
    from benchmarks.pattern7 import *
elif constants.benchmark_name == "pattern8":
    import benchmarks.pattern8 as p8
    importlib.reload(p8)
    from benchmarks.pattern8 import *
elif constants.benchmark_name == "pattern_example":
    import benchmarks.pattern_example as pe
    importlib.reload(pe)
    from benchmarks.pattern_example import *
elif constants.benchmark_name == "racetrack_easy":
    import benchmarks.racetrack_easy as re
    importlib.reload(re)
    from benchmarks.racetrack_easy import *
elif constants.benchmark_name == "racetrack_easy_classifier":
    import benchmarks.racetrack_easy_classifier as re
    importlib.reload(re)
    from benchmarks.racetrack_easy_classifier import *
elif constants.benchmark_name == "racetrack_easy_classifier_ITE":
    import benchmarks.racetrack_easy_classifier_ITE as re
    importlib.reload(re)
    from benchmarks.racetrack_easy_classifier_ITE import *
elif constants.benchmark_name == "racetrack_moderate_classifier_ITE":
    import benchmarks.racetrack_moderate_classifier_ITE as re
    importlib.reload(re)
    from benchmarks.racetrack_moderate_classifier_ITE import *
elif constants.benchmark_name == "racetrack_moderate_2_classifier_ITE":
    import benchmarks.racetrack_moderate_2_classifier_ITE as re
    importlib.reload(re)
    from benchmarks.racetrack_moderate_2_classifier_ITE import *
elif constants.benchmark_name == "racetrack_moderate_3_classifier_ITE":
    import benchmarks.racetrack_moderate_3_classifier_ITE as re
    importlib.reload(re)
    from benchmarks.racetrack_moderate_3_classifier_ITE import *
elif constants.benchmark_name == "racetrack_hard_classifier_ITE":
    import benchmarks.racetrack_hard_classifier_ITE as re
    importlib.reload(re)
    from benchmarks.racetrack_hard_classifier_ITE import *
elif constants.benchmark_name == "racetrack_easy_1_classifier":
    import benchmarks.racetrack_easy_1_classifier as re
    importlib.reload(re)
    from benchmarks.racetrack_easy_1_classifier import *
elif constants.benchmark_name == "racetrack_easy_2_classifier":
    import benchmarks.racetrack_easy_2_classifier as re
    importlib.reload(re)
    from benchmarks.racetrack_easy_2_classifier import *
elif constants.benchmark_name == "racetrack_easy_1":
    import benchmarks.racetrack_easy_1 as re1
    importlib.reload(re1)
    from benchmarks.racetrack_easy_1 import *
elif constants.benchmark_name == "thermostat_refined":
    import benchmarks.thermostat_refined as tr
    importlib.reload(tr)
    from benchmarks.thermostat_refined import *
elif constants.benchmark_name == "thermostat_new":
    import benchmarks.thermostat_new as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new import *
<<<<<<< HEAD
<<<<<<< HEAD
elif constants.benchmark_name == "thermostat_new_cnn":
    import benchmarks.thermostat_new_cnn as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_cnn import *
elif constants.benchmark_name == "thermostat_new_tinyinput":
    import benchmarks.thermostat_new_tinyinput as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_tinyinput import *
elif constants.benchmark_name == "thermostat_new_3branches":
    import benchmarks.thermostat_new_3branches as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_3branches import *
elif constants.benchmark_name == "thermostat_new_40":
    import benchmarks.thermostat_new_40 as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_40 import *
elif constants.benchmark_name == "thermostat_new_unsafe25":
    import benchmarks.thermostat_new_unsafe25 as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_unsafe25 import *
elif constants.benchmark_name == "thermostat_new_unsafe50":
    import benchmarks.thermostat_new_unsafe50 as tr
    importlib.reload(tr)
    from benchmarks.thermostat_new_unsafe50 import *
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
elif constants.benchmark_name == "racetrack_easy_sample":
    import benchmarks.racetrack_easy_sample as res
    importlib.reload(res)
    from benchmarks.racetrack_easy_sample import *
elif constants.benchmark_name == "racetrack_easy_multi":
    import benchmarks.racetrack_easy_multi as res
    importlib.reload(res)
    from benchmarks.racetrack_easy_multi import *
elif constants.benchmark_name == "racetrack_easy_multi2":
    import benchmarks.racetrack_easy_multi2 as res
    importlib.reload(res)
    from benchmarks.racetrack_easy_multi2 import *
elif constants.benchmark_name == "racetrack_relaxed_multi":
    import benchmarks.racetrack_relaxed_multi as res
    importlib.reload(res)
    from benchmarks.racetrack_relaxed_multi import *
elif constants.benchmark_name == "racetrack_relaxed_multi2":
    import benchmarks.racetrack_relaxed_multi2 as res
    importlib.reload(res)
    from benchmarks.racetrack_relaxed_multi2 import *
elif constants.benchmark_name == "aircraft_collision":
    import benchmarks.aircraft_collision as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision import *
elif constants.benchmark_name == "aircraft_collision_refined":
    import benchmarks.aircraft_collision_refined as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_refined import *
elif constants.benchmark_name == "aircraft_collision_refined_classifier":
    import benchmarks.aircraft_collision_refined_classifier as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_refined_classifier import *
elif constants.benchmark_name == "aircraft_collision_refined_classifier_ITE":
    import benchmarks.aircraft_collision_refined_classifier_ITE as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_refined_classifier_ITE import *
elif constants.benchmark_name == "aircraft_collision_new":
    import benchmarks.aircraft_collision_new as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_new import *
elif constants.benchmark_name == "aircraft_collision_new_1":
    import benchmarks.aircraft_collision_new_1 as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_new_1 import *
<<<<<<< HEAD
<<<<<<< HEAD
elif constants.benchmark_name == "aircraft_collision_new_1_cnn":
    import benchmarks.aircraft_collision_new_1_cnn as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_new_1_cnn import *
elif constants.benchmark_name == "aircraft_collision_new_1_unsafe25":
    import benchmarks.aircraft_collision_new_1_unsafe25 as ac
    importlib.reload(ac)
    from benchmarks.aircraft_collision_new_1_unsafe25 import *
elif constants.benchmark_name == "cartpole_v1":
    import benchmarks.cartpole_v1 as ac
    importlib.reload(ac)
    from benchmarks.cartpole_v1 import *
elif constants.benchmark_name == "cartpole_v2":
    import benchmarks.cartpole_v2 as ac
    importlib.reload(ac)
    from benchmarks.cartpole_v2 import *
elif constants.benchmark_name == "cartpole_v3":
    import benchmarks.cartpole_v3 as ac
    importlib.reload(ac)
    from benchmarks.cartpole_v3 import *
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f

# print(constants.benchmark_name)

