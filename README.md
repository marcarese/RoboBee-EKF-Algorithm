This program is meant to analyze and estimate the data given by a CSV file. It does this by using an Extended Kalman Filter, the Eigan C++ library, and Linear Algebra. Below is a list of resources to aid with understanding parts of the code as well as review for certain pertinent topics used in the algorithm.  


###---------------------------------- REFERENCES ----------------------------------###

EKF (Extended Kalman Filter) -> https://www.kalmanfilter.net/default.aspx
Eigen Handbook -> https://github.com/AIBluefisher/Eigen_Handbook/tree/master/EN
Basic how to use Eigen -> https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
Jacobian review -> https://www.youtube.com/playlist?list=PLEZWS2fT1672lJI7FT5OXHJU6cTgkSzV2
Eigen Matrix/Vector datatype nomenclature -> https://eigen.tuxfamily.org/dox/group__matrixtypedefs.html#ga9f54d6a47f5267f83c415ac54f5a89f3
Eigen block operations -> https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
Eigen Map class -> https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
.array() element access -> https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html

###----------------------------- Results and analytics -----------------------------###
Proccessor: 48Mhz M0 Arm Cortex

Memory used for entire program:                     539.96 Kilobytes
Number of cycles used to analyze one datapoint:     1.45e4 -> 14.5 Kilocycles
Number of cycles used to analyze all datapoints:    1.73e7 -> 17.3 Megacycles

