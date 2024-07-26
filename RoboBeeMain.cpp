#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace std; 
using namespace Eigen;

// -------------------------------- REFERENCES -------------------------------- //
// EKF (Extended Kalman Filter) -> https://www.kalmanfilter.net/default.aspx
// Eigen Handbook -> https://github.com/AIBluefisher/Eigen_Handbook/tree/master/EN
// Basic how to use Eigen -> https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
// Jacobian review -> https://www.youtube.com/playlist?list=PLEZWS2fT1672lJI7FT5OXHJU6cTgkSzV2
// Eigen Matrix/Vector datatype nomenclature -> https://eigen.tuxfamily.org/dox/group__matrixtypedefs.html#ga9f54d6a47f5267f83c415ac54f5a89f3
// Eigen Map class -> https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
// .array() element access -> https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html

class RobobeeEKF{
            // The RobobeeEKF class holds the information that is relevent for calculations moving RoboBee from point A to point B. 
            // The class itself has three functions: predictState(), jacobian(), and update() which have their own descriptions below.
            // Self explanitory things such as mass, drag, distance between center of mass, etc... pertain to RoboBee itself.
            // Things such as H, R, Q and P matrices/vectors are explained further in the link to the Kalman filer literature above.    

    public:

        Matrix<float, 4, 10> H  {                                   // nonlinear observation matrix
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        };

        Matrix<float, 4, 4> R {                                     // uncertainty estimation (uncertainty of position measurement)
            {0.07, 0, 0, 0 },
            {0, 0.07, 0, 0 },
            {0, 0, 0.07, 0 },
            {0, 0, 0, 0.002},
        };
        
           VectorXf x       = VectorXf::Zero(10);                   // To hold ten values -> rollAngle, pitchAngle, yawAngle, angleVelocityRoll, angleVelocityPitch, 
                                                                    // angleVelocityYaw, altitude, linearVelocityX, linearVelocityY, linearVelocityZ.
                                                                    // vectors default to column vectors unless specified by RowVector

           
           MatrixXf Q       = MatrixXf::Identity(10, 10);           // process noise matrix
           MatrixXf P       = MatrixXf::Identity(10, 10) * (M_PI/2);// uncertainty estimation (uncertainty of position estimate)
              float dt      = 4e-3f;                                // time differential          // f is float literal
        const float drag    = 2e-4f;                                // drag constant
        const float cmDist  = 9e-3f;                                // distance between center of mass and mid point of wings 
        RowVector3f I       {{1.42e-9, 1.34e-9, 4.5e-10}};          // moment of inertia x, y, z
        const float m       = 8.6e-05;                              // mass of robot

        Vector<float, 10> predictState(VectorXf xHat, Vector4f U){  // f(x), state prediction equation
                                                                    // predictState(vect 10x1, vect 4x1)
            // This function predicts RoboBee's future position after a small time differential has passed. 
            // Vectors are passed into the function for arithmetic

            float phi       = xHat(0);                              // angle of roll
            float theta     = xHat(1);                              // angle of pitch
            float psi       = xHat(2);                              // angle of yaw

            float angVelRol = xHat(3);                              // angular velocity roll rate
            float angVelPit = xHat(4);                              // angular velocity pitch rate
            float angVelYaw = xHat(5);                              // angular velocity yaw rate

            float z         = xHat(6);                              // height above the ground plane

            float linVelX   = xHat(7);                              // linear velocity in the x direction
            float linVelY   = xHat(8);                              // linear velocity in the y direction
            float linVelZ   = xHat(9);                              // linear velocity in the z direction

            float inertiaX  = I(0);                                 // moment of inertia in x direction
            float inertiaY  = I(1);                                 // moment of inertia in y direction
            float inertiaZ  = I(2);                                 // moment of inertia in z direction

            float torqX     = U(0);                                 // torque in the x direction
            float torqY     = U(1);                                 // torque in the y direction
            float torqZ     = U(2);                                 // torque in the z direction
            float force     = U(3);                                 // force of thrust

            float g         = 9.8;                                  // gravitational acceleration

            Vector<float, 10> xDot;                                 // 10x1 column vector to hold information
                                                                    // about future vel, accel, etc data

            float linVel    = linVelX*cos(psi)*cos(theta) +         // total linear velocity
                                linVelY*cos(theta)*sin(psi) - 
                                linVelZ*sin(theta);
            
                                                                    // total angular velocity
            float angVel    = angVelRol*(cos(psi)*sin(theta)*sin(psi) - sin(psi)*cos(phi)) +
                                angVelPit*(sin(psi)*sin(theta)*sin(psi) - cos(psi)*cos(phi)) + 
                                angVelYaw*(cos(theta)*sin(phi));

            float forceFunc = -drag*(cmDist*angVel + linVel);       // force of drag
            float torqFunc  = -cmDist*forceFunc;                    // torque of drag

                                                                    // total force from the world perspective
            Vector3f totalWorldForce(cos(psi)*cos(theta)*forceFunc + (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*force, 
                                    sin(psi)*cos(theta)*forceFunc + (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))*force, 
                                    -sin(theta)*forceFunc + cos(theta)*cos(phi)*force - m*g);

                                                                    // total torque from the world perspective
            Vector3f totalWorldTorq(cos(psi)*cos(theta)*torqX + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(torqY + torqFunc) + (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*torqZ,
                                    sin(psi)*cos(theta)*torqX + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(torqY + torqFunc) + (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))*torqZ,
                                    -sin(theta)*torqX + (cos(theta)*sin(phi))*(torqY + torqFunc) + cos(theta)*cos(phi)*torqZ);
            
            xDot(0)         = angVelRol;
            xDot(1)         = angVelPit;
            xDot(2)         = angVelYaw;
            xDot(3)         = (1/inertiaX)*totalWorldTorq(0);       // angular acceleration roll rate
            xDot(4)         = (1/inertiaY)*totalWorldTorq(1);       // angular acceleration pitch rate  
            xDot(5)         = (1/inertiaZ)*totalWorldTorq(2);       // angular acceleration yaw rate
            xDot(6)         = linVelZ;

            xDot(7)         = totalWorldForce(0)/m;                 // linear acceleration in the x direction
            xDot(8)         = totalWorldForce(1)/m;                 // linear acceleration in the y direction
            xDot(9)         = totalWorldForce(2)/m;                 // linear acceleration in the z direction

            return  xHat + xDot * dt;                               // returns 10x1 vector
        }// end predictState

        Matrix<float, 10, 10> jacobian(Vector<float, 10> xHat, Vector4f U){
            // This function takes in two vectors and a jacobian    // jacobian(vect 10x1, vect 4x1)
            // matrix operator is applied to the list of values.
            // A nonlinear path is remaped to a linear path. Meaning that if RoboBee takes
            // a slightly parabolic path in three dimensions, the jacobian will remap its
            // parabolic path to a linear path in three dimensions. 

            float phi       = xHat(0);                              // angle of roll
            float theta     = xHat(1);                              // angle of pitch
            float psi       = xHat(2);                              // angle of yaw

            float angVelRol = xHat(3);                              // angular velocity roll rate
            float angVelPit = xHat(4);                              // angular velocity pitch rate
            float angVelYaw = xHat(5);                              // angular velocity yaw rate

            float z         = xHat(6);                              // height above the ground plane

            float linVelX   = xHat(7);                              // linear velocity in the x direction
            float linVelY   = xHat(8);                              // linear velocity in the y direction
            float linVelZ   = xHat(9);                              // linear velocity in the z direction

            float inertiaX  = I(0);                                 // moment of inertia in x direction
            float inertiaY  = I(1);                                 // moment of inertia in y direction
            float inertiaZ  = I(2);                                 // moment of inertia in z direction

            float torqX     = U(0);                                 // torque in the x direction
            float torqY     = U(1);                                 // torque in the y direction
            float torqZ     = U(2);                                 // torque in the z direction
            float force     = U(3);                                 // force of thrust

            float linVel    = linVelX*cos(psi)*cos(theta) +         // linear velocity vector (v hat)
                              linVelY*cos(theta)*sin(psi) -
                              linVelZ*sin(theta);
                                                                    // angular velocity vector (omega hat)
            float angVel    = angVelRol*(cos(psi)*sin(theta)*sin(psi) - sin(psi)*cos(phi)) +
                              angVelPit*(sin(psi)*sin(theta)*sin(psi) - cos(psi)*cos(phi)) + 
                              angVelYaw*(cos(theta)*sin(phi));

            float forceFunc = -drag*(cmDist*angVel + linVel);       // force of drag
            float torqFunc  = -cmDist*forceFunc;                    // torque of drag

            float dtd_dfd   = -cmDist;                              // derivatives for jacobian are listed below

            float dfd_dw    = -drag*cmDist;
            float dfd_dv    = -drag;

            float dv_dvx    = cos(psi)*cos(theta);
            float dv_dvy    = cos(theta)*sin(psi);
            float dv_dvz    = -sin(theta);
            float dv_dtheta = -linVelX*cos(psi)*sin(theta) - linVelY*sin(theta)*sin(psi) - linVelZ*cos(theta);
            float dv_dpsi   = -linVelX*sin(psi)*cos(theta) + linVelY*cos(psi)*cos(theta);

            float dw_dp     = (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi));
            float dw_dq     = (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi));
            float dw_dr     = cos(theta)*sin(phi);
            float dw_dtheta = angVelRol*(cos(psi)*cos(theta)*sin(phi)) + angVelPit*(sin(psi)*cos(theta)*sin(phi)) - angVelYaw*sin(theta)*sin(phi);
            float dw_dpsi   = angVelRol*(-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi)) + angVelPit*(cos(psi)*sin(theta)*sin(phi) + sin(psi)*cos(phi));

                                                                    // create a 10x10 matrix and fill it with zeros
            Matrix<float, 10, 10> A = Matrix<float, 10, 10>::Zero(10, 10); 

            A(0, 3) = 1;
            A(1, 4) = 1;
            A(2, 5) = 1;

            A(3, 0) = (1/inertiaX)*((torqY + torqFunc)*(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)) + torqZ*(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)));
            A(3, 1) = (1/inertiaX)*(-torqX*(cos(psi)*sin(theta)) + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) + (torqY + torqFunc)*(cos(psi)*cos(theta)*sin(phi)) + (cos(psi)*cos(theta)*cos(phi))*torqZ);
            A(3, 2) = (1/inertiaX)*(-sin(psi)*cos(theta)*torqX + (-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(torqY + torqFunc) + (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi)) + (-sin(psi)*sin(theta)*cos(phi) + cos(psi)*sin(phi))*torqZ);
            A(3, 3) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dp);
            A(3, 4) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dq);
            A(3, 5) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dr);

            A(3, 7) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvx);
            A(3, 8) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvy);
            A(3, 9) = (1/inertiaX)*((cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvz);
            
            A(4, 0) = (1/inertiaY)*((torqY + torqFunc)*(-sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)) + torqZ*(-sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi)));
            A(4, 1) = (1/inertiaY)*(sin(psi)*-sin(theta)*torqX + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) + (torqY + torqFunc)*(sin(psi)*cos(theta)*sin(phi)) + (sin(psi)*cos(theta)*cos(phi))*torqZ);
            A(4, 2) = (1/inertiaY)*(cos(psi)*cos(theta)*torqX + (cos(psi)*sin(theta)*sin(phi) + sin(psi)*cos(phi))*(torqY + torqFunc) + (sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*(dtd_dfd*(dfd_dw*dw_dpsi + dfd_dv*dv_dpsi)) + (-cos(psi)*sin(theta)*cos(phi) - sin(psi)*sin(phi))*torqZ);
            A(4, 3) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dp);
            A(4, 4) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dq);
            A(4, 5) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dw*dw_dr);

            A(4, 7) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvx);
            A(4, 8) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvy);
            A(4, 9) = (1/inertiaY)*((sin(psi)*sin(theta)*sin(phi) - cos(psi)*cos(phi))*dtd_dfd*dfd_dv*dv_dvz);

            A(5, 0) = (1/inertiaZ)*((cos(theta)*cos(phi))*(torqY + torqFunc) - cos(theta)*sin(phi)*torqZ);
            A(5, 1) = (1/inertiaZ)*(-cos(theta)*torqX + (sin(theta)*sin(phi))*(torqY + torqFunc) + (cos(theta)*sin(phi))*(dtd_dfd*(dfd_dw*dw_dtheta + dfd_dv*dv_dtheta)) - sin(theta)*cos(phi)*torqZ);
            A(5, 3) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dp);
            A(5, 4) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dq);
            A(5, 5) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dw*dw_dr);

            A(5, 7) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvx);
            A(5, 8) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvy);
            A(5, 9) = (1/inertiaZ)*((cos(theta)*sin(phi))*dtd_dfd*dfd_dv*dv_dvz);

            A(6, 9) = 1;

            A(7, 0) = (1/m)*((cos(psi)*sin(theta)*-sin(phi) + sin(psi)*cos(phi))*force);
            A(7, 1) = (1/m)*(cos(psi)*-sin(theta)*forceFunc + cos(psi)*cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) + (cos(psi)*cos(theta)*cos(phi))*force);
            A(7, 2) = (1/m)*(-sin(psi)*cos(theta)*forceFunc + cos(psi)*cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi) + (-sin(psi)*sin(theta)*cos(phi) + cos(psi)*sin(phi))*force);
            A(7, 3) = (1/m)*cos(psi)*cos(theta)*dfd_dw*dw_dp;
            A(7, 4) = (1/m)*cos(psi)*cos(theta)*dfd_dw*dw_dq;
            A(7, 5) = (1/m)*cos(psi)*cos(theta)*dfd_dw*dw_dr;

            A(7, 7) = (1/m)*cos(psi)*cos(theta)*dfd_dv*dv_dvx;
            A(7, 8) = (1/m)*cos(psi)*cos(theta)*dfd_dv*dv_dvy;
            A(7, 9) = (1/m)*cos(psi)*cos(theta)*dfd_dv*dv_dvz;

            A(8, 0) = (1/m)*((cos(psi)*sin(theta)*-sin(phi) + sin(psi)*cos(phi))*force);
            A(8, 1) = (1/m)*(sin(psi)*-sin(theta)*forceFunc + sin(psi)*cos(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) + (sin(psi)*cos(theta)*cos(phi) - cos(psi)*sin(phi))*force);
            A(8, 2) = (1/m)*(cos(psi)*cos(theta)*forceFunc + sin(psi)*cos(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi) + (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))*force);
            A(8, 3) = (1/m)*sin(psi)*cos(theta)*dfd_dw*dw_dp;
            A(8, 4) = (1/m)*sin(psi)*cos(theta)*dfd_dw*dw_dq;
            A(8, 5) = (1/m)*sin(psi)*cos(theta)*dfd_dw*dw_dr;

            A(8, 7) = (1/m)*sin(psi)*cos(theta)*dfd_dv*dv_dvx;
            A(8, 8) = (1/m)*sin(psi)*cos(theta)*dfd_dv*dv_dvy;
            A(8, 9) = (1/m)*sin(psi)*cos(theta)*dfd_dv*dv_dvz;

            A(9, 0) = (1/m)*(cos(theta)*-sin(phi)*force);
            A(9, 1) = (1/m)*(-cos(theta)*forceFunc + -sin(theta)*(dfd_dv*dv_dtheta + dfd_dw*dw_dtheta) - sin(theta)*cos(phi)*force);
            A(9, 2) = (1/m)*(-sin(theta)*(dfd_dv*dv_dpsi + dfd_dw*dw_dpsi));
            A(9, 3) = (1/m)*-sin(theta)*dfd_dw*dw_dp;
            A(9, 4) = (1/m)*-sin(theta)*dfd_dw*dw_dq;
            A(9, 5) = (1/m)*-sin(theta)*dfd_dw*dw_dr;

            A(9, 7) = (1/m)*-sin(theta)*dfd_dv*dv_dvx;
            A(9, 8) = (1/m)*-sin(theta)*dfd_dv*dv_dvy;
            A(9, 9) = (1/m)*-sin(theta)*dfd_dv*dv_dvz;
                                                                    // create a 10x10 identity matrix
            Matrix<float, 10, 10> IdentMatrix = Matrix<float, 10, 10>::Identity(10, 10);

            return IdentMatrix + A*dt;                              // returns 10x10 matrix
        }// end jacobian

        Vector4f update(Vector4f altitudeVect, Vector4f U, float timeDiff = -1.0){
                                                                    // update(vect 4x1, vect 4x1, optional datapoint)
            // This function takes in two vectors an an optional time differential.
            // In this iteration of the code, the time differential is never passed in.
            // Perhaps in the future it will be. Each matrix that handles the statistical
            // information is updated in this function. 

            if (timeDiff != -1){ dt = timeDiff; }                   // if the time differential hasnt changed, use the same one
                                                                    // if it has, use the new one. 

            Matrix<float, 10, 10> A  = jacobian(x, U);              // 10x10 matrix,        jacobian matrix
            Vector<float, 10> xp     = predictState(x, U);          // 10x1 column vector,  estimation of current position
            Matrix<float, 10, 10> Pp = A * P * A.transpose() + Q;   // 10x10 matrix,        covariance extrapolation equation

            Matrix4f G               = H * Pp *  H.transpose() + R; // 4x4 matrix,          control matrix

                                                                    // 10x4 matrix,         kalman gain matrix       
            Matrix<float, 10, 4> K   = Pp * H.transpose() * G.completeOrthogonalDecomposition().pseudoInverse();
            
            Vector4f e               = altitudeVect - (H * xp);     // 4x1 column vector,   error in altitude

            Vector<float, 10> x      = xp + K * e;                  // 10x1 column vector,  state update equation
            Matrix<float, 10, 10> P  = Pp - K * H * Pp;             // 10x10 matrix,        uncertainty of position estimate

            return H * x;                                           // returns 4x1 column vector
        }// end update

}; // end RobobeeEKF class

    MatrixXf loadDataMatrix(string inputFile){
                // This function opens the CSV file where the datapoints live. After the
                // file is opened, the data is scanned one row at a time and converted   
                // datapoint by datapoint left to right from a string to a float. After  
                // individual datapoints are converted, they are then loaded into a 
                // into a traditional vector and the function then moves on to a new row.
                // The function keeps track of which row is which with matrixRowNumber.
                // After the dataMatrixContainer vector is fully loaded, it is remapped
                // to an MxN matrix which the rest fo the program uses. 

        vector<float> dataMatrixContainer;                          // vector to hold all data entries of csv 

        ifstream csvFile;                                           // csv file variable
        string csvRowEntry;                                         // string to hold the entire csv row data values
        string dataMatrixRow;                                       // string to hold csv data as a matrix
        int matrixRowNumber = 0;                                    // index of the rows

        csvFile.open(inputFile);
        getline(csvFile, csvRowEntry);                              // skip the first line of the csv with the header names

        while(getline(csvFile, csvRowEntry)){                       // while not at the end of the csv
            stringstream csvRowEntryStream(csvRowEntry);            // convert current row string to stream, enabling reading of string like normal

            while(getline(csvRowEntryStream, dataMatrixRow, ',')){  // while not at the end of the current row
                dataMatrixContainer.push_back(stof(dataMatrixRow)); // convert the row of data entries from a string -> float
                                                                    // and store them in the dataMatrixContainer vector
            }// end while

            matrixRowNumber++;                                      // increment the row index
        }// end while

        csvFile.close();
                                                                    // map the csv data in the vector to an MxN float matrix
        Map<Matrix<float, Dynamic, Dynamic, RowMajor>> dataMatrix(dataMatrixContainer.data(), matrixRowNumber, dataMatrixContainer.size()/matrixRowNumber);
                                                                    // -->                                                 cols = size/rows
        return dataMatrix;                                          // returns MxN matrix
    }// end loadDataMatrix

    Matrix<float, Dynamic, 4> getEstimatedTrajectory(RobobeeEKF ekfObjectData,  MatrixXf data){
                // This function attempts to create a path for      // getEstimatedTrajectory(object, matrix MxN) [golden.csv -> 1195x21]
                // RoboBee that is the exact same as the known
                // path that RoboBee will travel. The update function is used in the for loop to predict the next
                // state and position of RoboBee for the entire flight. The matrix columns in the output of the
                // matrix are rotation about the X-Axis, rotation about the Y-Axis, rotation about the Z-Axis,
                // and altitude from the time of flight sensor.

        int totalRows       = data.rows();                          // total rows in the csv data matrix

                                                                    // create an Nx4 matrix where the rows are equal to total csv rows
        Matrix<float, Dynamic, 4> est_Trajec = Matrix<float, Dynamic, 4>::Zero(totalRows, 4);

        for(int i = 0; i < totalRows; i++){
                                                                    // block() operations https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
            Vector3f imu    = data.block<1, 3>(i, 1).transpose();   // copy a 1x3 block of the csv data starting at row i col 1
                                                                    // and store it in imu 3x1 col vector

            Vector4f meas   = Vector4f::Zero();                     // create a 4x1 float vector with zeroes for entries meas(rx, ry, rz, tof)
                                                                    // meas(rx, ry, rz, tof) = meas(rotationAboutX-Axis, 
                                                                    // rotationAboutY-Axis, rotationAboutZ-Axis, TimeOfFlight[altitude])

            meas.head<3>()  = (M_PI/ 180) * imu;                    // convert rx, ry, rz values to radians
            meas(3)         = (1e-3) * data(i, 4);                  // scale the tof data value in csv data matrix and store in meas vector
                                                                    
                                                                    // syntax is fixed-sized block expression
            Vector4f U      = data.block<1, 4>(i, 17).transpose();  // create a 4x1 float vector U(tx, ty, tz, ft)
                                                                    // U(torqueX, torqueY, torqueZ, forceOfThrust)
            
            if(i > 0){                                              // if past the first datapoint
                                                                    // update meas, U and DeltaTimestamp and get back a 4x1 col vector from update function
                est_Trajec.row(i) = ekfObjectData.update(meas, U, data(i, 0) - data(i-1, 0)).transpose();
                                                                    // store the transpose (1x4 row vector) in the i'th row of est_Trajec matrix
            } 
            else{                                                   // if not past the first datapoint
                                                                    // update meas, U and DeltaTimestamp and get back a 4x1 col vector from update function
                est_Trajec.row(i) = ekfObjectData.update(meas, U).transpose();
                                                                    // store the transpose (1x4 row vector) in the i'th row of est_Trajec matrix
            }
            
        }// end for

        return est_Trajec;                                          // returns Nx4 matrix
    }// end getEstimatedTrajectory

    Matrix<float, Dynamic, 4> getGroundTruth(MatrixXf data){
                // This function loads the known values of          // getGroundTruth(matrix MxN) [golden.csv -> 1195x21]
                // RoboBee's position and other data. 
                // This is the data that is being measured against by the estimations
                // and predictions. The errors in the output of the program are the 
                // error between this functions output and estimated spatial positions. 

        int totalRows       = data.rows();                          // total rows in the csv data matrix
                                                                    // create an Nx4 matrix where the rows are equal to total csv rows
        Matrix<float, Dynamic, 4> gndTruth = Matrix<float, Dynamic, 4>::Zero(totalRows, 4);

                                                                    // syntax is dynamic-sized block expression
                                                                    // copy an Nx3 block of the csv data matrix starting at row 0 col 8 -> row 0 col 11
        gndTruth.block(0, 0, totalRows, 3) = data.block(0, 8, totalRows, 3);  
            // this ^ and V could be combined into one statement, start datablock above at 7 rather than 8
        gndTruth.col(3)     = data.col(7);                          // ground truth col 3 is the same as csv data matrix col 7

        VectorXf timeStamp  = data.col(0);                          // time is an Nx1 column vector                       


        Matrix<float, Dynamic, 4> altAndTime(totalRows, 4);         // create an Nx4 matrix to hold data values (ignore timestamp column)
                                                                    //          | TCP_pose_3, TCP_pose_4, TCP_pose_5, TCP_pose_2, timestamp
        altAndTime.col(0)   = gndTruth. col(0);                     //  row 1   |    value  ,    value  ,   value   ,   value   ,   value
        altAndTime.col(1)   = gndTruth. col(1);                     //  row 2   |    value  ,    value  ,   value   ,   value   ,   value 
        altAndTime.col(2)   = gndTruth. col(2);                     //   ...    |                           ...
        altAndTime.col(3)   = gndTruth. col(3);                     //  row n   |    value  ,    value  ,   value   ,   value   ,   value

        //altAndTime.col(4)   = timeStamp;                          // this part of the altAndTime col vector does 
                                                                    // not get used anywyere, perhaps in the future...

        return altAndTime;                                          // returns Nx4 matrix, this ^^^
    }// end getGroundTruth
    
    RowVector4f getRMSE(Matrix<float, Dynamic, 4> trueTraj, Matrix<float, Dynamic, 4> estimTraj){
                // This function takes the square root, mean        // getRMSE(matrix MxN, matrix MxN) [golden.csv -> 1195x21]
                // and square of each individual datapoint in
                // the error matrix. All entries are accessed individually with the .array() function.

        int totalRows       = trueTraj.rows();                      // trueTraj.rows() = estimTraj.rows()

                                                                    // error is an Nx4 matrix
        Matrix<float, Dynamic, 4> error = Matrix<float, Dynamic, 4>::Zero(totalRows, trueTraj.cols());

                                                                    // block copy the difference of the first three columns 
                                                                    // of two Nx4 matrices, trueTraj and estimTraj  
                                                                    // matrices block copying starts at row 0 col 0 -> row 0 col 2
                                                                    // convert form radians -> degrees
        error.block(0, 0, trueTraj.rows(), 3) = (180 / M_PI) * (trueTraj.block(0, 0, trueTraj.rows(), 3) - estimTraj.block(0, 0, estimTraj.rows(), 3));
        
        error.col(3)        = trueTraj.col(3) - estimTraj.col(3);   // error col 3 is the same as the difference between trajectory col 3's 

                                                                    // .array() element access
        Matrix<float, Dynamic,4> sqrError = error.array().square(); // square each individual element of the error matrix
        RowVector4f meanSqrError = sqrError.colwise().mean();       // take the mean of all columns
        RowVector4f result  = meanSqrError.array().sqrt();          // take the radical of each individual element of the error matrix
        
        return result;                                              // 1x4 matrix
    }// end getRMSE

int main(){
    // The main program creates an object to hold the data for analysis, then loads an MxN matrix with the
    // data from a CSV file to be used. The true trajectory is known from the CSV file data. A projected 
    // trajectory is estimated and then the RMSE between the estimated and known trajectory 
    // determines how accurate the algorithm as a whole is. 

    RobobeeEKF beeDataStructure;                                    // initialize RobobeeEKF object for use

    //MatrixXf csvDataMatrix  = loadDataMatrix("testCSV.csv");      // only for tesing purposes
    MatrixXf csvDataMatrix = loadDataMatrix("golden.csv");          // load all CSV data into a matrix for use
   
                                                                    // true traj -> Nx4 matrix
    Matrix<float, Dynamic, 4> true_trajectory = getGroundTruth(csvDataMatrix);
                                                                    // store all known values of RoboBee's position etc...

                                                                    // estim traj -> Nx4 matrix
                                                                    // store all estimated values of where Robobee will be
    Matrix<float, Dynamic, 4> estimated_trajectory = getEstimatedTrajectory(beeDataStructure, csvDataMatrix);

                                                                    // traj error -> 1x4 matrix
                                                                    // find the error between known trajectory and estimated trajectory
    RowVector4f TrajectoryError = getRMSE(true_trajectory, estimated_trajectory);

                                                                    // display the error values. These are percentages.
    cout << " errPitch     errRoll    errYaw   errAltitude: " << endl << TrajectoryError << endl;

return 0;
}// end main