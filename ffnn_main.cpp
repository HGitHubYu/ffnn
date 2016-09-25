///////////////////////////////////
// ffnn_main.cpp
// Huan Yu
///////////////////////////////////

#include <iostream>
#include <Eigen/Dense>
#include "ffnn.h"
#include <string>
#include <vector>
#include <time.h>

using namespace Eigen;

int main()
{
	int test=1;
	if(test==1){
		int num_InputUnits=3;
		Matrix<int, 1, Dynamic> num_HiddenNeurons(1,2);
		num_HiddenNeurons << 7, 5;
		int num_OutputUnits=2; // set the number of output units.
		ffnn net(num_InputUnits, num_HiddenNeurons, num_OutputUnits);
		
		std::string save_VerilogA_file_path="ffnn.va";
		if(!net.convert_ffnn_to_VerilogA(save_VerilogA_file_path))
			std::cerr<<"convert_ffnn_to_VerilogA failed"<<std::endl;

		return 0;
	}

	
	else{

	clock_t start_time=clock();

	MatrixXd TrainingData_input, TrainingData_output, simulate_output;
	std::string TrainingData_input_file_path="TrainingData_input_cpp.txt";
	std::string TrainingData_output_file_path="TrainingData_output_cpp.txt";
	if(!load_TrainingData(TrainingData_input, TrainingData_input_file_path)){
		std::cerr<<"load TrainingData input failed"<<std::endl;
		return 0;
	}
	if(!load_TrainingData(TrainingData_output, TrainingData_output_file_path)){
		std::cerr<<"load TrainingData output failed"<<std::endl;
		return 0;
	}
	

	int output_dimension=TrainingData_output.rows(); // set 
	int num_InputUnits=TrainingData_input.rows(); // set the number of input units = dimension of input data 
	Matrix<int, 1, Dynamic> num_HiddenNeurons(1,2);
	num_HiddenNeurons << 7, 5;
	int num_OutputUnits=output_dimension; // set the number of output units.
	ffnn net(num_InputUnits, num_HiddenNeurons, num_OutputUnits);

	double deltaWeight=0.00001; // delta weight change
	double ffnn_error_threshold=1.0; // the threshold of training error
    double ffnn_error_iteration=ffnn_error_threshold+0.1; // the training error of each iteration, initial value should be larger than threshold (thus +0.1)
    std::vector<double> ffnn_error; // store the traing errors of all iterations
    int training_iteration=0; // current iteration number
    int training_iteration_threshold=50; // maximum number of iterations

    while((ffnn_error_iteration>ffnn_error_threshold)&&(training_iteration<training_iteration_threshold)){
    	ffnn_error_iteration=0;
    	training_iteration++;
    	std::cout<<training_iteration<<std::endl;
    	net.train_bp(TrainingData_input, TrainingData_output, deltaWeight);
    	simulate_output=net.simulate(TrainingData_input);
    	for(int i=0; i< num_OutputUnits; i++)
    		for(int j=0; j<TrainingData_input.cols(); j++)
    			ffnn_error_iteration+=0.5*(simulate_output(i,j)-TrainingData_output(i,j))*(simulate_output(i,j)-TrainingData_output(i,j));
    	ffnn_error.push_back(ffnn_error_iteration);
    	std::cout<<ffnn_error_iteration<<std::endl<<std::endl;
    }

	std::string save_Net_file_path="ffnn_net_cpp.txt";
	if(!net.save_Net(save_Net_file_path))
		std::cerr<<"save ffnn Net failed"<<std::endl;

	// save the output simulated with the trained rnn
	std::string save_results_file_path="simulate_ffnn_output_cpp.txt";
	if(!save_Matrix(simulate_output, save_results_file_path))
		std::cerr<<"save results failed"<<std::endl;

	
	//save the training errors
	std::string save_error_file_path="ffnn_training_error_cpp.txt";
	if(!save_vector<double>(ffnn_error, save_error_file_path))
		std::cerr<<"save error file failed"<<std::endl;

	//
	// string save_VerilogA_file_path="osc_ffnn.va";
	// if(!net.convert_ffnn_to_VerilogA(save_VerilogA_file_path))
	// 	std::cerr<<"convert_ffnn_to_VerilogA failed"<<std::endl;

	clock_t end_time=clock();
	std::cout<< "Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC<<" s"<<std::endl;

	return 0;
	}
}
