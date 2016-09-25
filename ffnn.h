///////////////////////////////////
// ffnn.h
// Huan Yu
///////////////////////////////////

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <string>

using namespace Eigen;

class ffnn{
public:

	int numInputUnits;
	Matrix<int, 1, Dynamic> numHiddenNeurons;
	int numHiddenLayers;
	int numOutputUnits;
	int numAllUnits;
	int numWeights;
	std::vector<int> weights_dest;
	std::vector<int> weights_source;
	std::vector<double> weights_value;

	ffnn(int num_InputUnits, Matrix<int, 1, Dynamic> num_HiddenNeurons, int num_OutputUnits);
	void activation_function_and_derivative(double s, double& act, double& act_d);
	void train_bp(const MatrixXd& TrainingData_input, const MatrixXd& TrainingData_output, double deltaWeight);
	MatrixXd simulate(const MatrixXd& input); // simulate the current net
	bool save_Net(std::string save_Net_file_path); // save net weights to file
	bool convert_ffnn_to_VerilogA(std::string save_VerilogA_file_path); 
};


bool load_TrainingData(MatrixXd& TrainingData, std::string TrainingData_file_path);
bool save_Matrix(const MatrixXd& Matrix_to_save, std::string save_Matrix_file_path); // save a Matrix into file

template <class T>
bool save_vector(const std::vector<T>& vec, std::string save_file_path){
	
	Matrix<T, Dynamic, Dynamic> mat_from_vec = Matrix<T, Dynamic, Dynamic>::Zero(1,vec.size());
	for (int n=0; n<vec.size(); n++){
		mat_from_vec(0,n)=vec[n];
	}
	
	std::ofstream file;
	file.open(save_file_path.c_str());
	if (!file.is_open())
	{
	  std::cerr << "Couldn't open file '" << save_file_path << "' for writing." << std::endl;
	  return false;
	}
	
	// file << std::fixed;
	file << mat_from_vec;
	file.close();
	
	return true;
	
}