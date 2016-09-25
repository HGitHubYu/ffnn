///////////////////////////////////
// ffnn.cpp
// Huan Yu
///////////////////////////////////

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include "ffnn.h"

using namespace Eigen;

ffnn::ffnn(int num_InputUnits, Matrix<int, 1, Dynamic> num_HiddenNeurons, int num_OutputUnits){
	numInputUnits=num_InputUnits;
	numHiddenNeurons=num_HiddenNeurons;
	numHiddenLayers= num_HiddenNeurons.cols();
	numOutputUnits=num_OutputUnits;
	numAllUnits=1+numInputUnits+numHiddenNeurons.sum()+numOutputUnits;

	int index=0;
	int dst, src;
	double rand_num;
	double range_magnitude=1.0;
	time_t t;
	srand((unsigned) time(&t));

	// set network weights
	// set weights from input layer to the 1st hidden layer
	for (dst=numInputUnits+1; dst<=numInputUnits+numHiddenNeurons(0,0); dst++){
		for(src=0; src<=numInputUnits; src++){
			weights_dest.push_back(dst);
			weights_source.push_back(src);
			index++;
		}
	}

	// set weights from hidden layers to next hidden layer
	if(numHiddenLayers>1){
		for(int layer=2; layer<=numHiddenLayers; layer++){
			for (dst=numInputUnits+numHiddenNeurons.topLeftCorner(1,layer-1).sum()+1; dst<=numInputUnits+numHiddenNeurons.topLeftCorner(1,layer).sum(); dst++){
				src=0;
				weights_dest.push_back(dst);
				weights_source.push_back(src);
				index++;
				for(src=numInputUnits+numHiddenNeurons.topLeftCorner(1,layer-1).sum()-numHiddenNeurons(0,layer-2)+1; src<=numInputUnits+numHiddenNeurons.topLeftCorner(1,layer-1).sum(); src++){
						weights_dest.push_back(dst);
						weights_source.push_back(src);
						index++;
				}
			}
		}
	}

	// set weights from hidden layer to output layer. 
	// There is also a constant bias unit connected to the output layer
	for(dst=numAllUnits-numOutputUnits; dst<=numAllUnits-1; dst++){
		src=0;
		weights_dest.push_back(dst);
		weights_source.push_back(src);
		index++;
		for(src=numAllUnits-numOutputUnits-numHiddenNeurons(0,numHiddenLayers-1);src<=numAllUnits-numOutputUnits-1;src++){
			weights_dest.push_back(dst);
			weights_source.push_back(src);
			index++;
		}
	}

	numWeights=index;
	weights_dest.push_back(-1); //used as a sign if index goes out of range



	//initialize weights values
	int load_weights=0; // if load weight values from file
	if(load_weights==1){
		// load weight values from file
		std::string save_weights_value_file_path="ffnn_weights_value_cpp.txt";
		std::ifstream input(save_weights_value_file_path.c_str());
		if (input.fail())
		{
		  std::cerr << "ERROR. Cannot find file '" << save_weights_value_file_path << "'." << std::endl;
		}

		std::string line;
		double d;
		getline(input, line);
		std::stringstream input_line(line);
		while (!input_line.eof())
		{
		  input_line >> d;
		  weights_value.push_back(d);
		}
		input.close();

		if (weights_value.size()!=numWeights)
		{
		  std::cerr << "ERROR. numWeights incorrect." << std::endl
		  			<< "weights_value.size()=" << weights_value.size()<<std::endl
		  			<< "numWeights="<<numWeights << std::endl<< std::endl;
		}

	}
	else{
		// initialize weight values randomly
		for (index=0; index<numWeights; index++){
			rand_num = (rand()%1000)*0.001;
			weights_value.push_back(rand_num*2.0*range_magnitude-range_magnitude);
		}

	}

}



void ffnn::activation_function_and_derivative(double s, double& act, double& act_d){
	int opt=2;

	if (opt==1){
		// option 1: sigmoid function
		act = 1.0 / (1+exp(-s));
		act_d = act * (1-act);
	}
	else if(opt==2){
		// option 2: tanh
		act = (exp(s)-exp(-s)) / (exp(s)+exp(-s));
		act_d = 1- pow(act, 2);
	}
	else{
		// default option: sigmoid function
		act = 1.0 / (1+exp(-s));
		act_d = act * (1-act);
	}

}



void ffnn::train_bp(const MatrixXd& TrainingData_input, const MatrixXd& TrainingData_output, double deltaWeight){
	int input_dimension=TrainingData_input.rows();
	int input_length=TrainingData_input.cols();
	int output_dimension=TrainingData_output.rows();
	int output_length=TrainingData_output.cols();

	int firstStep=0;
	int lastStep=input_length-1; 

	MatrixXd ACT= MatrixXd::Zero(numAllUnits, lastStep+1);
	ACT.row(0).setOnes(); // set constant input unit
	ACT.block(1, firstStep, numInputUnits, input_length)=TrainingData_input;
	MatrixXd ACTD= MatrixXd::Zero(numAllUnits, lastStep+1); //derivatives: unit output / unit input

	MatrixXd step_OutputsError_drv_w[lastStep+1]; // ffnn derivatives: step_OutputsError/weights
	for (int step=0; step<=lastStep; step++){
		step_OutputsError_drv_w[step]= MatrixXd::Zero(numWeights, output_dimension);
	}

	MatrixXd total_Error_drv_w= MatrixXd::Zero(1, numWeights); // ffnn derivatives: total_Error/weights


	// main loop
	for(int step=firstStep; step<=lastStep; step++){
		// feed forward
		int nextDest = weights_dest[0];
		int Weight_Index=0;
		double unit_input_sum=0;
		int dest;

		while (Weight_Index<numWeights){
			unit_input_sum=0;
			dest=nextDest;
			while(dest==nextDest){
				unit_input_sum+=weights_value[Weight_Index]*ACT(weights_source[Weight_Index], step);
				Weight_Index++;
				nextDest=weights_dest[Weight_Index];
			}
			if(dest> numInputUnits+numHiddenNeurons.sum()){
				// output units: derivative = 1
				ACT(dest, step)=unit_input_sum;
				ACTD(dest, step)=1;
			}
			else{
				// hidden units: use activation function
				activation_function_and_derivative(unit_input_sum, ACT(dest, step), ACTD(dest, step));
			}
		}

		// back propagation
		MatrixXd ffnnOutputs_drv_w = MatrixXd::Zero(numWeights, numOutputUnits); // ffnn derivatives: ffnnOutput/weights
		MatrixXd ffnnOutputs_drv_ACT = MatrixXd::Zero(numAllUnits, numOutputUnits); // ffnn derivatives: ffnnOutput/units_outputs
		ffnnOutputs_drv_ACT.block(numInputUnits+numHiddenNeurons.sum()+1, 0, numOutputUnits, numOutputUnits).setIdentity();
		
		int source;
		nextDest=weights_dest[numWeights-1];
		Weight_Index=numWeights-1;
		while(Weight_Index>=0){
			dest=nextDest;
			while (dest==nextDest){
				source= weights_source[Weight_Index];
				ffnnOutputs_drv_w.row(Weight_Index)=ffnnOutputs_drv_ACT.row(dest)*ACTD(dest,step)*ACT(source, step);

				// calculate ffnn derivatives: ffnnOutput/unitOutputs
				ffnnOutputs_drv_ACT.row(source)+=ffnnOutputs_drv_ACT.row(dest)*ACTD(dest,step)*weights_value[Weight_Index];

				// get next
				Weight_Index--;
				if (Weight_Index<0) break;
				nextDest=weights_dest[Weight_Index];

			}
		}

		// calculate current step OutputError derivatives: step_OutputError/weights
		for(int UI=0; UI<numOutputUnits; UI++){
			step_OutputsError_drv_w[step].col(UI)=ffnnOutputs_drv_w.col(UI)*(ACT(UI+numAllUnits-numOutputUnits,step)-TrainingData_output(UI,step));
		}

	}

	// calculate total OutputError derivatives: total_OutputError/weights
	for(int WI=0; WI<numWeights; WI++){
		for(int UI=0; UI<numOutputUnits; UI++){
			for(int st=firstStep; st<=lastStep; st++){
				total_Error_drv_w(0,WI)+=step_OutputsError_drv_w[st](WI,UI);
			}
		}
	}

	// adjust weights
	for(int WI=0; WI<numWeights;WI++){
		weights_value[WI]-=total_Error_drv_w(0, WI)*deltaWeight;
	}

}



MatrixXd ffnn::simulate(const MatrixXd& input){
	int input_dimension=input.rows();
	int input_length=input.cols();

	int firstStep=0;
	int lastStep=input_length-1; 

	MatrixXd ACT= MatrixXd::Zero(numAllUnits, lastStep+1);
	ACT.row(0).setOnes(); // set constant input unit
	ACT.block(1, firstStep, numInputUnits, input_length)=input;

	// main loop
	for(int step=firstStep; step<=lastStep; step++){
		// feed forward
		int nextDest = weights_dest[0];
		int Weight_Index=0;
		double unit_input_sum=0;
		int dest;
		double actd;

		while (Weight_Index<numWeights){
			unit_input_sum=0;
			dest=nextDest;
			while(dest==nextDest){
				unit_input_sum+=weights_value[Weight_Index]*ACT(weights_source[Weight_Index], step);
				Weight_Index++;
				nextDest=weights_dest[Weight_Index];
			}
			if(dest> numInputUnits+numHiddenNeurons.sum()){
				// output units
				ACT(dest, step)=unit_input_sum;
			}
			else{
				// hidden units: use activation function
				activation_function_and_derivative(unit_input_sum, ACT(dest, step), actd);
			}
		}
	}
	MatrixXd output_sim= ACT.block(numInputUnits+numHiddenNeurons.sum()+1, firstStep, numOutputUnits, input_length);
	return output_sim;
}



bool ffnn::save_Net(std::string save_Net_file_path){
	
	MatrixXd w_value=MatrixXd::Zero(1,numWeights);
	for (int n=0; n<numWeights; n++){
		w_value(0,n)=weights_value[n];
	}
	
	std::ofstream file;
	file.open(save_Net_file_path.c_str());
	if (!file.is_open())
	{
	  std::cerr << "Couldn't open file '" << save_Net_file_path << "' for writing." << std::endl;
	  return false;
	}
	
	// file << std::fixed;
	file << w_value;
	file.close();
	
	return true;
	
}



bool ffnn::convert_ffnn_to_VerilogA(std::string save_VerilogA_file_path){

	std::ofstream file;
	file.open(save_VerilogA_file_path.c_str());
	if (!file.is_open())
	{
	  std::cerr << "Couldn't open file '" << save_VerilogA_file_path << "' for writing." << std::endl;
	  return false;
	}

	file<<"`include \"disciplines.vams\""<<std::endl
		<<std::endl
		<<"module osc(inp_0";
	for(int n=1;n<=numInputUnits; n++){
		file<<", inp_"<<n;
	}
	for(int n=1;n<=numOutputUnits; n++){
		file<<", outp_"<<n;
	}
	file<<");"<<std::endl;

	// port declearation 
	file<<"inout inp_0";
	for(int n=1;n<=numInputUnits; n++){
		file<<", inp_"<<n;
	}
	for(int n=1;n<=numOutputUnits; n++){
		file<<", outp_"<<n;
	}
	file<<";"<<std::endl;

	file<<"electrical inp_0";
	for(int n=1;n<=numInputUnits; n++){
		file<<", inp_"<<n;
	}
	for(int n=1;n<=numOutputUnits; n++){
		file<<", outp_"<<n;
	}
	file<<";"<<std::endl
		<<std::endl;

	file<<"real ACT[0:"<<numAllUnits-1<<"];"<<std::endl
		<<"real weights_value[0:"<<numWeights-1<<"] = {"<<weights_value[0];
	for(int n=1;n<numWeights; n++){
		file<<", "<<weights_value[n];
	}	
	file<<"};"<<std::endl
		<<"real unit_input_sum;"<<std::endl
		<<std::endl;

	file<<"integer weights_source[0:"<<numWeights-1<<"] = {"<<weights_source[0];
	for(int n=1;n<numWeights; n++){
		file<<", "<<weights_source[n];
	}	
	file<<"};"<<std::endl
		<<"integer weights_dest[0:"<<numWeights<<"] = {";
	for(int n=0;n<numWeights; n++){
		file<<weights_dest[n]<<", ";
	}	
	file<<"-1};"<<std::endl
		<<"integer Weight_Index, dest, nextDest;"<<std::endl
		<<std::endl;

	file<<"	analog begin"<<std::endl;

	for(int n=0;n<=numInputUnits; n++){
		file<<"		ACT["<<n<<"] = V(inp_"<<n<<");"<<std::endl;
	}

	file<<"		Weight_Index=0;"<<std::endl
		<<"		nextDest="<<weights_dest[0]<<";"<<std::endl
		<<"		while (Weight_Index<"<<numWeights<<") begin"<<std::endl
		<<"			unit_input_sum=0;"<<std::endl
		<<"			dest=nextDest;"<<std::endl
		<<"			while(dest==nextDest) begin"<<std::endl
		<<"				unit_input_sum=unit_input_sum+weights_value[Weight_Index]*ACT[weights_source[Weight_Index]];"<<std::endl
		<<"				Weight_Index=Weight_Index+1;"<<std::endl
		<<"				nextDest=weights_dest[Weight_Index];"<<std::endl
		<<"			end"<<std::endl
		<<"			if(dest> "<<numInputUnits+numHiddenNeurons.sum()<<") begin"<<std::endl
		<<"				ACT[dest]=unit_input_sum;"<<std::endl
		<<"			end else begin"<<std::endl
		<<"				ACT[dest]= (exp(unit_input_sum)-exp(-unit_input_sum)) / (exp(unit_input_sum)+exp(-unit_input_sum));"<<std::endl
		<<"			end"<<std::endl
		<<"		end"<<std::endl;

	for(int n=1;n<=numOutputUnits; n++){
		file<<"		V(outp_"<<n<<") <+ ACT["<<n+numAllUnits-1-numOutputUnits<<"];"<<std::endl;
	}

	file<<"	end"<<std::endl
		<<std::endl
		<<"endmodule";

	file.close();
	return true;

}



bool load_TrainingData(MatrixXd& TrainingData, std::string TrainingData_file_path){
	////////////////////
	// load data
	  std::ifstream input(TrainingData_file_path.c_str());
	  if (input.fail())
	  {
	    std::cerr << "ERROR. Cannot find file '" << TrainingData_file_path << "'." << std::endl;
	    TrainingData = MatrixXd(0,0);
	    return false;
	  }

	  std::string line;
	  double d;
	  
	  std::vector<double> v;
	  int n_rows = 0;

	  while (getline(input, line))
	  {
	    ++n_rows;
	    std::stringstream input_line(line);
	    while (!input_line.eof())
	    {
	      input_line >> d;
	      v.push_back(d);
	    }
	  }
	  input.close();
	  
	  int n_cols = v.size()/n_rows;
	  TrainingData = MatrixXd(n_rows,n_cols);
	  
	  for (int i=0; i<n_rows; i++)
	    for (int j=0; j<n_cols; j++)
	      TrainingData(i,j) = v[i*n_cols + j];

	  std::cout<<"TrainingData size is "<<n_rows<<"*"<<n_cols<<std::endl<<std::endl;
	  return true;

}




bool save_Matrix(const MatrixXd& output, std::string save_results_file_path){
	std::ofstream file;
	file.open(save_results_file_path.c_str());
	if (!file.is_open())
	{
	  std::cerr << "Couldn't open file '" << save_results_file_path << "' for writing." << std::endl;
	  return false;
	}
	
	// file << std::fixed;
	file << output;
	file.close();
	
	return true;
}




