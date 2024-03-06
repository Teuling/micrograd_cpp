// microgradc++.cpp : Defines the entry point for the application.
//
// the sanity checks and  microgradc++/tests.h

#include "microgradc++.hpp"
#include "engine.hpp"
#include "nn.hpp"
#include "sklearn.hpp"
#include "tests.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <set>

using namespace std;

std::vector<std::vector<double>> X;
std::vector<double> y;
double noise = 0.1;
vector<int> layers{ 16, 16, 1 };
// 2 - layer neural network
auto model = MLP(2, layers);

// loss function
pair<Value, Value> loss() {
	// no batches for now
	vector<Value> yb;
	std::for_each(y.begin(), y.end(), [&yb](double y) {
		yb.emplace_back(Value(y));
		});

	std::vector<std::vector<Value>> inputs;
	std::for_each(X.begin(), X.end(), [&inputs](std::vector<double> xb) {
		std::vector<Value> temp;
		for (auto& v : xb)
			temp.emplace_back(Value(v));
		inputs.emplace_back(temp);
		});

	// run model
	std::vector<Value> scores;
	for (auto& input : inputs) {
		auto result = model(input);
		scores.emplace_back(result[0]);
	}

	// calculate loss
	std::vector<std::pair<Value, Value>> zipped;

	// Use std::transform to zip yb and scores into zipped
	std::transform(yb.begin(), yb.end(), scores.begin(), std::back_inserter(zipped),
		[](Value a, Value b) { return std::make_pair(a, b); });

	//svm "max-margin" loss
	vector<Value> losses;
	std::for_each(zipped.begin(), zipped.end(), [&losses](pair<Value, Value> zip) {
		//yi*scorei
		auto val = (1 + (-zip.first) * zip.second).relu();
		losses.emplace_back(val);
		});

	Value data_loss{ 0 };
	for (auto& l : losses) {
		data_loss += l;
	}
	data_loss = data_loss / losses.size();

	// L2 regularization
	double alpha = 1e-4;
	Value reg_loss{ 0 };
	for (auto& l : model.parameters()) {
		reg_loss += l * l;
	}
	reg_loss = reg_loss * alpha;
	auto total_loss = data_loss + reg_loss;

	// also get accuracy
	vector<Value> accuracy;
	std::for_each(zipped.begin(), zipped.end(), [&accuracy](pair<Value, Value> zip) {
		//yi, scorei
		auto val = (zip.first.data->data > 0) == (zip.second.data->data > 0);
		accuracy.emplace_back(val);
		});
	Value sum_ack{ 0 };
	for (auto& l : accuracy) {
		sum_ack += l;
	}
	return { total_loss , sum_ack / accuracy.size() };
}

void testNN() {
	string filename{ "data.txt" };
	make_moons(100, noise, X, y);
	write_data_to_file(X, y, filename);
	for (auto& y_ : y) {
		y_ = y_ * 2 - 1;
	}

	cout << model << endl;
	cout << "number of parameters:" << model.parameters().size() << endl;
	pair<Value, Value> losti;
	for (int k = 0; k < 100; k++) {
		// forward
		losti = loss();
		// backward
		model.zero_grad();
		losti.first.backward();

		//update(sgd)
		double learning_rate = 1.0 - 0.9 * k / 100;
		for (auto& p : model.parameters()) {
			p.data->data -= learning_rate * p.data->grad;
		}
		if (k % 1 == 0)
			cout << "step " << k << ",loss " << losti.first.data->data << ", " << "accuracy " << losti.second.data->data * 100 << " % " << endl;
	}
	plotResults(model, X, y);
}

int main()
{
	//test1();
	//test2();
	//test3();
	//test4();
	//testdot1();
	//testdot2();
	testNN();
	return 0;
}