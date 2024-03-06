#ifndef SKLEARN_HPP
#define SKLEARN_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>

#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generate moon-shaped data similar to make_moons in scikit-learn
void make_moons(size_t n_samples, double noise, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-3.0, 3.0);  // Adjust range as needed

    // Generate data points
    X.clear();
    y.clear();
    for (size_t i = 0; i < n_samples; ++i) {
        double angle = dis(gen) * M_PI;
        double radius = 1.0 * dis(gen) + 1.0;
        double x = radius * cos(angle);
        double y_val = radius * sin(angle);
        X.push_back({ x, y_val });
        // Assign labels based on position
        y.push_back(y_val > sin(angle * 2.0) ? 1.0 : 0.0);
    }
}

// Write data to a file
void write_data_to_file(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                file << X[i][j] << " ";
            }
            file << y[i] << std::endl;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>> readDataFromFile(const std::string& filename) {
    // Vectors to store the data
    std::vector<std::vector<double>> data_X;
    std::vector<double> data_y;

    // Open the file for reading
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return { data_X, data_y };
    }

    // Read each line from the file
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row_X;
        double value_X;
        double value_y;

        // Read the first two values (X) from the line
        if (iss >> value_X) {
            row_X.push_back(value_X);
        }
        else {
            break; // End of file reached
        }

        if (iss >> value_X) {
            row_X.push_back(value_X);
        }
        else {
            break; // End of file reached
        }

        // Read the last value (y) from the line
        if (iss >> value_y) {
            data_X.push_back(row_X);
            data_y.push_back(value_y);
        }
        else {
            break; // End of file reached
        }
    }

    return { data_X, data_y };
}

// auxiliary functions

function<void(Data*)> build;
tuple<set<Data*>, set<tuple<Data*, Data*>>> trace(Data* root) {
	set<Data*> nodes;
	set<tuple<Data*, Data*>> edges;
	build = [&nodes, &edges](Data* v) {
		if (nodes.find(v) == nodes.end()) {
			nodes.insert(v);
			for (auto child : v->prev) {
				edges.insert(tuple<Data*, Data*>(child.get(), v));
				build(child.get());
			}
		}
	};
	build(root);
	return { nodes, edges };
}

Agraph_t* dot(Data* root, string format = "svg", string rankdir = "LR", string filename="dot.svg") {
	//format: png | svg | ...
	//rankdir : TB(top to bottom graph) | LR(left to right)

		// Initialize graph context
	GVC_t* gvc = gvContext();

	// Create a new directed graph
	char name[2]={'G','\0'};
	auto g = agopen(name, Agdirected, NULL);
	char* buffer = new char[rankdir.length() + 1];
	strcpy(buffer, rankdir.c_str());
	const char* namerank = "rankdir";
	const char* def = "";
	agsafeset(g, const_cast<char*>(namerank), buffer, const_cast<char*>(def));

	set<Data*> nodes;
	set<tuple<Data*, Data*>> edges;
	map<long, Agnode_t*> anodes;

	std::tie(nodes, edges) = trace(root);

	for (auto n : nodes) {
		// some horrible c-api crap
		char* name = new char[30];
		long ptrValue = reinterpret_cast<long>(n);
		snprintf(name, 30, "%ld", ptrValue);
		auto node = agnode(g, name, 1);
		anodes[(long)n] = node;
		char* lable = new char[30];
		sprintf(lable, "<data> %f | <grad> %f", n->data, n->grad);
		const char* label = "label";
		agset(node, const_cast<char*>("label"), lable);
		if (n->op != "") {
			// make a unique name out of op
			static int count = 0;
			char* name2 = new char[1000];
			strcpy(name2, n->op.c_str());
			name2[strlen(n->op.c_str())] = '\0';
			for (int i = strlen(name2); i < count; i++) {
				name2[i] = 64;
				name2[i + 1] = '\0';
			}
			count++;

			auto node2 = agnode(g, name2, 1);
			anodes[(long)n + (long)(n->op.c_str())] = node2;
			char* lable2 = new char[30];
			sprintf(lable2, "%s", n->op.c_str());
			const char* label_ = "label";
			agset(node2, const_cast<char*>(label_), lable2);
			auto e = agedge(g, node2, node, NULL, 1);
		}
	}

	for (tuple<Data*, Data*> edge : edges) {
		Data* n1 = get<0>(edge);
		Data* n2 = get<1>(edge);
		auto e = agedge(g, anodes[(long)n1], anodes[(long)n2 + (long)n2->op.c_str()], NULL, 1);
	}

	gvLayout(gvc, g, "dot");
	gvRenderFilename(gvc, g, format.c_str(), filename.c_str());

	// Free memory
	agclose(g);
	gvFreeContext(gvc);
	system(("start " + filename).c_str());

	return g;
}

void plotResults(MLP &model, std::vector<std::vector<double>>& X, std::vector<double>& y) {
	// Define plot parameters
	double h = 0.25;

	// Calculate decision boundary points and write to file
	double x_min = X[0][0], x_max = X[0][0];
	double y_min = X[0][1], y_max = X[0][1];

	for (const auto& point : X) {
		x_min = std::min(x_min, point[0]);
		x_max = std::max(x_max, point[0]);
		y_min = std::min(y_min, point[1]);
		y_max = std::max(y_max, point[1]);
	}

	x_min -= 1; // Adjusting x_min and y_min
	y_min -= 1;
	y_max += 1; // Adjusting x_min and y_min
	x_max += 1;

	// Create a grid of points over the input space
	int nx = (x_max - x_min) / h + 1;
	int ny = (y_max - y_min) / h + 1;
	double* xx = new double[nx * ny];
	double* yy = new double[nx * ny];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			xx[i * ny + j] = x_min + i * h;
			yy[i * ny + j] = y_min + j * h;
		}
	}

	// Evaluate the model on the grid of points and store the results in a binary array
	int* Z1 = new int[nx * ny];
	bool* Z = new bool[nx * ny];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			std::vector<Value> temp;
			temp.emplace_back(Value(xx[i * ny + j]));
			temp.emplace_back(Value(yy[i * ny + j]));
			Z1[i * ny + j] = model(temp)[0].data->data;
			Z[i * ny + j] = Z1[i * ny + j] < 0;
		}
	}

	// Write the data to a file that can be read by Gnuplot
	std::ofstream data_file("data.dat", std::ofstream::trunc);
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			data_file << xx[i * ny + j] << " " << yy[i * ny + j] << " " << Z[i * ny + j] << endl;
		}
		data_file << endl;
	}
	data_file.close();

	// Call Gnuplot and pass it the necessary commands to create the contour plot
	std::string command = "gnuplot plot.txt";
	system(command.c_str());
	system("start boundary.svg");

	// Clean up
	delete[] xx;
	delete[] yy;
	delete[] Z;
	delete[] Z1;
}

#endif
