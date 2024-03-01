#pragma once
#include "engine.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <map>

using namespace std;

struct Module
{
	void zero_grad() {
        for (auto& parameter : parameters()) {
            parameter.data->grad = 0;
        }
	}
    virtual vector<Value> parameters()  {
		return vector<Value>();
	}
};

struct Neuron : Module {
    vector<Value> w;
    Value b;
    bool nonlin;
    
    Neuron(int nin ,bool nonlin = true) : 
        w(nin),
        b(0),
        nonlin(nonlin) {
        dice(nin,-1,1);
    }
    Value operator()(vector<Value> x)  {
        Value act(0);
        for (auto i = 0; i < w.size(); i++) {
            act = act + x[i] * w[i];
        }
        act = act + b;
        return nonlin ? act.relu() : act;
    }
    vector<Value> parameters() override {
        auto ret = vector<Value>(w.begin(), w.end());
        ret.emplace_back(b);
        return ret;
    }
    friend ostream& operator<<(ostream& os, const Neuron& obj) {
        os << (obj.nonlin ? "Relu" : "Linear") << "Neuron(" << obj.w.size() << ")";
        return os;
    }
private:
    void dice(int nin, int low, int high)
    {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<> dis(low, high);
        for (int i = 0; i < nin; ++i) {
            w[i].data->data =dis(gen);
        }
    }
};

struct Layer : Module {
    vector<Neuron> neurons;
    Layer(int nin, int nout, bool kwargs)  {
        for (int i = 0; i < nout; i++) {
            neurons.emplace_back(Neuron(nin, kwargs/*[it - neurons.begin()]*/));
        }
    }
    auto operator()(vector<Value> x) {
        vector<Value> out;
        for (int i = 0; i < neurons.size(); i++) {
            out.emplace_back(neurons[i](x));
        }
        return out;
    }
    vector<Value> parameters() override {
        vector<Value> ret;
        std::for_each(neurons.begin(), neurons.end(), [&ret](Neuron& neuron) {
            auto params = neuron.parameters();
            ret.insert(ret.end(), params.begin(), params.end());
            });
        return ret;
    }
    friend ostream& operator<<(ostream& os, const Layer& obj) {
        os << "Layer of [";
        for (int i = 0; i < obj.neurons.size()-1; i++) {
            os << obj.neurons[i] << ", ";
        }
        os << obj.neurons[obj.neurons.size() - 1];
        os << "]";
        return os;
    }
};

struct MLP : Module {
    vector<Layer> layers;
    MLP(int nin, vector<int> nouts) {
        auto sz = nouts;
        sz.insert(sz.begin(), nin);
        for (int i = 0; i < nouts.size(); i++) {
            layers.emplace_back(Layer(sz[i], sz[i + 1], i != (nouts.size() - 1)));
        }
    }
    auto operator()(vector<Value> x) {
        for (auto& layer : layers)
            x = layer(x);
        return x;
    }
    std::vector<Value> parameters() override {
        std::vector<Value> ret;
        std::for_each(layers.begin(), layers.end(), [&ret](Layer& layer) {
            auto params = layer.parameters();
            ret.insert(ret.end(), params.begin(), params.end());
            });
        return ret;
    }
    friend ostream& operator<<(ostream& os, const MLP& obj) {
        os << "MLP of[";
        for (int i = 0; i < obj.layers.size() - 1; i++) {
            os << obj.layers[i] << ", ";
        }
        os << obj.layers[obj.layers.size() - 1];
        os << "]";
        return os;
    }
};
