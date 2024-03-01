#pragma once

#include <iostream>
#include <functional>
#include <set>
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>

using namespace std;

struct Data {
	double data{ 0 };
	double grad{ 0 };
	std::function<void()> _backward;
	set<std::shared_ptr<Data>> prev;
	string op;
};

struct Value {
	std::shared_ptr<Data> data;

	Value() : Value(0) {
	}

	Value(double dat, vector<std::shared_ptr<Data>> children = {}, string op = "") : 
		data(std::make_shared<Data>()) {
		data->data = dat;
		data->grad = 0;
		data->_backward = []() {};
		data->op = op;
		for (const auto& child : children) {
			data->prev.insert(child);
		}
	}
	void build_topo(vector<std::shared_ptr<Data>> &topo, 
					set<std::shared_ptr<Data>>&visited, 
					std::shared_ptr<Data> v) {
		if (visited.find(v) == visited.end()) {
			visited.insert(v);
			for (auto child : v->prev)
				build_topo(topo, visited, child);
			topo.push_back(v);
		}
	}
	void backward() {
		vector<std::shared_ptr<Data>> topo;
		set<std::shared_ptr<Data>> visited;
		build_topo(topo, visited, this->data);
		this->data->grad = 1.0;
		for (auto v = topo.rbegin(); v != topo.rend(); ++v) {
			(*v)->_backward();
		}
	}
// copy/assignment constructors
	Value& operator=(Value& other) {
		if (this != &other) {
			this->data = other.data;
			this->data->_backward = other.data->_backward;
		}
		return *this;
	}
	Value& operator=(Value&& other) {
		this->data = other.data;
		this->data->_backward = other.data->_backward;
		return *this;
	}
	Value(const Value& other) : 
		data(other.data)
	{
	}
// +operator
	Value operator+(Value& other) {
		Value out(data->data + other.data->data, { data, other.data }, "+");
		out.data->_backward = [self = this->data, out = out.data, other = other.data]() {
			self->grad += out->grad;
			other->grad += out->grad;
		};
		return out;
	}
	Value operator+(Value&& other) {
		Value out(data->data + other.data->data, { data, other.data }, "+");
		out.data->_backward = [self = this->data, other = other.data, out = out.data]() {
			self->grad += out->grad;
			other->grad += out->grad;
		};
		return out;
	}
	Value operator+(double other_) {
		auto other = make_shared<Value>(other_, std::vector<std::shared_ptr<Data>>{}, "c");
		other->data->_backward = []() {};
		Value out(data->data + other->data->data, { data, other->data }, "+");
		out.data->_backward = [self=this->data, other = other->data, out=out.data]() {
			self->grad += out->grad;
			other->grad += out->grad;
		};
		return out;
	}
	template<typename U>
	friend std::enable_if_t<!std::is_same_v<U, Value>, Value> operator+(U left, Value& right);
// *operator
	Value operator*(Value other) {
		Value out(data->data * other.data->data, { this->data, other.data }, "*");
		out.data->_backward = [self=this->data, other = other.data, out = out.data]() {
			self->grad += other->data * out->grad;
			other->grad += self->data * out->grad;
		};
		return out;
	}
	Value operator*(double other_) {
		auto other = make_shared<Value>(other_, std::vector<std::shared_ptr<Data>>{}, "c");
		other->data->_backward = []() {};
		Value out(data->data * other_, { data, other->data }, "*");
		out.data->_backward = [self=this->data, other = other->data, out=out.data]() {
			self->grad += other->data * out->grad;
			other->grad += self->data * out->grad;
		};
		return out;
	}
	template<typename U>
	friend std::enable_if_t<!std::is_same_v<U, Value>, Value> operator*(U left, Value& right);
// ^operator
	Value operator^(double other) {
		Value out(pow(data->data, other), { data }, "^");

		out.data->_backward = [self=this->data, other, out=out.data]() {
			self->grad += other*pow(self->data, other-1)*out->grad;
		};
		return out;
	}
// += operator
	Value operator+=(Value other) {
		auto newData = make_shared<Data>(*data);
		newData->op = "+=";
		newData->data = data->data + other.data->data;
		newData->_backward = [prev = data, cur = newData, other = other.data]() {
			prev->grad += cur->grad;
			other->grad += cur->grad;
		};
		newData->prev.clear();
		newData->prev.emplace(other.data);
		newData->prev.emplace(data);
		data = newData;
		return *this;
	}
	Value operator+=(double other) {
		auto otherData = make_shared<Data>();
		otherData->_backward = []() {};
		otherData->data = other;

		auto newData = make_shared<Data>(*data);
		newData->op = "+=";
		newData->data = data->data + otherData->data;
		newData->_backward = [prev = data, cur = newData, other = otherData]() {
			prev->grad += cur->grad;
			other->grad += cur->grad;
		};
		newData->prev.clear();
		newData->prev.emplace(otherData);
		newData->prev.emplace(data);
		data = newData;
		return *this;
	}
// ReLu 
	Value relu() {
		Value out(max(0.0,data->data), { data }, "ReLu");
		out.data->_backward = [self = this->data,  out=out.data]() {
			self->grad += (out->data > 0) ? out->grad : 0;
		};
		return out;
	}
// unary negative
	Value operator-() {
		return *this * -1;
	}
// minus
	Value operator-(double other_) {
		return *this + (-other_);
	}
	Value operator-(Value& other_) {
		return *this + -(other_);
	}
	template<typename U>
	friend std::enable_if_t<!std::is_same_v<U, Value>, Value> operator-(U left, Value& right);
// div
	Value operator/(double other_) {
		return *this * (pow(other_,-1));
	}
	Value operator/(Value& other_) {
		return *this * (other_^-1);
	}
	template<typename U>
	friend std::enable_if_t<!std::is_same_v<U, Value>, Value> operator/(U left, Value& right);
};
template<typename U>
std::enable_if_t<!std::is_same_v<U, Value>, Value> operator+(U left,  Value& right) {
	return right+(left);
}
template<typename U>
std::enable_if_t<!std::is_same_v<U, Value>, Value> operator*(U left, Value& right) {
	return right*(left);
}
template<typename U>
std::enable_if_t<!std::is_same_v<U, Value>, Value> operator-(U left, Value& right) {
	return left + (-right);
}
template<typename U>
std::enable_if_t<!std::is_same_v<U, Value>, Value> operator/(U left, Value& right) {
	return left * (right^-1);
}

