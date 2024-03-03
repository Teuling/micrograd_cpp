# micrograd c++

c++ version of the famous micrograd implementaton in python, by andrej karpathy

the trace_graph and sanity tests can be found be found in tests.hpp
the neural net is in microgradc++.cpp 

### Installation

you'll have to modify the CMakeLists.txt's (both of them) to get to the needed header files/libraries for armadillo and graphviz

### Tracing / visualization

```cpp
auto x = Value(1.0);
auto y = (x * 2 + 1).relu();
y.backward();
draw_dot(y.data.get(), "svg", "LR", "dot1.svg");
```

![a very simple example](dot1.svg)

```cpp
auto n = Neuron(2);
vector<Value> x{ Value(1.0), Value(-2.0) };
auto y = n(x);
y.backward();
draw_dot(y.data.get(), "svg", "LR", "dot2.svg");
```

![a simple 2D neuron](dot2.svg)

### Training a neural net, some sample results

![microgradc++/data.txt](data_boundary.svg)
![make_moon1](sample1_boundary.svg)
![make_moon2](sample2_boundary.svg)
![make_moon3](sample3_boundary.svg)