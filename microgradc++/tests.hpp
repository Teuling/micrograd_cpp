#ifndef TEST_HPP
#define TEST_HPP

void testdot1() {
	// a very simple example
	auto x = Value(1.0);
	auto y = (x * 2 + 1).relu();
	y.backward();
	dot(y.data.get(), "svg", "LR", "dot1.svg");
}

void testdot2() {
	//a simple 2D neuron
	// no need to seed, is done in the neuron
	auto n = Neuron(2);
	vector<Value> x{ Value(1.0), Value(-2.0) };
	auto y = n(x);
	y.backward();
	dot(y.data.get(), "svg", "LR", "dot2.svg");
}

void test1()
{
	Value a(2);
	Value b(1);
	Value test1 = a + b;
	auto test2 = b + a;
	auto test3 = a + 4;
	auto test4 = 4 + a;
	cout << "test1.data:" << test1.data << endl;
	cout << "test2.data:" << test2.data << endl;
	cout << "test3.data:" << test3.data << endl;
	cout << "test4.data:" << test4.data << endl;

	auto test5 = b * a;
	auto test6 = a * 4;
	auto test7 = 4 * a;
	auto test8 = a * b;
	cout << "test5.data:" << test5.data << endl;
	cout << "test6.data:" << test6.data << endl;
	cout << "test7.data:" << test7.data << endl;
	cout << "test8.data:" << test8.data << endl;

	auto test9 = b / a;
	auto test10 = a / 4;
	auto test11 = 4 / a;
	auto test12 = a / b;
	cout << "test9.data:" << test9.data << endl;
	cout << "test10.data:" << test10.data << endl;
	cout << "test11.data:" << test11.data << endl;
	cout << "test12.data:" << test12.data << endl;

	auto test13 = b - a;
	auto test14 = a - 4;
	auto test15 = 4 - a;
	auto test16 = a - b;
	cout << "test13.data:" << test13.data << endl;
	cout << "test14.data:" << test14.data << endl;
	cout << "test15.data:" << test15.data << endl;
	cout << "test16.data:" << test16.data << endl;
}

void test2()
{
	auto a = Value(2.0);
	auto b = Value(3.0);
	auto c = a + b;
	c.backward();
	assert(c.data->data == 5);
	assert(a.data->grad == 1);
	assert(b.data->grad == 1);

	auto ci = Value(7.0);
	auto di = -ci;
	di.backward();
	assert(di.data->data == -7);
	assert(ci.data->grad == -1);

	auto d = Value(2.0);
	auto e = Value(3.0);
	auto f = d * e;
	f.backward();
	assert(f.data->data == 6.0);
	assert(d.data->grad == 3);
	assert(e.data->grad == 2);

	auto am = Value(2.0);
	auto bm = Value(3.0);
	auto cm = am - bm;
	cm.backward();
	assert(cm.data->data == -1);
	assert(am.data->grad == 1);
	assert(bm.data->grad == -1);

	auto ad = Value(16.0);
	auto bd = Value(4.0);
	auto cd = ad / bd;
	cd.backward();
	assert(cd.data->data == 4);
	assert(ad.data->grad == 0.25);
	assert(bd.data->grad == -1);

	auto ap = Value(16.0);
	auto bp = Value(4.0);
	ap += (ap + 1);
	ap += 1 + ap + (-bp);//-1
	ap.backward();
	assert(bp.data->data == 4);
	assert(ap.data->grad == 1);
	assert(bp.data->grad == -1);

	auto bpp = Value(4.0);
	auto bcc = bpp ^ 2;
	bcc.backward();
	assert(bcc.data->data == 16);
	assert(bpp.data->grad == 8);

	//tests grad additions
	auto av = Value(-4.0);
	auto bv = Value(2.0);
	auto cv = av + bv;
	auto dv = av * bv;
	auto dvv = cv * dv;
	dvv.backward();
	assert(dvv.data->data == 16);
	assert(bv.data->grad == 0);
	assert(av.data->grad == -12);
}

void test3()
{
	// pytorch not easily added to c++
	// so gradients were printed from micrograd
	// and checked in the debugger
	//ymg.data:-20.0
	//xmg.grad : 46.0
	auto x = Value(-4.0);
	auto z = 2 * x + 2 + x;
	auto q = z.relu() + z * x;
	auto h = (z * z).relu();
	auto y = h + q + q * x;
	y.backward();
	assert(y.data->data == -20);
	assert(x.data->grad == 46);
}

void test4()
{
	// pytorch not easily added to c++
	// so gradients were printed from micrograd
	// and checked in the debugger
	//gmg.data : 24.70408163265306
	// 
	//amg.grad : 138.83381924198252
	//bmg.grad : 645.5772594752186
	//cmg.grad:-6.941690962099126
	//dmg.grad : 6.941690962099126
	//emg.grad : -6.941690962099126
	//fmg.grad : 0.4958350687213661
	auto a = Value(-4.0);
	auto b = Value(2.0);
	auto c = a + b;
	auto d = a * b + (b ^ 3);
	c += (c + 1);
	c += 1 + c + (-a);//-1
	d += d * 2 + (b + a).relu();
	d += 3 * d + (b - a).relu();
	auto e = c - d;
	auto f = e ^ 2;
	auto g = f / 2.0;
	g = g + (10.0 / f);
	g.backward();
}

#endif