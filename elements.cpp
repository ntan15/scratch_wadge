#include "elements.h"
#include "operators.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

void foo(int arg)
{
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  test_eigen();
}

// pass to eigen
void build_operators_C(int N, int Nq){

  build_operators_2D(N,Nq);
}

