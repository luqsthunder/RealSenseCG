#pragma once

#include <array>
#include <vector>

namespace rscg {

template<typename T, unsigned dim = 3>
class KDTree {

  struct Node {
    Node *left, right;
    std::array<T, dim> data;
  };

  Node root;
public:
  KDTree();
  KDTree(const std::vector<T> &vec);
  
  void insert(const T &t);
  
  void remove(const T &val);

  const std::vector<T>& nearestNeighborhood(const T &val, T range);

  T find(const T &val);
};

}