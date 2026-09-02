#pragma once

#include "Matrix.hpp"
#include <optional>
#include <memory>
#include <map>
#include <string>


class DecisionTree {
public:
    // constructors
    DecisionTree();
    DecisionTree(size_t maxDepth, size_t minSamplesSplit = 2, size_t minSamplesLeaf = 1);

    // training
    void fit(const Matrix& X, const Matrix& y);

    // predictions
    Matrix predict(const Matrix& X) const;

    // inspection
    void showTree() const;
    size_t getDepth() const;
    size_t getLeafCount() const;

private:
    //tree structure
    struct Node {
        bool isLeaf = false;

        // internal node
        size_t splitFeatureIndex = 0;
        double splitThreshold = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        // leaf node
        double predictedClass = 0.0;
    };

    std::unique_ptr<Node> root;

    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;

    std::unique_ptr<Node> buildTree(const Matrix& X, const Matrix& y, size_t depth) const;
    struct Split {
        size_t featureIndex;
        double threshold;
        double impurityDecrease;
    };
    std::optional<Split> findBestSplit(const Matrix& X, const Matrix& y) const;
    double computeGini(const Matrix& y) const;
    double majorityClass(const Matrix& y) const;


    double predictSingle(const Matrix& row, const Node* node) const;


    void printNode(const Node* node, size_t depth) const;
    size_t countDepth(const Node* node) const;
    size_t countLeaves(const Node* node) const;
};