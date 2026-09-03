#pragma once

#include "Matrix.hpp"
#include <optional>
#include <memory>
#include <map>
#include <string>


enum class TreeType { 
    Classification, 
    Regression 
};

class DecisionTree {
public:
    // constructors
    DecisionTree();
    DecisionTree(TreeType type=TreeType::Classification,size_t maxDepth=10, size_t minSamplesSplit = 2, size_t minSamplesLeaf = 1);

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
        double value = 0.0;
    };

    std::unique_ptr<Node> root;
    TreeType type;
    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    //build tree
    std::unique_ptr<Node> buildTree(const Matrix& X, const Matrix& y, size_t depth) const;

    struct Split {
        size_t featureIndex;
        double threshold;
        double impurityDecrease;
    };
    //tree splitting and leaf operations
    std::optional<Split> findBestSplit(const Matrix& X, const Matrix& y) const;
    double computeImpurity(const Matrix& y) const;
    double computeLeafValue(const Matrix& y) const;
    double computeGini(const Matrix& y) const;
    double majorityClass(const Matrix& y) const;
    double computeVariance(const Matrix& y) const;
    double meanValue(const Matrix& y) const;


    double predictSingle(const Matrix& X, size_t rowIndex, const Node* node) const;

    void printNode(const Node* node, size_t depth) const;
    size_t countDepth(const Node* node) const;
    size_t countLeaves(const Node* node) const;
};