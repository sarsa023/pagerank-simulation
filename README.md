# PageRank Algorithm Simulation

A Python implementation and numerical study of the **PageRank algorithm**, exploring three different approaches to computing page rankings from a directed web graph.

The project uses NumPy for matrix operations and numerical computation, and Matplotlib for visualizing the resulting page rankings.

## Overview

PageRank is an algorithm used to estimate the relative importance of web pages based on their incoming links.

This project implements three approaches to computing PageRank:

1. **Linear System Method:** Computes the PageRank vector by solving a system of linear equations.
2. **Eigenvector Method:** Computes the PageRank vector as the dominant eigenvector of the Google matrix.
3. **Iterative Method:** Approximates the PageRank vector through repeated matrix-vector multiplication until convergence.

## Implementation

The implementation consists of two main components:

- `functions.py`: Contains the PageRank algorithms and supporting functions.
- `main.py`: Constructs example adjacency matrices, runs the three methods, compares their results, processes an external web graph, and visualizes the rankings.

### Matrix Preprocessing

The adjacency matrix is transformed into a column-stochastic matrix by:

- Handling dangling nodes (pages with no outgoing links).
- Normalizing each column so that its entries sum to 1.

### PageRank Computation

The implementation uses a default damping factor of **0.85**.

The iterative method uses an L1-norm convergence threshold of \(10^{-6}\), with a maximum iteration limit of 1000.

## Experiments

The project includes the following experiments:

- Constructing and preprocessing an adjacency matrix using a four-page example.
- Computing PageRank using all three methods.
- Comparing the numerical results to verify agreement within a tolerance of \(10^{-6}\).
- Ranking pages from highest to lowest PageRank.
- Reading a web graph from an external text file, constructing its adjacency matrix, and computing its PageRank values.

## Visualization

The project generates visualizations of the resulting PageRank distributions using:

- A pie chart for the four-page example.
- A bar chart for the externally supplied web graph.

## Technologies

- Python
- NumPy
- Matplotlib

## Getting Started

### Requirements

Python 3 and the dependencies listed in `requirements.txt`.

### Installation

Clone the repository:

```bash
git clone https://github.com/[username]/pagerank-simulation.git
cd pagerank-simulation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python main.py
```

## Repository Structure

```text
pagerank-simulation/
├── main.py
├── functions.py
├── web_pages.txt
├── requirements.txt
└── README.md
```

## Academic Context

Developed as part of university coursework at the University of Jordan.
