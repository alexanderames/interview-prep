# Interview Prep

## Dijkstra's Algorithm

**Interview Question:** How would you find the shortest path between two nodes in a weighted graph?

**Relevance to Financial Services:**

- Optimal Routing for Transactions
  - Financial services often need to route transactions through a network of banks or financial institutions. Finding the shortest path ensures that transactions are processed quickly and efficiently, minimizing delays and costs.
- Risk Management:
  - In risk management, identifying the shortest path can help in determining the quickest way to mitigate risks or transfer assets in response to market changes.
- Network Analysis:
  - Financial networks can be modeled as graphs where nodes represent entities (banks, accounts) and edges represent transactions or relationships. Finding the shortest path helps in analyzing the flow of money and detecting potential bottlenecks or vulnerabilities.

### Code Examples

#### pseudo

```shell
initialize a priority queue with the starting node and distance 0
initialize a dictionary to store distances from the start node to all other nodes, set all distances to infinity except the start node which is 0

while the priority queue is not empty:
    extract the node with the smallest distance from the priority queue
    for each neighbor of the current node:
        calculate the distance to the neighbor through the current node
        if this new distance is smaller than the previously known distance:
            update the distance to this neighbor
            add the neighbor to the priority queue with the updated distance

return the dictionary of distances
```

#### Python

```python
import heapq

def dijkstra(graph, start):
    heap = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances

# Example usage:
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
print(dijkstra(graph, 'A'))  # Outputs shortest paths from 'A'
```

#### Ruby

```ruby
require 'priority_queue'

def dijkstra(graph, start)
  distances = Hash.new(Float::INFINITY)
  distances[start] = 0
  pq = PriorityQueue.new
  pq[start] = 0

  until pq.empty?
    current_node, current_distance = pq.delete_min

    graph[current_node].each do |neighbor, weight|
      distance = current_distance + weight
      if distance < distances[neighbor]
        distances[neighbor] = distance
        pq[neighbor] = distance
      end
    end
  end

  distances
end

# Example usage:
graph = {
  'A' => {'B' => 1, 'C' => 4},
  'B' => {'A' => 1, 'C' => 2, 'D' => 5},
  'C' => {'A' => 4, 'B' => 2, 'D' => 1},
  'D' => {'B' => 5, 'C' => 1}
}
puts dijkstra(graph, 'A').inspect  # Outputs shortest paths from 'A'
```


## BFS (Breadth-First Search)

**Breadth-First Nature:**
BFS explores nodes (or states) level by level. In a maze, imagine a ripple or wave expanding outward from the start. Every cell at distance 1 is visited first, then all cells at distance 2, and so on.

**Shortest Path Guarantee:**
In unweighted graphs (or grids/mazes where every move costs the same), the first time you reach the target, you’ve found the shortest path. This is why BFS is often the go-to algorithm for these problems.

### Code examples

First, let's learn using pseudocode so that we understand the overall concept

#### pseudo

```shell
initialize a queue with the starting node/state
mark the starting node/state as visited

while the queue is not empty:
    current = dequeue from the queue
    if current is the target:
        return the solution (or number of steps, etc.)
    for each neighbor of current:
        if neighbor is valid and not visited:
            mark neighbor as visited
            enqueue neighbor

if the target was not found:
    return failure (or -1, etc.)
```

#### python

Arguments: start, end, and max_steps

```py
from collections import deque

def can_reach_end(maze, start, end, max_steps):
    """
    Determines if you can reach 'end' from 'start' within max_steps in the maze.

    Parameters:
    - maze: 2D list where 0 represents open cell and 1 represents wall.
    - start: Tuple (row, col) for the starting cell.
    - end: Tuple (row, col) for the ending cell.
    - max_steps: Maximum number of steps allowed.

    Returns:
    - True if the end is reachable within max_steps, False otherwise.
    """
    # Maze dimensions.
    rows, cols = len(maze), len(maze[0])
    
    # Directions: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS: stores tuples of (row, col, steps_taken).
    queue = deque([(start[0], start[1], 0)])
    
    # A set to keep track of visited positions.
    visited = set([start])
    
    while queue:
        r, c, steps = queue.popleft()
        
        # Check if we reached the end.
        if (r, c) == end:
            # Even if we reached the end, ensure we did it within max_steps.
            if steps <= max_steps:
                return True
            # though in BFS this should rarely occur.
            continue
        
        # If we've already taken max_steps, we cannot move further.
        if steps == max_steps:
            continue
        
        # Explore all four directions.
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Check boundaries and if the cell is open and not visited.
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))
    
    # If we exit the loop without returning, the end is not reachable within max_steps.
    return False

# Example usage:
if __name__ == "__main__":
    # Define a sample maze:
    # 0: open cell, 1: wall
    maze = [
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ]
    
    start = (0, 0)    # Starting at top-left corner.
    end = (3, 3)      # Ending at bottom-right corner.
    max_steps = 6     # Maximum steps allowed.
    
    if can_reach_end(maze, start, end, max_steps):
        print("The end is reachable within", max_steps, "steps.")
    else:
        print("The end is NOT reachable within", max_steps, "steps.")

```

Without arguments: implied start and end

```py
from collections import deque

def min_steps_through_maze(maze):
    """
    Returns the minimum number of steps required to go from the top-left
    to the bottom-right of the maze. Returns -1 if no such path exists.

    Parameters:
    - maze: 2D list where 0 represents an open cell and 1 represents a wall.
    """
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)
    
    # If the start or end is blocked, return -1 immediately.
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return -1

    # Queue for BFS: (row, col, steps_taken)
    queue = deque([(start[0], start[1], 0)])
    visited = {(start[0], start[1])}

    # Possible movements: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c, steps = queue.popleft()

        # Check if we reached the exit.
        if (r, c) == end:
            return steps

        # Explore neighbors.
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                maze[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))

    # If exit is not reached, return -1.
    return -1

# Example maze:
maze = [
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
]

steps = min_steps_through_maze(maze)
if steps != -1:
    print("Minimum steps to exit the maze:", steps)
else:
    print("The exit is not reachable from the start.")
```

#### Ruby

```ruby
def min_steps_through_maze(maze)
  rows = maze.size
  cols = maze[0].size
  start = [0, 0]
  finish = [rows - 1, cols - 1]

  # Return -1 immediately if the start or finish is blocked.
  return -1 if maze[start[0]][start[1]] == 1 || maze[finish[0]][finish[1]] == 1

  # Queue for BFS: each element is [row, col, steps_taken]
  queue = [[0, 0, 0]]
  visited = { [0, 0] => true }

  # Possible movements: up, down, left, right.
  directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

  until queue.empty?
    r, c, steps = queue.shift

    return steps if [r, c] == finish

    directions.each do |dr, dc|
      nr, nc = r + dr, c + dc
      if nr.between?(0, rows - 1) && nc.between?(0, cols - 1) &&
         maze[nr][nc] == 0 && !visited.key?([nr, nc])
        visited[[nr, nc]] = true
        queue.push([nr, nc, steps + 1])
      end
    end
  end

  # If no path is found, return -1.
  -1
end

# Example maze:
maze = [
  [0, 0, 0, 1],
  [1, 0, 0, 1],
  [0, 0, 0, 0],
  [0, 1, 1, 0]
]

steps = min_steps_through_maze(maze)
if steps != -1
  puts "Minimum steps to exit the maze: #{steps}"
else
  puts "The exit is not reachable from the start."
end
```

### Conclusion

To memorize BFS, think of it like spreading ripples in water. You start from one point, explore all possible moves at the same distance first, then move outward step by step. Always use a **queue** (FIFO) to track what to explore next and a **visited set** to avoid repeating work.  

#### Steps to Remember:  
1. **Start with the queue** → Put the starting point in.  
2. **Loop until the queue is empty** → Take out the first item.  
3. **Check if it’s the goal** → If yes, return the number of steps.  
4. **Expand outward** → Add all valid, unvisited neighbors to the queue.  
5. **Repeat until the goal is found or the queue is empty.**  

Practice this with mazes, shortest paths in graphs, tree levels, and word ladders. Draw things out, write code often, and explain it to yourself. The more you use BFS, the more natural it will feel! 

## DFS (Depth-First Search)

**Depth-First Nature:**
DFS explores as far as possible along a branch before backtracking. This is useful for problems requiring exhaustive exploration, such as solving puzzles, detecting cycles, or finding connected components in a graph.

### Code examples

#### pseudo

```shell
initialize a stack with the starting node/state
mark the starting node/state as visited

while the stack is not empty:
    current = pop from the stack
    if current is the target:
        return the solution (or number of steps, etc.)
    for each neighbor of current in reverse order:
        if neighbor is valid and not visited:
            mark neighbor as visited
            push neighbor onto the stack

if the target was not found:
    return failure (or -1, etc.)
```

#### python

```py
from collections import deque

def dfs_traverse(graph, start):
    """
    Performs DFS traversal on a graph represented as an adjacency list.
    """
    visited = set()
    stack = [start]
    traversal = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            traversal.append(node)
            stack.extend(reversed(graph.get(node, [])))
    
    return traversal

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(dfs_traverse(graph, 'A'))  # Expected output: ['A', 'B', 'D', 'E', 'F', 'C']
```

#### ruby

```ruby
def dfs_traverse(graph, start)
  visited = {}
  stack = [start]
  traversal = []

  until stack.empty?
    node = stack.pop
    next if visited[node]
    
    visited[node] = true
    traversal << node
    stack.concat(graph[node].reverse) if graph[node]
  end

  traversal
end

# Example usage:
graph = {
  'A' => ['B', 'C'],
  'B' => ['D', 'E'],
  'C' => ['F'],
  'D' => [],
  'E' => ['F'],
  'F' => []
}

puts dfs_traverse(graph, 'A').inspect # Expected output: ["A", "B", "D", "E", "F", "C"]
```

### Conclusion

DFS is great for problems requiring deep exploration before considering alternative paths. It's useful for:
- **Pathfinding** (when the path doesn’t need to be shortest).
- **Cycle detection** in graphs.
- **Connected component detection** in an undirected graph.
- **Backtracking problems** (like Sudoku solving, maze exploration, and N-Queens).

#### Steps to Remember:
1. **Use a stack** → Push the starting node onto it.
2. **Pop from stack, process, and mark visited**.
3. **Push neighbors in reverse order** → Ensures correct traversal order.
4. **Repeat until the stack is empty**.
5. **Use recursion for cleaner implementation when applicable**.

## Hash Map (Dictionary)

> Write a function that takes a list of tuples, where each tuple contains (department, employee, salary), and returns a dictionary mapping each department to the average salary for that department. Assume the list is unsorted.

This problem is best solved using a **hash map (dictionary in Python, hash in Ruby)** to group data by department, sum salaries, and count employees. Then, we compute the average for each department. This is a classic **"grouping and aggregation"** problem, similar to SQL’s `GROUP BY`.

### Pseudocode Solution

```shell
initialize an empty dictionary dept_totals

for each record in records:
    extract department, employee, salary
    if department not in dept_totals:
        dept_totals[department] = [0, 0]  # [total_salary, count]
    add salary to dept_totals[department][0]
    increment dept_totals[department][1] by 1

initialize empty dictionary avg_salary
for each department in dept_totals:
    extract total_salary and count
    avg_salary[department] = total_salary / count

return avg_salary
```

### Python Solution

```python
from collections import defaultdict

def average_salary(records):
    dept_totals = defaultdict(lambda: [0, 0])  # {department: [total_salary, count]}

    for department, _, salary in records:
        dept_totals[department][0] += salary
        dept_totals[department][1] += 1

    return {dept: total / count for dept, (total, count) in dept_totals.items()}

# Example Usage
records = [("HR", "Alice", 50000), ("HR", "Bob", 55000), ("Eng", "Charlie", 80000), ("Eng", "David", 90000)]
print(average_salary(records))  # {'HR': 52500.0, 'Eng': 85000.0}
```

### Ruby Solution

```ruby
def average_salary(records)
  dept_totals = Hash.new { |hash, key| hash[key] = [0, 0] }  # {department => [total_salary, count]}

  records.each do |department, _, salary|
    dept_totals[department][0] += salary
    dept_totals[department][1] += 1
  end

  dept_totals.transform_values { |total, count| total.to_f / count }
end

# Example Usage
records = [["HR", "Alice", 50000], ["HR", "Bob", 55000], ["Eng", "Charlie", 80000], ["Eng", "David", 90000]]
puts average_salary(records)  # {"HR"=>52500.0, "Eng"=>85000.0}
```

### Easy-to-Memorize Summary
🧠 **"Group & Average"** → Use a **hash map (dictionary)** to:
1. **Group** records by department.
2. **Accumulate** total salary and count.
3. **Compute average** by dividing total by count.

Any question that involves **grouping data and performing calculations on it (like averages, sums, or counts)** can be solved using this approach. 

## Data Design Scenarios

### Common Data Types

#### Heaps (Priority Queues)
Heaps are specialized tree-based data structures that satisfy the heap property, where the parent node is either greater than or equal to (max-heap) or less than or equal to (min-heap) its children. Priority queues are abstract data types that allow for efficient retrieval of the highest (or lowest) priority element.

Used in financial applications for managing priority-based data.

```python
import heapq
# Example: Maintaining a min-heap for order matching
order_heap = []
heapq.heappush(order_heap, (100, "buy", 10))
print(heapq.heappop(order_heap))  # Retrieves the lowest price order: (100, 'buy', 10)
```
```ruby
require 'pqueue'
# Example: Maintaining a min-heap for order matching
order_heap = PQueue.new([ [100, "buy", 10], [101, "buy", 5] ])
puts order_heap.pop  # Retrieves the lowest price order: [100, "buy", 10]
```
Segment trees and Fenwick trees are data structures used for efficient range queries and updates. Segment trees allow querying the sum, minimum, or maximum of elements in a subarray in logarithmic time, while Fenwick trees (or Binary Indexed Trees) provide a way to perform prefix sum queries and updates efficiently.

#### Trees (Segment Trees, Fenwick Trees)
Used for efficient range queries and financial analysis.

```python
# Example: Segment tree for market depth tracking
class SegmentTree:
    def __init__(self, data):
        self.data = data  # Placeholder for implementation
segment_tree = SegmentTree([10, 20, 30, 40])
print(segment_tree.data)  # Outputs: [10, 20, 30, 40]
```
```ruby
# Example: Simple tree structure
class SegmentTree
  def initialize(data)
    @data = data # Placeholder for implementation
  end
end
segment_tree = SegmentTree.new([10, 20, 30, 40])
puts segment_tree.inspect  # Outputs: #<SegmentTree:0x000056 @data=[10, 20, 30, 40]>
```

#### Graphs

Graphs are data structures that consist of nodes (vertices) connected by edges. They are used to model relationships and connections between entities. Applications of graphs include social networks, transportation networks, and financial networks, where they can represent market structures and transactions.

Graphs model financial networks and market structures.

```python
import networkx as nx
G = nx.Graph()
G.add_edge("Bank A", "Bank B", weight=5)  # 5 billion transaction
print(list(G.edges(data=True)))  # Outputs: [('Bank A', 'Bank B', {'weight': 5})]
```
```ruby
require 'rgl/adjacency'
g = RGL::AdjacencyGraph.new
g.add_edge("Bank A", "Bank B")
puts g.edges.to_a  # Outputs: [["Bank A", "Bank B"]]
```

#### Time Series Storage
Efficiently storing and querying stock price movements.

Time series storage involves storing data points collected or recorded at specific time intervals. This type of storage is crucial for applications that require tracking changes over time, such as stock price movements, sensor data, and economic indicators. Efficient time series storage allows for quick querying and analysis of historical data to identify trends, patterns, and anomalies.

```python
# Example: Using Pandas for time series storage
import pandas as pd
price_data = pd.DataFrame({"timestamp": ["2024-02-01", "2024-02-02"], "price": [150.25, 151.00]})
print(price_data)  # Outputs the stored time series data
```
```ruby
# Example: Using an array to store time series data
time_series = [{timestamp: "2024-02-01", price: 150.25}, {timestamp: "2024-02-02", price: 151.00}]
puts time_series.inspect  # Outputs stored time series data
```
#### Real-Time Alerting System

A real-time alerting system continuously monitors stock price changes, economic indicators, and trade executions to trigger alerts when predefined conditions are met. These systems are crucial in financial markets for timely decision-making and risk management.

```python
# Example: Threshold-based alert system
threshold = 150.00
current_price = 151.00
if current_price > threshold:
    print("Alert: Price exceeded threshold!")  # Outputs: Alert: Price exceeded threshold!
```
```ruby
# Example: Threshold-based alert system
threshold = 150.00
current_price = 151.00
puts "Alert: Price exceeded threshold!" if current_price > threshold
```

**Interview Question:** How would you design a system to trigger alerts when stock prices cross predefined thresholds in real time?

#### Order Matching Engine

An order matching engine is a core component of a trading platform that matches buy and sell orders for financial instruments. It ensures that orders are executed at the best possible price by maintaining a limit order book, which records all outstanding buy and sell orders. The engine matches orders based on price and time priority, facilitating efficient and fair trading.

Handles financial transactions by managing a **limit order book**.

```python
# Example: Simple order book using a sorted list
class OrderBook:
    def __init__(self):
        self.orders = []
    def add_order(self, price, quantity):
        self.orders.append((price, quantity))
        self.orders.sort()
book = OrderBook()
book.add_order(100, 10)
book.add_order(101, 5)
print(book.orders)  # Outputs: [(100, 10), (101, 5)]
```
```ruby
# Example: Simple order book implementation
class OrderBook
  def initialize
    @orders = []
  end
  def add_order(price, quantity)
    @orders.push([price, quantity])
    @orders.sort!
  end
end
book = OrderBook.new
book.add_order(100, 10)
book.add_order(101, 5)
puts book.inspect  # Outputs: Sorted order book
```

**Interview Question:** Implement a matching engine for a stock exchange that efficiently processes buy and sell orders.

#### Handling High-Frequency Data Streams

Efficiently managing high-volume financial data updates.

High-frequency data streams involve the continuous and rapid flow of data, often in real-time, such as stock price updates, sensor data, or social media feeds. These streams are crucial in applications like algorithmic trading, real-time analytics, and monitoring systems, where timely processing and analysis of data are essential.

```python
# Example: Streaming stock prices using Kafka (conceptual)
from kafka import KafkaConsumer
consumer = KafkaConsumer('stock_prices')
for message in consumer:
    print(message.value)  # Outputs real-time stock price updates
```
```ruby
# Example: Streaming data conceptually (not native Kafka support in Ruby)
stock_prices = Queue.new
Thread.new do
  loop do
    price = rand(100..200)  # Simulate stock price updates
    stock_prices.push(price)
    sleep(1)
  end
end
puts stock_prices.pop  # Outputs real-time stock price update
```
#### Latency Optimization

Reducing execution delays in trading environments.

Latency optimization involves reducing the time delay between the initiation and execution of a process. In trading environments, this is crucial for ensuring that transactions are executed as quickly as possible, minimizing the risk of price changes and maximizing the efficiency of trading strategies.

```python
# Example: Using memory-mapped files for fast access
import mmap
with open("trade_data.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    print(mm.readline())  # Outputs: First line of file data
```
```ruby
# Example: Using IO for fast access
File.open("trade_data.bin", "rb") do |f|
  puts f.readline  # Outputs: First line of file data
end
```
#### Data Aggregation

Computing key financial indicators efficiently.

Data aggregation involves collecting and summarizing data to extract meaningful insights. It is commonly used in financial analysis to compute key indicators such as moving averages, sums, and counts, which help in understanding market trends and making informed decisions.

```python
# Example: Using NumPy for fast moving average calculation
import numpy as np
prices = np.array([150, 152, 153, 155])
moving_avg = np.convolve(prices, np.ones(3)/3, mode='valid')
print(moving_avg)  # Outputs: Moving average values
```
```ruby
# Example: Simple moving average calculation
prices = [150, 152, 153, 155]
moving_avg = prices.each_cons(3).map { |subarr| subarr.sum / 3.0 }
puts moving_avg.inspect  # Outputs: Moving average values
```

**Interview Question:** Implement a system that efficiently calculates moving averages on live stock price data.

