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