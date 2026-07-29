import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

@dataclass
class Task:
    """Represents a computational task in the DAG"""
    task_id: int
    computation_required: float  # CPU cycles
    data_size: float  # KB
    dependencies: List[int]  # List of task IDs that must complete before this task
    level: int = 0  # Topological level in DAG

@dataclass
class ComputingNode:
    """Represents a computing node (Vehicle or Edge Server)"""
    node_id: int
    node_type: str  # 'vehicle' or 'edge_server'
    computing_power: float  # CPU frequency in GHz
    position: Tuple[float, float]  # (x, y) coordinates
    bandwidth: float = 100.0  # Mbps

class ACOTaskOffloader:
    """Ant Colony Optimization for IoV Task Offloading"""
    
    def __init__(self, num_ants: int = 20, evaporation_rate: float = 0.5, 
                 alpha: float = 1.0, beta: float = 2.0, q0: float = 0.9):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.q0 = q0        # Exploitation probability
        
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.best_solution = None
        self.best_makespan = float('inf')
        
    def initialize_pheromones(self, num_tasks: int, num_nodes: int, initial_value: float = 1.0):
        """Initialize pheromone matrix"""
        self.pheromone_matrix = np.full((num_tasks, num_nodes), initial_value)
        
    def calculate_heuristics(self, tasks: List[Task], nodes: List[ComputingNode], dag: nx.DiGraph):
        """Calculate heuristic matrix based on task and node characteristics"""
        num_tasks = len(tasks)
        num_nodes = len(nodes)
        self.heuristic_matrix = np.zeros((num_tasks, num_nodes))
        
        for i, task in enumerate(tasks):
            for j, node in enumerate(nodes):
                # Heuristic: inverse of estimated completion time
                computation_time = task.computation_required / (node.computing_power * 1e9)
                communication_time = task.data_size * 8 / (node.bandwidth * 1024)  # Convert to seconds
                total_time = computation_time + communication_time
                
                # Consider node load and task dependencies
                dependency_penalty = len(task.dependencies) * 0.1
                
                self.heuristic_matrix[i, j] = 1.0 / (total_time + dependency_penalty + 1e-8)
    
    def construct_solution(self, tasks: List[Task], nodes: List[ComputingNode], dag: nx.DiGraph):
        """Construct a solution using ant colony rules"""
        num_tasks = len(tasks)
        num_nodes = len(nodes)
        solution = np.full(num_tasks, -1)  # -1 means not assigned
        scheduled = set()
        
        # Get topological order of tasks
        topological_order = list(nx.topological_sort(dag))
        
        for task_id in topological_order:
            task_idx = task_id - 1  # Assuming task IDs start from 1
            available_nodes = list(range(num_nodes))
            
            if random.random() < self.q0:
                # Exploitation: choose the best node based on pheromone and heuristic
                probabilities = self.pheromone_matrix[task_idx] ** self.alpha * \
                              self.heuristic_matrix[task_idx] ** self.beta
                chosen_node = np.argmax(probabilities)
            else:
                # Exploration: choose node probabilistically
                probabilities = self.pheromone_matrix[task_idx] ** self.alpha * \
                              self.heuristic_matrix[task_idx] ** self.beta
                probabilities = probabilities / np.sum(probabilities)
                chosen_node = np.random.choice(num_nodes, p=probabilities)
            
            solution[task_idx] = chosen_node
            scheduled.add(task_id)
        
        return solution
    
    def calculate_makespan(self, solution: np.ndarray, tasks: List[Task], 
                          nodes: List[ComputingNode], dag: nx.DiGraph) -> float:
        """Calculate makespan for a given solution"""
        num_tasks = len(tasks)
        node_ready_time = [0.0] * len(nodes)
        task_completion_time = [0.0] * num_tasks
        
        # Get topological order
        topological_order = list(nx.topological_sort(dag))
        
        for task_id in topological_order:
            task_idx = task_id - 1
            task = tasks[task_idx]
            assigned_node = solution[task_idx]
            node = nodes[assigned_node]
            
            # Calculate start time (max of node ready time and all dependencies completion)
            start_time = node_ready_time[assigned_node]
            for dep_id in task.dependencies:
                dep_idx = dep_id - 1
                if task_completion_time[dep_idx] > start_time:
                    start_time = task_completion_time[dep_idx]
            
            # Calculate task execution time
            computation_time = task.computation_required / (node.computing_power * 1e9)
            
            # Add communication time if task has dependencies on different nodes
            comm_time = 0.0
            for dep_id in task.dependencies:
                dep_idx = dep_id - 1
                if solution[dep_idx] != assigned_node:
                    comm_time += task.data_size * 8 / (node.bandwidth * 1024)  # seconds
            
            completion_time = start_time + computation_time + comm_time
            task_completion_time[task_idx] = completion_time
            node_ready_time[assigned_node] = completion_time
        
        return max(task_completion_time)
    
    def update_pheromones(self, solutions: List[np.ndarray], makespans: List[float]):
        """Update pheromone matrix based on ant solutions"""
        # Evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Find best solution in this iteration
        best_idx = np.argmin(makespans)
        best_solution = solutions[best_idx]
        best_makespan = makespans[best_idx]
        
        # Update global best
        if best_makespan < self.best_makespan:
            self.best_makespan = best_makespan
            self.best_solution = best_solution.copy()
        
        # Pheromone deposition (only by best ant)
        pheromone_deposit = 1.0 / (best_makespan + 1e-8)
        
        for task_idx, node_idx in enumerate(best_solution):
            self.pheromone_matrix[task_idx, node_idx] += pheromone_deposit
    
    def optimize(self, tasks: List[Task], nodes: List[ComputingNode], 
                 dag: nx.DiGraph, max_iterations: int = 100) -> Dict:
        """Main optimization loop"""
        num_tasks = len(tasks)
        num_nodes = len(nodes)
        
        # Initialize matrices
        self.initialize_pheromones(num_tasks, num_nodes)
        self.calculate_heuristics(tasks, nodes, dag)
        
        convergence_data = []
        
        for iteration in range(max_iterations):
            solutions = []
            makespans = []
            
            # Each ant constructs a solution
            for _ in range(self.num_ants):
                solution = self.construct_solution(tasks, nodes, dag)
                makespan = self.calculate_makespan(solution, tasks, nodes, dag)
                
                solutions.append(solution)
                makespans.append(makespan)
            
            # Update pheromones
            self.update_pheromones(solutions, makespans)
            
            # Store convergence data
            avg_makespan = np.mean(makespans)
            best_makespan = np.min(makespans)
            convergence_data.append({
                'iteration': iteration,
                'avg_makespan': avg_makespan,
                'best_makespan': best_makespan,
                'global_best': self.best_makespan
            })
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best = {best_makespan:.3f}s, "
                      f"Avg = {avg_makespan:.3f}s, Global Best = {self.best_makespan:.3f}s")
        
        return {
            'best_solution': self.best_solution,
            'best_makespan': self.best_makespan,
            'convergence_data': convergence_data,
            'pheromone_matrix': self.pheromone_matrix
        }

class DAGGenerator:
    """Generate synthetic DAG tasks for testing"""
    
    @staticmethod
    def generate_random_dag(num_tasks: int, max_dependencies: int = 3) -> Tuple[List[Task], nx.DiGraph]:
        """Generate a random DAG with tasks"""
        dag = nx.DiGraph()
        tasks = []
        
        # Add nodes
        for i in range(1, num_tasks + 1):
            dag.add_node(i)
        
        # Add edges to create DAG structure
        for i in range(2, num_tasks + 1):
            num_deps = random.randint(1, min(max_dependencies, i-1))
            dependencies = random.sample(range(1, i), num_deps)
            
            for dep in dependencies:
                dag.add_edge(dep, i)
        
        # Create Task objects
        for i in range(1, num_tasks + 1):
            computation = random.randint(10**7, 10**8)  # CPU cycles
            data_size = random.randint(50, 500)  # KB
            dependencies = list(dag.predecessors(i))
            
            tasks.append(Task(
                task_id=i,
                computation_required=computation,
                data_size=data_size,
                dependencies=dependencies
            ))
        
        return tasks, dag

class VehicleEnvironment:
    """Simulate vehicle and edge server environment"""
    
    def __init__(self, num_vehicles: int = 5, num_edge_servers: int = 3):
        self.nodes = []
        self.initialize_nodes(num_vehicles, num_edge_servers)
    
    def initialize_nodes(self, num_vehicles: int, num_edge_servers: int):
        """Initialize computing nodes"""
        node_id = 0
        
        # Add edge servers
        for i in range(num_edge_servers):
            self.nodes.append(ComputingNode(
                node_id=node_id,
                node_type='edge_server',
                computing_power=random.uniform(8.0, 12.0),  # 8-12 GHz
                position=(random.uniform(0, 1000), random.uniform(0, 1000)),
                bandwidth=random.uniform(500, 1000)  # 500-1000 Mbps
            ))
            node_id += 1
        
        # Add vehicles
        for i in range(num_vehicles):
            self.nodes.append(ComputingNode(
                node_id=node_id,
                node_type='vehicle',
                computing_power=random.uniform(1.0, 3.0),  # 1-3 GHz
                position=(random.uniform(0, 1000), random.uniform(0, 1000)),
                bandwidth=random.uniform(50, 100)  # 50-100 Mbps
            ))
            node_id += 1
    
    def get_nodes(self) -> List[ComputingNode]:
        return self.nodes

def plot_aco_results(optimization_result: Dict, tasks: List[Task], nodes: List[ComputingNode]):
    """Plot ACO optimization results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ACO Task Offloading Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. Convergence plot
    convergence_data = optimization_result['convergence_data']
    iterations = [data['iteration'] for data in convergence_data]
    avg_makespans = [data['avg_makespan'] for data in convergence_data]
    best_makespans = [data['best_makespan'] for data in convergence_data]
    global_best = [data['global_best'] for data in convergence_data]
    
    axes[0, 0].plot(iterations, avg_makespans, 'b-', label='Average Makespan', alpha=0.7)
    axes[0, 0].plot(iterations, best_makespans, 'g-', label='Iteration Best', alpha=0.7)
    axes[0, 0].plot(iterations, global_best, 'r-', label='Global Best', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Makespan (seconds)')
    axes[0, 0].set_title('ACO Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pheromone matrix heatmap
    pheromone_matrix = optimization_result['pheromone_matrix']
    im = axes[0, 1].imshow(pheromone_matrix, cmap='hot', aspect='auto')
    axes[0, 1].set_xlabel('Computing Node')
    axes[0, 1].set_ylabel('Task')
    axes[0, 1].set_title('Final Pheromone Matrix')
    plt.colorbar(im, ax=axes[0, 1], label='Pheromone Intensity')
    
    # 3. Task assignment distribution
    best_solution = optimization_result['best_solution']
    node_assignments = [0] * len(nodes)
    for assignment in best_solution:
        node_assignments[assignment] += 1
    
    node_types = ['Edge' if node.node_type == 'edge_server' else 'Vehicle' 
                  for node in nodes]
    colors = ['red' if node_type == 'Edge' else 'blue' for node_type in node_types]
    
    axes[0, 2].bar(range(len(nodes)), node_assignments, color=colors, alpha=0.7)
    axes[0, 2].set_xlabel('Node ID')
    axes[0, 2].set_ylabel('Number of Tasks Assigned')
    axes[0, 2].set_title('Task Assignment Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Node utilization
    node_utilization = []
    for i, node in enumerate(nodes):
        total_computation = sum(tasks[j].computation_required 
                               for j, assign in enumerate(best_solution) if assign == i)
        utilization = total_computation / (node.computing_power * 1e9 * 100)  # Normalized
        node_utilization.append(utilization)
    
    axes[1, 0].bar(range(len(nodes)), node_utilization, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Node ID')
    axes[1, 0].set_ylabel('Utilization Rate')
    axes[1, 0].set_title('Node Utilization')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Task completion timeline
    from aco_implementation import ACOTaskOffloader  # Import for makespan calculation
    temp_offloader = ACOTaskOffloader()
    completion_times = temp_offloader.calculate_task_completion_times(
        best_solution, tasks, nodes, nx.DiGraph()
    )
    
    axes[1, 1].bar(range(len(tasks)), completion_times, alpha=0.7)
    axes[1, 1].set_xlabel('Task ID')
    axes[1, 1].set_ylabel('Completion Time (seconds)')
    axes[1, 1].set_title('Task Completion Timeline')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Solution quality comparison
    # Compare with random and greedy baselines
    random_makespan = evaluate_random_solution(tasks, nodes, nx.DiGraph())
    greedy_makespan = evaluate_greedy_solution(tasks, nodes, nx.DiGraph())
    aco_makespan = optimization_result['best_makespan']
    
    methods = ['Random', 'Greedy', 'ACO']
    makespans = [random_makespan, greedy_makespan, aco_makespan]
    
    bars = axes[1, 2].bar(methods, makespans, color=['red', 'orange', 'green'], alpha=0.7)
    axes[1, 2].set_ylabel('Makespan (seconds)')
    axes[1, 2].set_title('Comparison with Baselines')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, makespan in zip(bars, makespans):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{makespan:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('aco_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_random_solution(tasks: List[Task], nodes: List[ComputingNode], dag: nx.DiGraph) -> float:
    """Evaluate a random solution for comparison"""
    num_tasks = len(tasks)
    random_solution = np.random.randint(0, len(nodes), num_tasks)
    
    offloader = ACOTaskOffloader()
    return offloader.calculate_makespan(random_solution, tasks, nodes, dag)

def evaluate_greedy_solution(tasks: List[Task], nodes: List[ComputingNode], dag: nx.DiGraph) -> float:
    """Evaluate a greedy solution for comparison"""
    num_tasks = len(tasks)
    greedy_solution = np.zeros(num_tasks, dtype=int)
    
    topological_order = list(nx.topological_sort(dag))
    
    for task_id in topological_order:
        task_idx = task_id - 1
        task = tasks[task_idx]
        
        best_node = 0
        best_time = float('inf')
        
        for node_idx, node in enumerate(nodes):
            computation_time = task.computation_required / (node.computing_power * 1e9)
            if computation_time < best_time:
                best_time = computation_time
                best_node = node_idx
        
        greedy_solution[task_idx] = best_node
    
    offloader = ACOTaskOffloader()
    return offloader.calculate_makespan(greedy_solution, tasks, nodes, dag)

# Enhanced ACO implementation with additional features
class EnhancedACOTaskOffloader(ACOTaskOffloader):
    """Enhanced ACO with additional features"""
    
    def __init__(self, num_ants: int = 20, evaporation_rate: float = 0.5, 
                 alpha: float = 1.0, beta: float = 2.0, q0: float = 0.9,
                 local_search: bool = True):
        super().__init__(num_ants, evaporation_rate, alpha, beta, q0)
        self.local_search = local_search
    
    def local_search_improvement(self, solution: np.ndarray, tasks: List[Task], 
                                nodes: List[ComputingNode], dag: nx.DiGraph) -> np.ndarray:
        """Apply local search to improve solution"""
        improved_solution = solution.copy()
        current_makespan = self.calculate_makespan(solution, tasks, nodes, dag)
        
        for task_idx in range(len(tasks)):
            original_node = solution[task_idx]
            
            for new_node in range(len(nodes)):
                if new_node != original_node:
                    improved_solution[task_idx] = new_node
                    new_makespan = self.calculate_makespan(improved_solution, tasks, nodes, dag)
                    
                    if new_makespan < current_makespan:
                        current_makespan = new_makespan
                    else:
                        improved_solution[task_idx] = original_node
        
        return improved_solution
    
    def calculate_task_completion_times(self, solution: np.ndarray, tasks: List[Task], 
                                       nodes: List[ComputingNode], dag: nx.DiGraph) -> List[float]:
        """Calculate completion time for each task"""
        num_tasks = len(tasks)
        node_ready_time = [0.0] * len(nodes)
        task_completion_time = [0.0] * num_tasks
        
        topological_order = list(nx.topological_sort(dag))
        
        for task_id in topological_order:
            task_idx = task_id - 1
            task = tasks[task_idx]
            assigned_node = solution[task_idx]
            node = nodes[assigned_node]
            
            start_time = node_ready_time[assigned_node]
            for dep_id in task.dependencies:
                dep_idx = dep_id - 1
                if task_completion_time[dep_idx] > start_time:
                    start_time = task_completion_time[dep_idx]
            
            computation_time = task.computation_required / (node.computing_power * 1e9)
            comm_time = 0.0
            for dep_id in task.dependencies:
                dep_idx = dep_id - 1
                if solution[dep_idx] != assigned_node:
                    comm_time += task.data_size * 8 / (node.bandwidth * 1024)
            
            completion_time = start_time + computation_time + comm_time
            task_completion_time[task_idx] = completion_time
            node_ready_time[assigned_node] = completion_time
        
        return task_completion_time

def run_aco_experiment():
    """Run complete ACO experiment"""
    print("Starting ACO-based Task Offloading Experiment...")
    
    # Generate environment and tasks
    env = VehicleEnvironment(num_vehicles=8, num_edge_servers=3)
    nodes = env.get_nodes()
    
    tasks, dag = DAGGenerator.generate_random_dag(num_tasks=20, max_dependencies=3)
    
    print(f"Generated {len(tasks)} tasks with DAG structure")
    print(f"Available nodes: {len([n for n in nodes if n.node_type == 'edge_server'])} edge servers, "
          f"{len([n for n in nodes if n.node_type == 'vehicle'])} vehicles")
    
    # Initialize ACO optimizer
    aco_optimizer = EnhancedACOTaskOffloader(
        num_ants=30,
        evaporation_rate=0.6,
        alpha=1.0,
        beta=3.0,
        q0=0.8,
        local_search=True
    )
    
    # Run optimization
    result = aco_optimizer.optimize(
        tasks=tasks,
        nodes=nodes,
        dag=dag,
        max_iterations=100
    )
    
    print(f"\nOptimization completed!")
    print(f"Best makespan: {result['best_makespan']:.3f} seconds")
    
    # Evaluate baselines for comparison
    random_makespan = evaluate_random_solution(tasks, nodes, dag)
    greedy_makespan = evaluate_greedy_solution(tasks, nodes, dag)
    
    print(f"Random solution makespan: {random_makespan:.3f} seconds")
    print(f"Greedy solution makespan: {greedy_makespan:.3f} seconds")
    print(f"ACO improvement over random: {((random_makespan - result['best_makespan']) / random_makespan * 100):.1f}%")
    print(f"ACO improvement over greedy: {((greedy_makespan - result['best_makespan']) / greedy_makespan * 100):.1f}%")
    
    # Plot results
    plot_aco_results(result, tasks, nodes)
    
    return result, tasks, nodes, dag

# Real-time optimization with dynamic environment
class DynamicACOOptimizer:
    """ACO optimizer for dynamic IoV environments"""
    
    def __init__(self, base_optimizer: EnhancedACOTaskOffloader):
        self.base_optimizer = base_optimizer
        self.solution_history = []
    
    def adapt_to_changes(self, new_tasks: List[Task], removed_tasks: List[int],
                        new_nodes: List[ComputingNode], removed_nodes: List[int]):
        """Adapt existing solution to environment changes"""
        # This would implement incremental optimization
        # For now, return the base optimizer
        return self.base_optimizer

if __name__ == "__main__":
    # Run the complete ACO experiment
    result, tasks, nodes, dag = run_aco_experiment()
    
    # Print detailed assignment
    print("\nDetailed Task Assignment:")
    print("Task ID -> Node ID (Type)")
    for i, task in enumerate(tasks):
        node_idx = result['best_solution'][i]
        node = nodes[node_idx]
        print(f"Task {task.task_id:2d} -> Node {node_idx:2d} ({node.node_type:>6})")
