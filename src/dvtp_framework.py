import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch_geometric.nn import GATConv
import gym
from collections import deque
import random

class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, 2)  # x, y coordinates

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.output_projection(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VariationalGATEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATConv(in_features, hidden_dim, heads=num_heads, concat=True))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True))
        
        # Last layer
        self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False))
        
        self.activation = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = self.activation(layer(x, edge_index))
        
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

class DVTPPolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_nodes, num_locations, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_locations = num_locations
        
        # Shared feature extractor
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Node selection head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_nodes)
        )
        
        # Location selection head
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_locations)
        )
        
        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, node_mask=None, location_mask=None):
        features = self.shared_network(state)
        
        # Node logits
        node_logits = self.node_head(features)
        if node_mask is not None:
            node_logits = node_logits + node_mask
        
        # Location logits
        location_logits = self.location_head(features)
        if location_mask is not None:
            location_logits = location_logits + location_mask
        
        # Value
        value = self.value_network(features)
        
        return node_logits, location_logits, value

class DAGEnvironment:
    def __init__(self, num_vehicles=3, num_servers=2, max_nodes=20):
        self.num_vehicles = num_vehicles
        self.num_servers = num_servers
        self.max_nodes = max_nodes
        self.reset()
    
    def generate_dag(self):
        """Generate a random DAG for simulation"""
        G = nx.DiGraph()
        num_nodes = random.randint(5, self.max_nodes)
        
        # Add nodes with computational requirements
        for i in range(num_nodes):
            G.add_node(i, 
                      computation=random.randint(10**7, 10**8),
                      data_size=random.randint(50, 500),
                      in_degree=0,
                      out_degree=0)
        
        # Add edges with dependencies
        for i in range(1, num_nodes):
            parent = random.randint(0, i-1)
            G.add_edge(parent, i)
        
        # Update degrees
        for node in G.nodes():
            G.nodes[node]['in_degree'] = G.in_degree(node)
            G.nodes[node]['out_degree'] = G.out_degree(node)
        
        return G
    
    def reset(self):
        """Reset environment"""
        self.dags = [self.generate_dag() for _ in range(self.num_vehicles)]
        self.scheduled_nodes = set()
        self.available_nodes = self._get_available_nodes()
        self.current_step = 0
        self.completion_times = {i: 0 for i in range(self.num_vehicles)}
        
        return self._get_state()
    
    def _get_available_nodes(self):
        """Get nodes that are ready to be scheduled (all predecessors scheduled)"""
        available = set()
        for veh_id, dag in enumerate(self.dags):
            for node in dag.nodes():
                if (veh_id, node) in self.scheduled_nodes:
                    continue
                
                # Check if all predecessors are scheduled
                predecessors = list(dag.predecessors(node))
                if all((veh_id, pred) in self.scheduled_nodes for pred in predecessors):
                    available.add((veh_id, node))
        
        return available
    
    def _get_state(self):
        """Get current state representation"""
        # Simplified state representation for demo
        state = []
        
        # Node features
        for veh_id, dag in enumerate(self.dags):
            for node in dag.nodes():
                node_features = [
                    dag.nodes[node]['computation'] / 10**8,
                    dag.nodes[node]['data_size'] / 500,
                    dag.nodes[node]['in_degree'] / 10,
                    dag.nodes[node]['out_degree'] / 10,
                    1 if (veh_id, node) in self.scheduled_nodes else 0,
                    1 if (veh_id, node) in self.available_nodes else 0
                ]
                state.extend(node_features)
        
        # Pad state to fixed size
        max_state_size = self.num_vehicles * self.max_nodes * 6
        if len(state) < max_state_size:
            state.extend([0] * (max_state_size - len(state)))
        else:
            state = state[:max_state_size]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, node_action, location_action):
        """Execute action and return next state, reward, done"""
        reward = 0
        
        if len(self.available_nodes) == 0:
            return self._get_state(), reward, True
        
        # Convert actions to actual node and location
        veh_id, node = list(self.available_nodes)[node_action % len(self.available_nodes)]
        location = location_action % (self.num_vehicles + self.num_servers)
        
        # Schedule the node
        self.scheduled_nodes.add((veh_id, node))
        
        # Simulate computation time (simplified)
        computation_time = self.dags[veh_id].nodes[node]['computation'] / 10**7
        if location < self.num_vehicles:  # Vehicle execution
            computation_time *= 2  # Slower than servers
        else:  # Server execution
            computation_time *= 0.5  # Faster execution
        
        self.completion_times[veh_id] += computation_time
        
        # Reward based on reduction in completion time
        reward = -computation_time / 100  # Negative reward for time taken
        
        # Update available nodes
        self.available_nodes = self._get_available_nodes()
        self.current_step += 1
        
        done = len(self.available_nodes) == 0 or self.current_step >= 100
        
        return self._get_state(), reward, done

class PPO:
    def __init__(self, state_dim, num_nodes, num_locations, lr=1e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = DVTPPolicyNetwork(state_dim, num_nodes, num_locations)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
    def get_action(self, state, node_mask=None, location_mask=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            node_logits, location_logits, value = self.policy_net(state_tensor, node_mask, location_mask)
            
            node_probs = torch.softmax(node_logits, dim=-1)
            location_probs = torch.softmax(location_logits, dim=-1)
            
            node_dist = torch.distributions.Categorical(node_probs)
            location_dist = torch.distributions.Categorical(location_probs)
            
            node_action = node_dist.sample()
            location_action = location_dist.sample()
            
            log_prob = node_dist.log_prob(node_action) + location_dist.log_prob(location_action)
            
        return (node_action.item(), location_action.item()), log_prob.item(), value.item()
    
    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Get new log probs and values
        node_logits, location_logits, values = self.policy_net(states)
        
        node_probs = torch.softmax(node_logits, dim=-1)
        location_probs = torch.softmax(location_logits, dim=-1)
        
        node_dist = torch.distributions.Categorical(node_probs)
        location_dist = torch.distributions.Categorical(location_probs)
        
        log_probs_new = node_dist.log_prob(actions[:, 0]) + location_dist.log_prob(actions[:, 1])
        
        # PPO loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # Total loss
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def train_dvtp():
    # Environment parameters
    num_vehicles = 3
    num_servers = 2
    max_nodes = 20
    state_dim = num_vehicles * max_nodes * 6  # Simplified state dimension
    
    # Initialize environment and agent
    env = DAGEnvironment(num_vehicles, num_servers, max_nodes)
    agent = PPO(state_dim, max_nodes, num_vehicles + num_servers)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    batch_size = 64
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        states, actions, log_probs, rewards, values = [], [], [], [], []
        
        for step in range(max_steps):
            # Get action from policy
            action, log_prob, value = agent.get_action(state)
            
            # Execute action
            next_state, reward, done = env.step(action[0], action[1])
            
            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Calculate returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + agent.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        if len(states) >= batch_size:
            agent.update(states, actions, log_probs, returns, advantages)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train_dvtp()
