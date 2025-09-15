"""Network-based migration corridor analysis and simulation."""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

# Try to import additional network analysis libraries
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print("Warning: python-louvain not installed. Install with: pip install python-louvain")

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network analysis metrics."""
    nodes: int
    edges: int
    density: float
    clustering_coefficient: float
    average_path_length: float
    diameter: int
    centralization: float
    modularity: float
    communities: List[List[str]]
    critical_edges: List[Tuple[str, str, float]]

@dataclass
class CascadeSimulation:
    """Results of cascade simulation."""
    affected_nodes: List[str]
    impact_scores: Dict[str, float]
    propagation_paths: List[List[str]]
    total_impact: float
    simulation_steps: int

class MigrationNetwork:
    """Network analysis for migration corridors and flow patterns."""
    
    def __init__(self, df: pd.DataFrame, 
                 origin_col: str = 'origin_id',
                 dest_col: str = 'dest_id',
                 flow_col: str = 'flow',
                 period_col: str = 'period'):
        """
        Initialize migration network from DataFrame.
        
        Args:
            df: DataFrame with migration flow data
            origin_col: Name of origin column
            dest_col: Name of destination column
            flow_col: Name of flow column
            period_col: Name of period column
        """
        self.df = df.copy()
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.flow_col = flow_col
        self.period_col = period_col
        
        # Build network
        self.G = self._build_network()
        self.metrics = None
        
        logger.info(f"Built migration network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def _build_network(self) -> nx.DiGraph:
        """Build directed network from migration flows."""
        G = nx.DiGraph()
        
        # Aggregate flows across time periods
        flow_aggregate = self.df.groupby([self.origin_col, self.dest_col])[self.flow_col].sum().reset_index()
        
        for _, row in flow_aggregate.iterrows():
            origin = str(row[self.origin_col])
            dest = str(row[self.dest_col])
            flow = float(row[self.flow_col])
            
            # Add edge with flow as weight
            if G.has_edge(origin, dest):
                G[origin][dest]['weight'] += flow
            else:
                G.add_edge(origin, dest, weight=flow)
        
        # Add node attributes
        self._add_node_attributes(G)
        
        return G
    
    def _add_node_attributes(self, G: nx.DiGraph):
        """Add node attributes for analysis."""
        # Calculate in-degree and out-degree
        for node in G.nodes():
            in_degree = G.in_degree(node, weight='weight')
            out_degree = G.out_degree(node, weight='weight')
            
            G.nodes[node]['in_degree'] = in_degree
            G.nodes[node]['out_degree'] = out_degree
            G.nodes[node]['net_flow'] = out_degree - in_degree  # Positive = net outflow
    
    def calculate_network_metrics(self) -> NetworkMetrics:
        """Calculate comprehensive network metrics."""
        if self.metrics is not None:
            return self.metrics
        
        logger.info("Calculating network metrics...")
        
        # Basic metrics
        nodes = self.G.number_of_nodes()
        edges = self.G.number_of_edges()
        density = nx.density(self.G)
        
        # Clustering (convert to undirected for global clustering)
        G_undirected = self.G.to_undirected()
        clustering_coeff = nx.average_clustering(G_undirected, weight='weight')
        
        # Path length and diameter
        if nx.is_weakly_connected(self.G):
            # Convert to undirected for path calculations
            G_undirected = self.G.to_undirected()
            average_path_length = nx.average_shortest_path_length(G_undirected, weight='weight')
            diameter = nx.diameter(G_undirected, weight='weight')
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.weakly_connected_components(self.G), key=len)
            G_sub = self.G.subgraph(largest_cc).to_undirected()
            average_path_length = nx.average_shortest_path_length(G_sub, weight='weight')
            diameter = nx.diameter(G_sub, weight='weight')
        
        # Centralization
        centralization = self._calculate_centralization()
        
        # Community detection
        communities, modularity = self._detect_communities()
        
        # Critical edges (highest betweenness centrality)
        critical_edges = self.identify_critical_corridors(top_k=10)
        
        self.metrics = NetworkMetrics(
            nodes=nodes,
            edges=edges,
            density=density,
            clustering_coefficient=clustering_coeff,
            average_path_length=average_path_length,
            diameter=diameter,
            centralization=centralization,
            modularity=modularity,
            communities=communities,
            critical_edges=critical_edges
        )
        
        return self.metrics
    
    def _calculate_centralization(self) -> float:
        """Calculate network centralization (degree-based)."""
        degrees = [d for n, d in self.G.degree(weight='weight')]
        if not degrees:
            return 0.0
        
        max_degree = max(degrees)
        n = len(degrees)
        
        if n <= 1:
            return 0.0
        
        # Freeman's centralization formula
        centralization = sum(max_degree - d for d in degrees) / ((n - 1) * (n - 2))
        return centralization
    
    def _detect_communities(self) -> Tuple[List[List[str]], float]:
        """Detect communities using Louvain algorithm."""
        if not HAS_COMMUNITY:
            logger.warning("python-louvain not available, using connected components")
            communities = list(nx.weakly_connected_components(self.G))
            return [list(comm) for comm in communities], 0.0
        
        # Convert to undirected for community detection
        G_undirected = self.G.to_undirected()
        
        # Detect communities
        partition = community_louvain.best_partition(G_undirected)
        modularity = community_louvain.modularity(partition, G_undirected)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        return list(communities.values()), modularity
    
    def identify_critical_corridors(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Identify most critical migration corridors using betweenness centrality."""
        edge_betweenness = nx.edge_betweenness_centrality(self.G, weight='weight')
        
        # Sort by betweenness centrality
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        
        critical_edges = []
        for (u, v), centrality in sorted_edges[:top_k]:
            weight = self.G[u][v]['weight']
            critical_edges.append((u, v, centrality))
        
        return critical_edges
    
    def calculate_node_centrality(self, method: str = 'betweenness') -> Dict[str, float]:
        """Calculate node centrality metrics."""
        if method == 'betweenness':
            return nx.betweenness_centrality(self.G, weight='weight')
        elif method == 'closeness':
            return nx.closeness_centrality(self.G, distance='weight')
        elif method == 'eigenvector':
            return nx.eigenvector_centrality(self.G, weight='weight')
        elif method == 'pagerank':
            return nx.pagerank(self.G, weight='weight')
        elif method == 'in_degree':
            return dict(self.G.in_degree(weight='weight'))
        elif method == 'out_degree':
            return dict(self.G.out_degree(weight='weight'))
        else:
            raise ValueError(f"Unknown centrality method: {method}")
    
    def predict_cascade_effects(self, 
                              shock_node: str, 
                              shock_magnitude: float,
                              propagation_model: str = 'pagerank',
                              max_steps: int = 10) -> CascadeSimulation:
        """
        Simulate cascade effects of a shock at one node.
        
        Args:
            shock_node: Node where shock occurs
            shock_magnitude: Magnitude of shock (0-1)
            propagation_model: 'pagerank', 'random_walk', or 'epidemic'
            max_steps: Maximum propagation steps
            
        Returns:
            CascadeSimulation results
        """
        if shock_node not in self.G.nodes():
            raise ValueError(f"Node {shock_node} not found in network")
        
        logger.info(f"Simulating cascade from {shock_node} with magnitude {shock_magnitude}")
        
        if propagation_model == 'pagerank':
            return self._pagerank_propagation(shock_node, shock_magnitude)
        elif propagation_model == 'random_walk':
            return self._random_walk_propagation(shock_node, shock_magnitude, max_steps)
        elif propagation_model == 'epidemic':
            return self._epidemic_propagation(shock_node, shock_magnitude, max_steps)
        else:
            raise ValueError(f"Unknown propagation model: {propagation_model}")
    
    def _pagerank_propagation(self, shock_node: str, shock_magnitude: float) -> CascadeSimulation:
        """Simulate propagation using PageRank algorithm."""
        # Create personalization vector
        personalization = {node: 0 for node in self.G.nodes()}
        personalization[shock_node] = shock_magnitude
        
        # Calculate PageRank with personalization
        pagerank_scores = nx.pagerank(self.G, personalization=personalization, weight='weight')
        
        # Convert to impact scores
        impact_scores = {node: score * shock_magnitude for node, score in pagerank_scores.items()}
        
        # Identify affected nodes (above threshold)
        threshold = shock_magnitude * 0.1
        affected_nodes = [node for node, score in impact_scores.items() if score > threshold]
        
        return CascadeSimulation(
            affected_nodes=affected_nodes,
            impact_scores=impact_scores,
            propagation_paths=[],  # PageRank doesn't provide paths
            total_impact=sum(impact_scores.values()),
            simulation_steps=1
        )
    
    def _random_walk_propagation(self, shock_node: str, shock_magnitude: float, max_steps: int) -> CascadeSimulation:
        """Simulate propagation using random walk."""
        impact_scores = {node: 0 for node in self.G.nodes()}
        impact_scores[shock_node] = shock_magnitude
        
        current_nodes = [shock_node]
        propagation_paths = [[shock_node]]
        
        for step in range(max_steps):
            next_nodes = []
            new_paths = []
            
            for node, path in zip(current_nodes, propagation_paths):
                if node in self.G:
                    neighbors = list(self.G.successors(node))
                    if neighbors:
                        # Distribute impact to neighbors
                        total_weight = sum(self.G[node][neighbor]['weight'] for neighbor in neighbors)
                        
                        for neighbor in neighbors:
                            edge_weight = self.G[node][neighbor]['weight']
                            propagation_strength = (edge_weight / total_weight) * impact_scores[node] * 0.5
                            
                            if propagation_strength > 0.01:  # Threshold
                                impact_scores[neighbor] += propagation_strength
                                next_nodes.append(neighbor)
                                new_paths.append(path + [neighbor])
            
            if not next_nodes:
                break
            
            current_nodes = next_nodes
            propagation_paths = new_paths
        
        # Identify affected nodes
        affected_nodes = [node for node, score in impact_scores.items() if score > shock_magnitude * 0.05]
        
        return CascadeSimulation(
            affected_nodes=affected_nodes,
            impact_scores=impact_scores,
            propagation_paths=propagation_paths,
            total_impact=sum(impact_scores.values()),
            simulation_steps=max_steps
        )
    
    def _epidemic_propagation(self, shock_node: str, shock_magnitude: float, max_steps: int) -> CascadeSimulation:
        """Simulate propagation using epidemic model (SIR)."""
        # Initialize states: 0=susceptible, 1=infected, 2=recovered
        states = {node: 0 for node in self.G.nodes()}
        states[shock_node] = 1
        
        impact_scores = {node: 0 for node in self.G.nodes()}
        impact_scores[shock_node] = shock_magnitude
        
        infected_nodes = [shock_node]
        propagation_paths = [[shock_node]]
        
        for step in range(max_steps):
            new_infected = []
            new_paths = []
            
            for node in infected_nodes:
                if states[node] == 1:  # Currently infected
                    neighbors = list(self.G.successors(node))
                    
                    for neighbor in neighbors:
                        if states[neighbor] == 0:  # Susceptible
                            # Calculate infection probability based on edge weight
                            edge_weight = self.G[node][neighbor]['weight']
                            max_weight = max([self.G[node][n]['weight'] for n in neighbors], default=1)
                            infection_prob = (edge_weight / max_weight) * 0.3  # Base infection rate
                            
                            if np.random.random() < infection_prob:
                                states[neighbor] = 1
                                impact_scores[neighbor] = impact_scores[node] * 0.8
                                new_infected.append(neighbor)
                                
                                # Find path to this node
                                for path in propagation_paths:
                                    if path[-1] == node:
                                        new_paths.append(path + [neighbor])
                                        break
                    
                    # Recover after one step
                    states[node] = 2
            
            if not new_infected:
                break
            
            infected_nodes.extend(new_infected)
            propagation_paths.extend(new_paths)
        
        affected_nodes = [node for node, state in states.items() if state in [1, 2]]
        
        return CascadeSimulation(
            affected_nodes=affected_nodes,
            impact_scores=impact_scores,
            propagation_paths=propagation_paths,
            total_impact=sum(impact_scores.values()),
            simulation_steps=max_steps
        )
    
    def find_alternative_routes(self, origin: str, destination: str, max_paths: int = 5) -> List[List[str]]:
        """Find alternative migration routes between two nodes."""
        if origin not in self.G.nodes() or destination not in self.G.nodes():
            return []
        
        try:
            # Find shortest paths with different algorithms
            paths = []
            
            # Dijkstra shortest path
            try:
                shortest_path = nx.shortest_path(self.G, origin, destination, weight='weight')
                paths.append(shortest_path)
            except nx.NetworkXNoPath:
                pass
            
            # Yen's k-shortest paths
            try:
                k_paths = list(nx.shortest_simple_paths(self.G, origin, destination, weight='weight'))
                paths.extend(k_paths[:max_paths])
            except nx.NetworkXNoPath:
                pass
            
            # Remove duplicates while preserving order
            unique_paths = []
            seen = set()
            for path in paths:
                path_tuple = tuple(path)
                if path_tuple not in seen:
                    unique_paths.append(path)
                    seen.add(path_tuple)
            
            return unique_paths[:max_paths]
            
        except Exception as e:
            logger.error(f"Error finding alternative routes: {e}")
            return []
    
    def analyze_vulnerability(self) -> Dict[str, float]:
        """Analyze network vulnerability to node/edge removal."""
        vulnerability_scores = {}
        
        # Test node removal
        original_metrics = self.calculate_network_metrics()
        
        for node in list(self.G.nodes()):
            # Create copy without node
            G_copy = self.G.copy()
            G_copy.remove_node(node)
            
            if G_copy.number_of_nodes() > 0:
                # Calculate metrics for reduced network
                reduced_density = nx.density(G_copy)
                density_reduction = original_metrics.density - reduced_density
                vulnerability_scores[f'node_{node}'] = density_reduction
        
        return vulnerability_scores
    
    def export_network_data(self, format: str = 'json') -> Union[Dict, str]:
        """Export network data in various formats."""
        if format == 'json':
            # Convert to JSON-serializable format
            data = {
                'nodes': [
                    {
                        'id': node,
                        'in_degree': data.get('in_degree', 0),
                        'out_degree': data.get('out_degree', 0),
                        'net_flow': data.get('net_flow', 0)
                    }
                    for node, data in self.G.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'weight': data['weight']
                    }
                    for u, v, data in self.G.edges(data=True)
                ],
                'metrics': self.metrics.__dict__ if self.metrics else {}
            }
            return data
        elif format == 'gexf':
            # Export as GEXF for Gephi
            nx.write_gexf(self.G, 'migration_network.gexf')
            return 'migration_network.gexf'
        else:
            raise ValueError(f"Unsupported format: {format}")

def create_network_from_flows(df: pd.DataFrame) -> MigrationNetwork:
    """Convenience function to create migration network from flows DataFrame."""
    return MigrationNetwork(df)

def analyze_migration_corridors(df: pd.DataFrame) -> Dict:
    """Comprehensive migration corridor analysis."""
    network = MigrationNetwork(df)
    metrics = network.calculate_network_metrics()
    
    # Calculate various centrality measures
    betweenness = network.calculate_node_centrality('betweenness')
    pagerank = network.calculate_node_centrality('pagerank')
    in_degree = network.calculate_node_centrality('in_degree')
    out_degree = network.calculate_node_centrality('out_degree')
    
    # Identify critical corridors
    critical_corridors = network.identify_critical_corridors(top_k=10)
    
    # Analyze vulnerability
    vulnerability = network.analyze_vulnerability()
    
    return {
        'network_metrics': metrics.__dict__,
        'centrality_measures': {
            'betweenness': betweenness,
            'pagerank': pagerank,
            'in_degree': in_degree,
            'out_degree': out_degree
        },
        'critical_corridors': critical_corridors,
        'vulnerability_analysis': vulnerability,
        'network_data': network.export_network_data('json')
    }

if __name__ == "__main__":
    # Test network analysis
    print("Testing migration network analysis...")
    
    # Create test data
    test_data = pd.DataFrame({
        'origin_id': ['A', 'B', 'C', 'D', 'A', 'B'],
        'dest_id': ['B', 'C', 'D', 'A', 'C', 'D'],
        'flow': [100, 150, 200, 80, 120, 90],
        'period': ['2020-01', '2020-01', '2020-01', '2020-01', '2020-01', '2020-01']
    })
    
    # Create network
    network = MigrationNetwork(test_data)
    
    # Calculate metrics
    metrics = network.calculate_network_metrics()
    print(f"Network metrics: {metrics.nodes} nodes, {metrics.edges} edges, density {metrics.density:.3f}")
    
    # Test cascade simulation
    cascade = network.predict_cascade_effects('A', 0.5, 'pagerank')
    print(f"Cascade simulation: {len(cascade.affected_nodes)} nodes affected, total impact {cascade.total_impact:.3f}")
    
    # Find alternative routes
    routes = network.find_alternative_routes('A', 'D')
    print(f"Alternative routes from A to D: {routes}")
    
    print("Migration network analysis test completed!")
