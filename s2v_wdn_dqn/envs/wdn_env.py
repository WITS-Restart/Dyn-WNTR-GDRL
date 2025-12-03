from collections import namedtuple
import random
import sys
import networkx
import numpy as np
import torch

sys.path.append("../..")
from global_state import State
from s2v_wdn_dqn.agents.dueling_scorer_noisy_R2D2.dqn_agent import DQNAgent
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
from wntr.network.model import WaterNetworkModel
from wntr.network.elements import LinkStatus
from wntr.network.io import write_inpfile
from logger import DEFAULT_LOGGER_KEY, LogType, Logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])


def find_touching_nodes(nodes, grid_size=30):
    ids = [int(n[1:]) for n in nodes]
    
    positions = {n: divmod(i - 1, grid_size) for n, i in zip(nodes, ids)}    
    touching = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            r1, c1 = positions[nodes[i]]
            r2, c2 = positions[nodes[j]]
            if (r1 == r2 and abs(c1 - c2) == 1) or (c1 == c2 and abs(r1 - r2) == 1):
                touching.append((nodes[i], nodes[j]))
    
    return touching


def nodes_on_loops(G: networkx.Graph):
    H = G.copy()
    H.remove_edges_from(networkx.bridges(G))
    return {n for comp in networkx.connected_components(H) if len(comp) > 1 for n in comp}


class WDNEnv():
    agent: DQNAgent
    simulation: InteractiveWNTRSimulator
    n_node_features: int
    n_edge_features: int
    normalize_reward: bool = True
    node_list: list
    node_idx: dict
    edge_list: list
    edge_map: dict
    global_timestep: int
    duration: int

    def __init__(self, simulation_file_path=None, normalize_reward=True, double_dqn=False, global_timestep=60, simulation_duration=48600, sensors_percentage=1.0, demand_percentage=0.05, num_leaks=1):
        """
        Initialize the WDN environment with the given parameters.
        :param n_node_features: Number of node features
        :param n_edge_features: Number of edge features
        :param graph: NetworkX graph representing the water distribution network
        :param normalize_reward: Whether to normalize the reward
        """
        #super().__init__()
        self.simulation_file_path = simulation_file_path
        self.simulation = None

        self.normalize_reward = normalize_reward
        self.double_dqn = double_dqn

        wn = WaterNetworkModel(self.simulation_file_path)
        # list of all junction/reservoir/tank names in order
        self.node_list = list(wn.node_name_list)
        self.node_idx  = {name:i for i,name in enumerate(self.node_list)}
        # list of all links (pipes, valves, pumps…)
        self.edge_list = list(wn.link_name_list)
        # map (u_idx,v_idx) → link_name (and vice-versa)
        self.edge_map = {}
        for link_name in self.edge_list:
            link = wn.get_link(link_name)
            u = self.node_idx[link.start_node_name]
            v = self.node_idx[link.end_node_name]
            self.edge_map[(u,v)] = link_name
            self.edge_map[(v,u)] = link_name
        self.global_timestep = global_timestep
        self.duration = simulation_duration
        self.leak_nodes = []
        self.demanding_nodes = []
        self.closed_links = 0

        #self.node_features = ['demand', 'elevation', 'has_elevation', 'has_level', 'has_max_level', 'has_min_level', 'has_overflow', 'has_setting', 'head', 'leak_area', 'leak_demand', 'leak_discharge_coeff', 'leak_status', 'level', 'max_level', 'min_level', 'node_type', 'overflow', 'pressure', 'setting', 'satisfied_demand']
        
        #node_featurs without leak features
        #self.node_features = ['demand', 'elevation', 'has_elevation', 'has_level', 'has_max_level', 'has_min_level', 'has_overflow', 'has_setting', 'head', 'level', 'max_level', 'min_level', 'node_type', 'overflow', 'pressure', 'setting', 'satisfied_demand']
        
        #node_features with less fields
        self.node_features = ['demand', 'head', 'node_type', 'pressure', 'satisfied_demand']
        self.n_node_features = len(self.node_features) + 3 #one-hot for element_type (junction, reservoir, tank) + sensor present/absent
        
        #self.edge_features = ['base_speed', 'diameter', 'flow', 'has_base_speed', 'has_diameter', 'has_headloss', 'has_roughness', 'has_setting', 'has_velocity', 'headloss', 'link_type', 'roughness', 'setting', 'status', 'velocity']
        
        #no status in edge features, given in a separate array
        #self.edge_features = ['base_speed', 'diameter', 'flow', 'has_base_speed', 'has_diameter', 'has_headloss', 'has_roughness', 'has_setting', 'has_velocity', 'headloss', 'link_type', 'roughness', 'setting', 'velocity']
        
        #edge features with less fields
        self.edge_features = ['flow', 'link_type']
        self.n_edge_features = len(self.edge_features) + 3 #one-hot for element_type (pipe, valve, pump) + sensor present/absent

        self.agent = DQNAgent(n_node_features=self.n_node_features, n_edge_features=self.n_edge_features, nstep=1, double_dqn=self.double_dqn, embedding_dim=64, embedding_layers=2, partial_observability=False, target_update="hard")

        unique_edges = sorted({(min(u, v), max(u, v)) for (u, v) in self.edge_map.keys()})
        self.agent.set_valid_edges(np.array(unique_edges, dtype=np.int64))
        self.no_op_action = len(unique_edges)  # E -> no-op action is the last one

        self.wn = wn

        self.episode_count = -1
        self.step_count = 0
        self.episode_log_file = None

        self.last_action = None
        self.retries = 0

        self.prev_phi = None
        self.shaping_gamma = 0.99   
        self.shaping_beta  = 0.2    

        self.early_return_counter = 0
        self.early_return_threshold = 1
        self.all_sources_stranded_round = 0
        self.all_demands_not_satisfied_round = 0
        self.episode_actions = {}

        self.sensors_percentage = sensors_percentage
        self.sensored_nodes = []
        self.sensored_pipes = []

        self.num_leaks = num_leaks
        self.demand_percentage = demand_percentage


        #self.previous_observations = {
        #    'average_demand_satisfaction': 1,
        #    'average_leak_satisfaction': 1,
        #    'fraction_closed_edges': 0,
        #    'fraction_wrongly_isolated_junctions': 0
        #}

        self.reset()

    def get_observation(self) -> tuple:
        try:
            snap = self.simulation.extract_snapshot(scale_values=True)
        except Exception as e:
            print(f"#### Failed to extract snapshot: {e}")
            raise e

        node_feats = [None] * len(self.node_list)
        for name in self.node_list:
            feats = snap['nodes'][name]
            flat = []
            if name in self.sensored_nodes:
                for f in self.node_features:
                    v = feats[f]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        flat.extend(v)
                    else:
                        flat.append(v)
                flat.append(1.0)  # sensor present
            else:
                for f in self.node_features:
                    v = feats[f]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        flat.extend([0.0] * len(v))
                    else:
                        flat.append(0.0)
                flat.append(0.0)  # sensor absent
            node_feats[self.node_idx[name]] = flat
        state = np.asarray(node_feats, dtype=np.float32)


        # --- edge features tensor ---
        edge_feats = []
        edge_status = []
        for (i, j) in self.agent.valid_edges:
            link_name = self.edge_map.get((i, j), self.edge_map.get((j, i)))
            flat = []
            if link_name in self.sensored_pipes:
                for f in self.edge_features:  
                    val = snap['edges'][link_name][f]    
                    if isinstance(val, (list, tuple, np.ndarray)):
                        flat.extend(val)
                    else:
                        flat.append(val)
                flat.append(1.0)  # sensor present
            else:
                for f in self.edge_features:  
                    val = snap['edges'][link_name][f]    
                    if isinstance(val, (list, tuple, np.ndarray)):
                        flat.extend([0.0] * len(val))
                    else:
                        flat.append(0.0)
                flat.append(0.0)  # sensor absent

            edge_feats.append(flat)
            
            status = snap['edges'][link_name]['status']
            edge_status.append(status)

        edge_feats = np.asarray(edge_feats, dtype=np.float32)
        edge_status = np.asarray(edge_status, dtype=np.float32)  # (E,)

        assert edge_feats.shape[0] == len(self.agent.valid_edges), f"edge_feats has {edge_feats.shape[0]} rows, valid_edges has {len(self.agent.valid_edges)}"

        avg_satisfaction = self._compute_avg_satisfaction(snap)
        average_leak = self._compute_average_leak(snap)
        frac_closed = self.closed_links / max(1, len(self.agent.valid_edges))
        isolated_junctions, _ = self.simulation._get_isolated_junctions_and_links()
        frac_isolated = len(set(isolated_junctions)) / max(1, self.simulation._wn.num_junctions)
        num_sources = max(1, len(getattr(self.simulation, "sources", [])))
        frac_stranded = len(self.simulation.get_stranded_sources()) / num_sources

        frac_angry_clients = 1.0 - avg_satisfaction

        global_feats = np.array([frac_angry_clients, frac_closed, frac_isolated, frac_stranded, self.sensors_percentage], dtype=np.float32)
        #global_feats = np.array([frac_angry_clients, average_leak, frac_closed, wrong_iso, frac_stranded], dtype=np.float32)
        return state, edge_feats, edge_status, global_feats

    def reset(self) -> tuple:
        # initialize/reset the hydraulic sim

        assert self.simulation_file_path is not None, "Simulation file path must be provided"
        wn = WaterNetworkModel(self.simulation_file_path)


        #all_nodes = set(wn.junction_name_list + wn.reservoir_name_list + wn.tank_name_list)
        nodes_on_loops_set = nodes_on_loops(wn.get_graph().to_undirected())
        nodes_on_loops_set = nodes_on_loops_set - set(wn.reservoir_name_list + wn.tank_name_list)
        #not_on_loops = all_nodes - nodes_on_loops_set

        #while True:
        #    wn = WaterNetworkModel(self.simulation_file_path)
        #    edges_list = wn.link_name_list
        #    for edge in edges_list:
        #        if random.random() < 0.2:
        #            wn.remove_link(edge)
        #    if networkx.is_connected(wn.get_graph().to_undirected()):
        #        break
        #    else:
        #        print("Generated disconnected network, retrying...")


        wn.options.hydraulic.demand_model = 'PDD'

        wn.add_pattern('constant', [1.0])  # add a constant pattern for demand
        wn.add_pattern('gaussian', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0])  # example pattern

        self.wn = wn
        self.simulation = InteractiveWNTRSimulator(wn)

        self.simulation.plot_network()
        #c = input("Ci piace il network? (y/n) ")
        #if c.lower() == 'y':
        #    write_inpfile(wn, "20x20_branched.inp")
        #from sys import exit
        exit(0)
        self.sensored_nodes = random.sample(self.node_list, int(len(self.node_list) * self.sensors_percentage))
        #self.nodes_sensor_mask = np.array([1 if n in self.sensored_nodes else 0 for n in self.node_list], dtype=np.float32)

        self.sensored_pipes = random.sample(self.edge_list, int(len(self.edge_list) * self.sensors_percentage))
        #self.pipes_sensor_mask = np.array([1 if l in self.sensored_pipes else 0 for l in self.edge_list], dtype=np.float32)

        print(f"Resetting environment for episode {self.episode_count + 1} with global_timestep={self.global_timestep}, duration={self.duration}")
        self.simulation.init_simulation(global_timestep=self.global_timestep, duration=self.duration)

        self.demanding_nodes = []
        self.leak_nodes = []
        self.running_leaks = {}

        self.closed_links = 0

        self.episode_count += 1
        self.step_count = 0

        State.get(DEFAULT_LOGGER_KEY).reset_episode(self.episode_count)

        touching_nodes = True
        while touching_nodes:
        #add leaks and demands to random nodes
            for _ in range(self.num_leaks):
                #node_name = random.choice(list(filter(lambda x: x not in self.leak_nodes, self.wn.junction_name_list)))
                node_name = random.choice(list(filter(lambda x: x not in self.leak_nodes, nodes_on_loops_set)))
                self.leak_nodes.append(node_name)
            
            num_demanding = int(len(self.wn.junction_name_list) * self.demand_percentage)
            for _ in range(num_demanding):
                node_name = random.choice(list(filter(lambda x: x not in self.leak_nodes and x not in self.demanding_nodes, self.wn.junction_name_list)))
                self.demanding_nodes.append(node_name)
            
            touching_pairs = find_touching_nodes(self.leak_nodes + self.demanding_nodes, grid_size=20)
            # keep only pairs where one is a leak and the other is a demanding node
            touching_leak_demand = [
                (a, b) for (a, b) in touching_pairs
                if (a in self.leak_nodes and b in self.demanding_nodes) or (b in self.leak_nodes and a in self.demanding_nodes)
            ]
            if len(touching_leak_demand) == 0:
                 touching_nodes = False
                 for leaking_node in self.leak_nodes:
                    self.simulation.start_leak(leaking_node, leak_area=random.uniform(0.05, 0.1))
                    State.get(DEFAULT_LOGGER_KEY).log(f"@@@@ Added leak at {leaking_node} @@@", msg_type=LogType.EPISODE)
                    print(f"@@@@ Added leak at {leaking_node} @@@")
                    self.running_leaks[leaking_node] = True

                 for demanding_node in self.demanding_nodes:
                    self.simulation.add_demand(demanding_node, base_demand=random.uniform(0.1, 0.3), name='constant')
                    State.get(DEFAULT_LOGGER_KEY).log(f"@@@@ Added demand at {demanding_node} @@@", msg_type=LogType.EPISODE)
            else:
                print(f"Touching leak-demand pairs found {touching_leak_demand}, resetting leaks and demands")
                self.leak_nodes = []
                self.demanding_nodes = []


        self.total_num_steps = self.duration // self.global_timestep
        self.last_action = None
        self.retries = 0
        #self.previous_observations = {
        #    'average_demand_satisfaction': 1,
        #    'average_leak_satisfaction': 1,
        #    'fraction_closed_edges': 0,
        #    'fraction_wrongly_isolated_junctions': 0
        #}

        self.prev_phi = None
        self.early_return_counter = 0
        self.all_sources_stranded_round = 0
        self.all_demands_not_satisfied_round = 0
        self.episode_actions = {}

        self.simulation.step_sim()
        return self.get_observation()

    def _compute_avg_satisfaction(self, snap) -> float:
        satisfaction_nodes = [n['satisfied_demand'] for n in snap['nodes'].values() if n['expected_demand'] > 0.0001]
        average_demand_satisfaction = 1 if len(satisfaction_nodes) == 0 else np.average(satisfaction_nodes)
        return average_demand_satisfaction
    
    def _compute_average_leak(self, snap) -> float:
        leaking_nodes = [snap['nodes'][n]['satisfied_leak'] for n in self.leak_nodes if snap['nodes'][n]['expected_leak'] > 0.0001]
        average_leak_satisfaction = 0 if len(leaking_nodes) == 0 else np.average(leaking_nodes)
        return average_leak_satisfaction

    def _fraction_closed_edges(self) -> float:
        return self.closed_links / max(1, len(self.agent.valid_edges))
    
    def _fraction_wrongly_isolated_junctions(self) -> float:
        isolated_junctions, _ = self.simulation._get_isolated_junctions_and_links()
        return len(set(isolated_junctions) - set(self.leak_nodes)) / max(1, self.simulation._wn.num_junctions)

    def calculate_reward(self, action, prev_snap, post_snap) -> float:
        #timestep = self.simulation.get_sim_time()
        #with open(f"single_round_logs/snapshot-{timestep}.json", "w") as f:
        #    import json
        #    json.dump(snap, f, indent=4)

        previous_average_demand_satisfaction = np.clip(self._compute_avg_satisfaction(prev_snap), 0.0, 1.0)
        previous_average_leak_satisfaction = self._compute_average_leak(prev_snap)

        average_demand_satisfaction = np.clip(self._compute_avg_satisfaction(post_snap), 0.0, 1.0)
        average_leak_satisfaction = self._compute_average_leak(post_snap)

        State.get(DEFAULT_LOGGER_KEY).log(f"Average leak satisfaction: {average_leak_satisfaction}, Average Demand Satisfaction: {average_demand_satisfaction}", msg_type=LogType.EPISODE)
        if len(self.leak_nodes) > 0:
            for node_name in self.leak_nodes:
                node = post_snap['nodes'][node_name]
                State.get(DEFAULT_LOGGER_KEY).log(f"  Leak at {node_name}: area={node['leak_area']}, demand={node['leak_demand']}/{node['expected_leak']} = {node['satisfied_leak']}, status={node['leak_status']}", msg_type=LogType.EPISODE)
                
            leaking_nodes = [(name, val["leak_demand"]) for (name, val) in post_snap['nodes'].items() if val['leak_status'] == 1]
            State.get(DEFAULT_LOGGER_KEY).log(f"Leaking Nodes ({len(leaking_nodes)}): {leaking_nodes}", msg_type=LogType.EPISODE)

        unsatisfied_nodes = [(name, val["satisfied_demand"], val["expected_demand"]) for (name, val) in post_snap['nodes'].items() if val['expected_demand'] > 0.0001 and val['satisfied_demand'] < 0.999]
        State.get(DEFAULT_LOGGER_KEY).log(f"Unsatisfied Nodes ({len(unsatisfied_nodes)}): {unsatisfied_nodes if len(unsatisfied_nodes) < 20 else 'Too many'} ", msg_type=LogType.EPISODE)
        State.get(DEFAULT_LOGGER_KEY).log(f"Demanding nodes ({len(self.demanding_nodes)}): {self.demanding_nodes if len(self.demanding_nodes) < 20 else 'Too many'} ", msg_type=LogType.EPISODE)
        for node_name in self.demanding_nodes:
            node = post_snap['nodes'][node_name]
            State.get(DEFAULT_LOGGER_KEY).log(f"  Demand at {node_name}: expected={node['expected_demand']}, satisfied={node['satisfied_demand']}" , msg_type=LogType.EPISODE)

        frac_closed_edges = self._fraction_closed_edges()        # fraction of closed controllable edges
        frac_wrong_isolated  = self._fraction_wrongly_isolated_junctions()
        stranded_sources = self.simulation.get_stranded_sources()
        num_sources = max(1, len(getattr(self.simulation, "sources", [])))
        frac_stranded = len(stranded_sources) / num_sources

        phi_curr = (
            2.0 * float(average_demand_satisfaction)
            - 2.0 * float(average_leak_satisfaction)
            - 1.0 * float(frac_wrong_isolated)
            - 0.5 * float(frac_closed_edges)
        )
        phi_prev = 0.0 if self.prev_phi is None else self.prev_phi
        shaping = self.shaping_beta * (self.shaping_gamma * phi_curr - phi_prev)

        reward_str = ""

        reward = 0.0
        reward += +5.0 * average_demand_satisfaction
        reward += -0.5 * average_leak_satisfaction
        reward += -1.0 * frac_closed_edges
        reward += -1.0 * frac_wrong_isolated
        reward += -1.0 * frac_stranded

        if action == (self.last_action[0] if self.last_action else None):
            reward -= 5.0

        if abs(previous_average_demand_satisfaction - average_demand_satisfaction) < 0.01 \
        and abs(previous_average_leak_satisfaction - average_leak_satisfaction) < 0.01:
            reward -= 2.0

        reward += shaping
        
        if frac_stranded >= 0.99: #1.0:
            self.all_sources_stranded_round += 1
        else:
            self.all_sources_stranded_round = 0

        if average_demand_satisfaction < 0.01:
            self.all_demands_not_satisfied_round += 1
        else:
            self.all_demands_not_satisfied_round = 0
        

        #make a print of the factors in the reward
        reward_str += f"+5*{average_demand_satisfaction:.3f} (demand) "
        reward_str += f"-5*{average_leak_satisfaction:.3f} (leak) "
        reward_str += f"-1*{frac_closed_edges:.3f} (closed) "
        reward_str += f"-1*{frac_wrong_isolated:.3f} (wrongly isolated) "
        reward_str += f"-2*{frac_stranded:.3f} (stranded) "
        if action == (self.last_action[0] if self.last_action else None):
            reward_str += f"-5 (repeated action) "
        if abs(previous_average_demand_satisfaction - average_demand_satisfaction) < 0.01 \
        and abs(previous_average_leak_satisfaction - average_leak_satisfaction) < 0.01:
            reward_str += f"-2 (no improvement) "
        reward_str += f"+{shaping:.3f} (shaping) "

        good_and_idle = (
            (average_leak_satisfaction <= 1e-5) and
            (average_demand_satisfaction >= 0.95) #and
            #(action == self.no_op_action)
        )
        if good_and_idle:
            self.early_return_counter += 1
        else:
            self.early_return_counter = 0
            
        self.prev_phi = phi_curr

        for leaking_node in self.leak_nodes:
            was_running = self.running_leaks.get(leaking_node, True)
            now_running = post_snap['nodes'][leaking_node]['leak_demand'] > 1e-5

            if was_running and not now_running:
                # leak just stopped
                State.get(DEFAULT_LOGGER_KEY).log(f"@@@@ Leak at {leaking_node} has stopped! @@@", msg_type=LogType.EPISODE)
                print(f"@@@@ Leak at {leaking_node} has stopped! @@@")
                reward += 5.0
                self.running_leaks[leaking_node] = False

            elif (not was_running) and now_running:
                # leak just restarted
                State.get(DEFAULT_LOGGER_KEY).log(f"@@@@ Leak at {leaking_node} has started again! @@@", msg_type=LogType.EPISODE)
                print(f"@@@@ Leak at {leaking_node} has started again! @@@")
                reward -= 7.0
                self.running_leaks[leaking_node] = True

        reward = float(reward) / 10.0


        State.get(DEFAULT_LOGGER_KEY).log(f"Step reward calculation: {reward_str} => Total reward: {reward:.4f}", msg_type=LogType.EPISODE)
        return reward

    def add_random_event(self):
        if random.random() < 0.5:
            if random.random() < 0.5:
                # add a leak with some random parameters
                if len(self.leak_nodes) < 5:
                    node_name = random.choice(list(filter(lambda n: n not in self.leak_nodes and hasattr(self.wn.get_node(n), 'demand_timeseries_list'), self.wn.junction_name_list)))
                    self.simulation.start_leak(node_name, leak_area=0.03)
                    print(f"@@@@ Added leak at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.leak_nodes.append(node_name)
            else:
                # remove a leak
                if len(self.leak_nodes) > 0:
                    node_name = random.choice(self.leak_nodes)
                    self.simulation.stop_leak(node_name)
                    print(f"@@@@ Removed leak at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.leak_nodes.remove(node_name)
        else:
            if random.random() < 0.5:
                # add a random demand to a random node
                node_name = random.choice(list(filter(lambda n: n not in self.demanding_nodes and hasattr(self.wn.get_node(n), 'demand_timeseries_list'), self.wn.junction_name_list)))
                self.simulation.add_demand(node_name, base_demand=0.1, name='gaussian')
                print(f"@@@@ Added demand at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                self.demanding_nodes.append(node_name)
            else:
                # remove a demand
                if len(self.demanding_nodes) > 0:
                    node_name = random.choice(self.demanding_nodes)
                    self.simulation.remove_demand(node_name, name='gaussian')
                    print(f"@@@@ Removed demand at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.demanding_nodes.remove(node_name)

    def step(self, action: int) -> EnvInfo:
        State.get(DEFAULT_LOGGER_KEY).log(f"Episode {self.episode_count} Step {self.step_count}: Taking action {action}", msg_type=LogType.EPISODE)
        temp_last_action = None

        prev_snap = self.simulation.extract_snapshot(scale_values=True)

        self.episode_actions[action] = self.episode_actions.get(action, 0) + 1

        if action == self.no_op_action:
            #print("No-op action taken, advancing simulation.", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            State.get(DEFAULT_LOGGER_KEY).log("No-op action taken, advancing simulation.", msg_type=LogType.EPISODE)
            # no action taken, just return current state
        else:
            u, v = self.agent.valid_edges[action]
            link_name = self.edge_map[(u, v)]
            link = self.simulation._wn.get_link(link_name)
            State.get(DEFAULT_LOGGER_KEY).log(f"Current status: {link_name}->{link.status} (Closed = {LinkStatus.Closed}, open = {LinkStatus.Open}) ", msg_type=LogType.EPISODE)
            is_open = link.status != LinkStatus.Closed # 0=closed, else=open
            State.get(DEFAULT_LOGGER_KEY).log(f"Action: {link_name} {'open' if not is_open else 'close'}", msg_type=LogType.EPISODE)
            if is_open:
                self.simulation.close_pipe(link_name)
                self.closed_links += 1
            else:
                self.simulation.open_pipe(link_name)
                self.closed_links -= 1
            temp_last_action = (action, link_name, 'open' if not is_open else 'close')

        #if random.random() < 0.01:
        #    self.add_random_event()
        try: 
            self.simulation.step_sim()  # advance one hydraulic timestep
        except Exception as e:
            #import traceback
            #traceback.print_exc()
            print(f"#### Simulation step failed: {e}")
            State.get(DEFAULT_LOGGER_KEY).log(f"Simulation step failed: {e}", msg_type=LogType.EPISODE)
            return None
            #self.retries += 1
            #if self.retries < 3:
            #    print(f"####@@@@####\nRetrying simulation step (attempt {self.retries})...\n####@@@@####", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            #    if self.last_action:
            #        _, link_name, act = self.last_action
            #        if act == 'close':
            #            self.simulation.open_pipe(link_name)
            #            self.closed_links -= 1
            #            print(f"Reverted last action: opened {link_name}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            #        else:
            #            self.simulation.close_pipe(link_name)
            #            self.closed_links += 1
            #            print(f"Reverted last action: closed {link_name}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            #        self.last_action = None
            #    return EnvInfo(self.get_observation(), -0.5, False)  # retry the same action
            #else:
            #    print(f"Exceeded maximum retries. Ending episode.", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            #    return EnvInfo(self.get_observation(), -1, True)


        post_snap = self.simulation.extract_snapshot(scale_values=True)

        self.retries = 0
        next_state, next_edge_feats, edge_status, global_feats = self.get_observation()
        reward = self.calculate_reward(action, prev_snap, post_snap)

        _, link_name, act = temp_last_action if temp_last_action else (None, None, None)
        if link_name:
            link = self.simulation._wn.get_link(link_name)
            if link.start_node_name in self.leak_nodes or link.end_node_name in self.leak_nodes:
                State.get(DEFAULT_LOGGER_KEY).log(f"Action affected a leak node ({link.start_node_name} or {link.end_node_name}).", msg_type=LogType.EPISODE)
                if act == 'close':
                    reward += 0.5
                else:
                    reward -= 1.0
            elif link.start_node_name in self.demanding_nodes or link.end_node_name in self.demanding_nodes:
                State.get(DEFAULT_LOGGER_KEY).log(f"Action affected a demanding node ({link.start_node_name} or {link.end_node_name}).", msg_type=LogType.EPISODE)
                if act == 'close':
                    reward -= 0.5
                else:
                    reward += 0.25
            elif link.start_node_name in self.simulation._wn.source_name_list or link.end_node_name in self.simulation._wn.source_name_list:
                State.get(DEFAULT_LOGGER_KEY).log(f"Action affected a source node ({link.start_node_name} or {link.end_node_name}).", msg_type=LogType.EPISODE)
                if act == 'close':
                    reward -= 0.5
                else:
                    reward += 0.25
            else:
                State.get(DEFAULT_LOGGER_KEY).log(f"Action affected a useless node ({link.start_node_name} or {link.end_node_name}).", msg_type=LogType.EPISODE)
                if act == 'close':
                    reward -= 1.0
                else:
                    reward += 0.5


        done = self.simulation.is_terminated()

        self.last_action = temp_last_action


        if not done and self.early_return_counter >= 1: #self.early_return_threshold:
            print("Early return condition met. Ending episode.")
            State.get(DEFAULT_LOGGER_KEY).log(f"Early return condition met (counter={self.early_return_counter}). Ending episode.", msg_type=LogType.EPISODE)
            next_state, next_edge_feats, edge_status, global_feats = self.get_observation()
            reward += ((self.total_num_steps - self.step_count) / self.total_num_steps) * 2

            State.get(DEFAULT_LOGGER_KEY).log(f"Final reward: {reward}", msg_type=LogType.EPISODE)
            return EnvInfo((next_state, next_edge_feats, edge_status, global_feats), reward, True)
        
        if not done and self.all_sources_stranded_round >= 120:
            print("All sources have been stranded for too long. Ending episode.")
            State.get(DEFAULT_LOGGER_KEY).log(f"All sources have been stranded for {self.all_sources_stranded_round} consecutive rounds. Ending episode.", msg_type=LogType.EPISODE)
            next_state, next_edge_feats, edge_status, global_feats = self.get_observation()
            reward -= 1
            
            State.get(DEFAULT_LOGGER_KEY).log(f"Final reward: {reward}", msg_type=LogType.EPISODE)
            return EnvInfo((next_state, next_edge_feats, edge_status, global_feats), reward, True)
        
        if not done and self.all_demands_not_satisfied_round >= 120:
            print("All demands have been unsatisfied for too long. Ending episode.")
            State.get(DEFAULT_LOGGER_KEY).log(f"All demands have been unsatisfied for {self.all_demands_not_satisfied_round} consecutive rounds. Ending episode.", msg_type=LogType.EPISODE)
            next_state, next_edge_feats, edge_status, global_feats = self.get_observation()
            reward -= 1
            State.get(DEFAULT_LOGGER_KEY).log(f"Final reward: {reward}", msg_type=LogType.EPISODE)
            return EnvInfo((next_state, next_edge_feats, edge_status, global_feats), reward, True)

        self.step_count += 1
        State.get(DEFAULT_LOGGER_KEY).log(f"End of step {self.episode_count} Step {self.step_count}: Reward: {reward}, Done: {done}", msg_type=LogType.EPISODE)
        return EnvInfo((next_state, next_edge_feats, edge_status, global_feats), reward, done)

'''


    def different_get_observation(self):
        """
        Returns:
            state:        (N, F_node)     node features (NO dense adjacency concatenation)
            edge_feats:   (E, F_edge)     features only for the E canonical undirected edges
        """
        # ---- Node features (keep what you already had, just don't concat adjacency) ----
        node_feats = []
        for name in self.node_list:
            n = self.wn.get_node(name)
            # Example: keep your exact feature list/order here
            node_feats.append([
                getattr(n, "demand", 0.0) or 0.0,
                getattr(n, "head", 0.0) or 0.0,
                getattr(n, "pressure", 0.0) or 0.0,
                float(getattr(n, "_leak_status", False)),
                getattr(n, "_leak_area", 0.0) or 0.0,
                getattr(n, "_leak_demand", 0.0) or 0.0,
                # add/remove to match your previous node F_node precisely
            ])
        state = np.asarray(node_feats, dtype=np.float32)

        # ---- Sparse edge features aligned to self.agent.valid_edges (E, F_edge) ----
        edge_rows = []
        for (i, j) in self.agent.valid_edges:
            # Map canonical (i,j) back to a physical link name (either direction exists in edge_map)
            link_name = self.edge_map.get((i, j), self.edge_map.get((j, i)))
            l = self.wn.get_link(link_name)
            # Compute velocity if diameter present
            diameter = (getattr(l, "diameter", None) or 0.0)
            flow = getattr(l, "flow", 0.0) or 0.0
            velocity = (abs(flow) * 4.0 / (math.pi * diameter ** 2)) if diameter else 0.0
            status = 0.0 if l.status == LinkStatus.Closed else 1.0

            # IMPORTANT: keep this list/ordering identical to what your model expects (n_edge_features)
            edge_rows.append([
                status,
                flow,
                getattr(l, "headloss", 0.0) or 0.0,
                getattr(l, "roughness", 0.0) or 0.0,
                diameter,
                velocity,
                # add/remove to match your previous F_edge exactly
            ])
        edge_feats = np.asarray(edge_rows, dtype=np.float32)

        return state, edge_feats

    def old_get_observation(self) -> tuple:
        snap = self.simulation.extract_snapshot()
        N = len(self.node_list)

        # --- node features matrix ---
        node_feats = np.zeros((N, self.n_node_features), dtype=float)
        for name, feats in snap['nodes'].items():
            i = self.node_idx[name]
            flat = []
            for f in self.node_features:
                v = feats[f]
                if isinstance(v, (list, tuple, np.ndarray)):
                    # a one‐hot vector: extend with all its entries
                    flat.extend(v)
                else:
                    # a single scalar
                    flat.append(v)
            node_feats[i] = np.array(flat, dtype=float)

        # --- adjacency rows (open pipes only) ---
        adj = np.zeros((N, N), dtype=float)
        for (u,v), link_name in self.edge_map.items():
            status = snap['edges'][link_name]['status']
            adj[u,v] = status  # 1=open, 0=closed

        # stack node_feats ∥ adjacency
        state = np.hstack([node_feats, adj])

        # --- edge features tensor ---
        edge_feats = np.zeros((N, N, self.n_edge_features), dtype=float)
        for (u,v), link_name in self.edge_map.items():
            flat = []
            for f in self.edge_features:
                val = snap['edges'][link_name][f]
                if isinstance(val, (list, tuple, np.ndarray)):
                    flat.extend(val)
                else:
                    flat.append(val)
            edge_feats[u, v] = np.array(flat, dtype=float)

        return state, edge_feats
'''
