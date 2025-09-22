import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from geopy.distance import geodesic

class SchoolBoardNode:
    def __init__(self, **kwargs):
        self.board_id = kwargs.get('board_id')
        self.name = kwargs.get('name')
        self.region = kwargs.get('region')
        self.language = kwargs.get('language')
        self.school_size = kwargs.get('school_size')
        self.location = kwargs.get('location')
        self.funding = kwargs.get('funding')
        self.enrolment = kwargs.get('enrolment')
        self.node_size = kwargs.get('node_size', 10) 


class SchoolBoardNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.boards = {}

    def load_school_data(self, file_path):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        self.df = df
    
    def build_nodes(self):
        df = self.df
        
        for _, row in df.iterrows():
            board_id = row["Board Number"]
            name = row["Board Name"]
            region = row["Region"]
            language = row["Board Language"]
            location = (float(row["Latitude"]), float(row["Longitude"]))
            school_size = row["Total_norm"]
            funding = row['Funding_norm']
            funding_orig = row['Funding']
            enrolment = row['Enrolment_norm']
            enrolment_orig = row['Enrolment']
            node_size = row["Total"]
            
            node = SchoolBoardNode(**{
                "board_id": board_id,
                "name": name, 
                "region": region, 
                "language": language, 
                "school_size": school_size,
                "location": location,
                "funding": funding,
                "funding_orig": funding_orig,
                "enrolment": enrolment,
                "enrolment_orig": funding_orig,
                "node_size": node_size
            })
            self.boards[board_id] = node
            self.graph.add_node(
                board_id,
                name=name,
                region=region,
                language=language,
                school_size=school_size,
                location=location,
                funding=funding,
                funding_orig=funding_orig,
                enrolment=enrolment,
                enrolment_orig=enrolment_orig,
                node_size=node_size,
            )
            
            
        d = {n: data['node_size'] for n, data in self.graph.nodes(data=True)}
        nx.set_node_attributes(self.graph, d, 'size')
        
    def build_edges_by_proximity(self, max_distance_km=50):
        ids = list(self.boards.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                loc1 = self.boards[ids[i]].location
                loc2 = self.boards[ids[j]].location
                distance = geodesic(loc1, loc2).km
                
                lan1 = self.boards[ids[i]].language
                lan2 = self.boards[ids[j]].language
                if distance <= max_distance_km:
                    weight = 100 / (1 + distance) # weight subject to change
                    self.graph.add_edge(ids[i], ids[j], weight=weight)

    
    def build_edges_by_region(self, distance_threshold_km=100):
        region_groups = {}
        for board_id, board in self.boards.items():
            key = board.region
            if key not in region_groups:
                region_groups[key] = []
            region_groups[key].append(board_id)
        
        self.region_groups = region_groups
        count = 0
        for region, board_ids in region_groups.items():
            for i in range(len(board_ids)):
                for j in range(i + 1, len(board_ids)):
                    
                    # check if the grographic distance is small enough and if edge already exists
                    if (geodesic(self.boards[board_ids[i]].location, self.boards[board_ids[j]].location).km <= distance_threshold_km) and \
                        (not self.graph.has_edge(board_ids[i], board_ids[j])):
                        self.graph.add_edge(board_ids[i], board_ids[j], weight=1)
                    
    def refine_edge_weights(self):
        # to-do as exercise
        for u, v, data in self.graph.edges(data=True):
            board_u = self.boards[u]
            board_v = self.boards[v]
            
            # Calculate similarity based on funding, size and enrolment
            funding_diff = abs(board_u.funding - board_v.funding)
            size_diff = abs(board_u.school_size - board_v.school_size)
            enrolment_diff = abs(board_u.enrolment - board_v.enrolment)
            
            # Similarity score (the smaller the difference, the higher the similarity)
            similarity_score = 1 / (1 + funding_diff + size_diff + enrolment_diff)
            
            # Update edge weight (you can adjust the formula as needed)
            data['weight'] *= (1 + similarity_score)
 


def select_initial_adopters(network, n_initial_adopters=5, seed=None):
    if seed is not None:
        random.seed(seed)
    initial = []
    num_per_region = max(1, n_initial_adopters // len(network.region_groups))
    additional = n_initial_adopters - num_per_region * len(network.region_groups)
    
    for _, group in network.region_groups.items():
        # handle case where region has fewer boards than num_per_region
        act_num_per_region = min(num_per_region, len(group))
        additional += (num_per_region - act_num_per_region)
        selected = random.sample(group, act_num_per_region)
        initial.extend(selected)
        
    if additional > 0:
        all_boards = [b for boards in network.region_groups.values() for b in boards if b not in initial]
        initial.extend(random.sample(all_boards, additional))
    return initial

def get_unadopted_boards_info(state_dict, G):
    unadopted = [n for n, s in state_dict.items() if s == 'S']  # nodes still Susceptible
    
    rows = []
    for n in unadopted:
        data = G.nodes[n]
        rows.append({
            "Board ID": n,
            "Board Name": data.get("name"),
            "Region": data.get("region"),
            "Language": data.get("language"),
            "Funding": data.get("funding_orig"),
            "Enrolment": data.get("enrolment_orig"),
            "Total Schools": data.get("node_size"),
        })
    return pd.DataFrame(rows)

def show_unadopted_boards(state_dict, G, title="Unadopted School Boards"):
    df_unadopted = get_unadopted_boards_info(state_dict, G)
    df_unadopted.index = df_unadopted.index + 1

    st.header(title)
    st.dataframe(df_unadopted)

    if not df_unadopted.empty:
        region_counts = df_unadopted['Region'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Unadopted Boards by Region")
        st.pyplot(fig1)

        language_counts = df_unadopted['Language'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Unadopted Boards by Language")
        st.pyplot(fig2)

    
class DiffusionModel:
    def __init__(self, G):
        self.G = G
        self.get_simulation_parameters()
        
    def get_simulation_parameters(self):
        # Heterogeneous thresholds and incubation periods
        thresholds = {}
        timers = {}
        for n in self.G.nodes():  
            funding = self.G.nodes(data=True)[n].get('funding')
            enrolment = self.G.nodes(data=True)[n].get('enrolment')
            size = self.G.nodes(data=True)[n].get('school_size')
            
            thresholds[n] = max(0.1, 0.5 - (funding + enrolment + size))
            timers[n] = 1 + int(4 * (1 - (funding + enrolment + size)))
        self.thresholds = thresholds
        self.timers = timers

    def create_initial_state_dict(self, default='S'):
        return {n: default for n in self.G.nodes()}
    
    def count_states(self, state_dict):
        counts = defaultdict(int)
        for s in state_dict.values():
            counts[s] += 1
        return dict(counts)
    
    def history_to_dataframe(self, history_counts):
        keys = sorted({k for d in history_counts for k in d.keys()})
        rows = []
        for d in history_counts:
            rows.append({k: d.get(k, 0) for k in keys})
        return pd.DataFrame(rows)
    
    def plot_counts(self, counts_history, title=None, ax=None):
        """Plot time series of counts (list of dicts)."""
        df = self.history_to_dataframe(counts_history)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,4))
        df.plot(ax=ax)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        if title:
            ax.set_title(title)
        ax.grid(True)
        plt.show()
        
    def simple_diffusion(self, initial_adopters, transmission_prob=0.1, steps=50, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        state = self.create_initial_state_dict(default='S')
        for n in initial_adopters:
            if n in state:
                state[n] = 'I'
        node_history = [state.copy()]
        counts_history = [self.count_states(state)]
        for t in range(steps):
            new_state = state.copy()
            to_infect = set()
            for u in self.G.nodes():
                if state[u] == 'I':
                    for v in self.G.neighbors(u):
                        if state[v] == 'S':
                            if np.random.rand() < transmission_prob:
                                to_infect.add(v)
            for v in to_infect:
                new_state[v] = 'I'
            state = new_state
            node_history.append(state.copy())
            counts_history.append(self.count_states(state))
            if all(s != 'S' for s in state.values()):
                break
        return node_history, counts_history

    def threshold_diffusion(self, initial_adopters, steps=50, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        state = self.create_initial_state_dict(default='S')
        for n in initial_adopters:
            if n in state:
                state[n] = 'I'

        node_history = [state.copy()]
        counts_history = [self.count_states(state)]
        
        for t in range(steps):
            print(f"Step {t+1}/{steps}")
            new_state = state.copy()
            for n in self.G.nodes():
                if state[n] == 'S':
                    neighbors = list(self.G.neighbors(n))
                    frac = (sum(1 for nbr in neighbors if state[nbr] == 'I') / len(neighbors)) if neighbors else 0
                    # print(f"Node {n} - Neighbors: {len(neighbors)}, Infected Neighbors: {sum(1 for nbr in neighbors if state[nbr] == 'I')}, Fraction: {frac:.2f}, Threshold: {thresholds_used[n]:.2f}")
                    if frac >= self.thresholds[n]:
                        new_state[n] = 'I'
            state = new_state
            node_history.append(state.copy())
            counts_history.append(self.count_states(state))
            # if node_history[-1] == node_history[-2]:
            #     break
            
        return node_history, counts_history
    
   
    def sir_diffusion(self, initial_adopters, informed_prob=0.2, recovery_prob=0.3, steps=20, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        state = self.create_initial_state_dict(default='S')

        # Initialize with adopters
        for n in initial_adopters:
            state[n] = "I"

        node_history = [state.copy()]
        counts_history = [self.count_states(state)]

        for t in range(steps):
            new_states = state.copy()

            for node in self.G.nodes():
                if state[node] == "S":
                    can_be_informed = False
                    # After exposure, board may adopt with some probability
                    for nbr in self.G.neighbors(node):
                        if state[nbr] == "I":
                            can_be_informed = True
                    if can_be_informed:
                        if random.random() < informed_prob:
                            new_states[node] = "I"

                elif state[node] == "I":
                    # After adoption, board eventually stops influencing
                    if random.random() < recovery_prob:
                        new_states[node] = "R"

            state = new_states
            node_history.append(state.copy())
            counts_history.append(self.count_states(state))

        return node_history, counts_history     

    def seir_diffusion(self, initial_adopters, recovery_prob=0.2, steps=20, seed=None):
        """
        Assumptions:
            1. Exposure probability (S->E) depends on neighbors depends on the influence power (how much a board can influence its neighbors) of neighbors.
            2. Influence power depends on edge weight weighted by board size 
            2. Heterogenous waiting periods (E->I) based on funding, enrolment, and school size. 
                Boards with more funding, enrolment, and larger size decide faster.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        state = self.create_initial_state_dict(default='S')

        for n in initial_adopters:
            state[n] = "I"

        node_history = [state.copy()]
        counts_history = [self.count_states(state)]

        for t in range(steps):
            new_states = state.copy()

            for node in self.G.nodes():
                if state[node] == "S":
                    influence = 0
                    for nbr in self.G.neighbors(node):
                        if state[nbr] == "I":
                            influence += self.G[node][nbr]["weight"] * self.G.nodes[nbr]["school_size"]

                    prob_exposure = min(1.0, influence / (1 + influence))
                    # print(f"Node {node} - Influence: {influence}, Prob Exposure: {prob_exposure:.2f}")

                    if random.random() < prob_exposure:
                        new_states[node] = "E"

                elif state[node] == "E":
                    self.timers[node] -= 1
                    if self.timers[node] <= 0:
                        new_states[node] = "I"

                elif state[node] == "I":
                    # After adoption, board eventually stops influencing
                    if random.random() < recovery_prob:
                        new_states[node] = "R"

            state = new_states
            node_history.append(state.copy())
            counts_history.append(self.count_states(state))

        return node_history, counts_history
       
