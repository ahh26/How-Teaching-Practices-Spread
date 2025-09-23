import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from helpers import SchoolBoardNetwork, DiffusionModel, select_initial_adopters,get_unadopted_boards_info,show_unadopted_boards

data_folder='./source' 

network = SchoolBoardNetwork()
network.load_school_data(data_folder + "/board_summary.csv")
network.build_nodes()
network.build_edges_by_proximity(max_distance_km=70)
network.build_edges_by_region(distance_threshold_km=100)
network.refine_edge_weights()

G=network.graph

with st.sidebar:
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home","Data Visualization","Simulation","School Board Info", "Feeling Confused?"],
    icons = ["house","bar-chart","activity","book"],
    orientation="vertical",
    default_index = 0,
)


if selected == "Home":
    st.header('How Teaching Practices Spread')
    st.subheader('Modeling Educational Diffusion in School Board Networks')
    st.write("This app explores how teaching practices spread through Ontario’s school board networks using data visualization and simulation techniques. Users can see how information spread, experiment with different diffusion models, and identify boards that are underserved or isolated — helping educators and policymakers design more equitable strategies for sharing knowledge.")

    st.markdown("### Explore the App")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div style="background-color:#FFDAB9; padding:15px; border-radius:10px">'
            '<h4>Data Visualization</h4>'
            '<p>Explore different stats of school boards through charts and graphs.</p>'
            '</div>', 
            unsafe_allow_html=True
        )
    with col2:
       st.markdown(
            '<div style="background-color:#E0FFFF; padding:15px; border-radius:10px">'
            '<h4>Simulation</h4>'
            '<p>Run different diffusion models to see how teaching practices spread among boards.</p>'
            '</div>', 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div style="background-color:#FFFACD; padding:15px; border-radius:10px">'
            '<h4>School Board Info</h4>'
            '<p>Check out the basic information of all school boards.</p>'
            '</div>', 
            unsafe_allow_html=True
        )
    
    
    st.markdown("### Why this matters")
    st.write("School boards don’t operate in isolation. Their connections shape how quickly new programs, technologies, and teaching practices reach students. By visualizing these networks, we can see inequities and design smarter interventions.")
    
    
    st.markdown("### Notable trends")
    st.write("A clear finding from the network and data visualization is that North Region school boards are often isolated, with fewer connections to other school boards and a slower adoption rate. Even though these school boards already receive the highest funding, many remain unadopted, showing that funding alone is not sufficient. Stronger connections and collaboration are also very important. Possible approaches include selecting certain North Region school boards as initial adopters in diffusion programs or by forming connections for the isolated school boards through joint initiatives and partnerships. These targeted strategies could help knowledge and best practices spread more effectively across the network, ensuring that all boards benefit from innovations.")

    
elif selected == "Data Visualization":
    df=pd.read_csv(data_folder+'/board_summary.csv')
    
    c1, c2= st.columns(2)
    # c3, c4= st.columns(2)
    c3=st.columns(1)[0]
    c4=st.columns(1)[0]
    

    with st.container():
        c1.write("Average Funding by Region")
        c2.write("Number of Schools per Region")

    with st.container():
        c3.write("Enrolment vs. Funding")
        # c4.write("c4")
        
    with st.container():
        c4.write("Region vs. Funding")
        

    with c1:
        avg_funding_by_region = df.groupby("Region")["Funding"].sum().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(x="Region", y="Funding", data=avg_funding_by_region, ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
           
    with c2:
        schools_by_region = df.groupby('Region')['Total'].sum().reset_index()
        plt.figure(figsize=(8, 5))
        fig2, ax2=plt.subplots()
        sns.barplot(x="Region",y="Total",data=schools_by_region,ax=ax2)
        plt.ylabel("Number of Schools")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        

    with c3:
        fig3,ax3=plt.subplots()
        sns.scatterplot(data=df, x='Enrolment', y='Funding', hue='Region', palette='deep',ax=ax3)
        plt.tight_layout()
        st.pyplot(fig3)

    with c4:
        fig4,ax4 = plt.subplots(figsize=(10, 6))
        sns.swarmplot(data=df, x='Region', y='Funding', hue='Board Language', dodge=True)
        plt.xticks(rotation=45)
        plt.ylabel("Funding")
        plt.title("Funding by Region")
        plt.tight_layout()
        st.pyplot(fig4)
        
elif selected == "Simulation":
    with st.sidebar:
        st.header("Parameters")

        option = st.selectbox(
            'Diffusion method',
            ('Simple', 'Threshold', 'SIR', "SEIR"))
    

        max_steps = st.slider(
            label="Max steps",
            min_value=0,
            max_value=100,
            value=20,
            step=10,
        )

        random_seed = st.number_input(label="Random seed", value=42)
        
        if option == "Simple":
            transmission_prob = st.slider("Transmission Probability", 0.0, 1.0, 0.1, 0.01)
        elif option == "SIR":
            informed_prob = st.slider("Informed Probability", 0.0, 1.0, 0.05, 0.01)
            recovery_prob = st.slider("Recovery Probability", 0.0, 1.0, 0.02, 0.01)
        elif option == "SEIR":
            recovery_prob = st.slider("Recovery Probability", 0.0, 1.0, 0.02, 0.01)
            
        n_initial_adopters = st.slider(
            "Number of default initial adopters",
            min_value=0,
            max_value=len(G.nodes()),
            value=5,
            step=1
        )

        use_default = st.checkbox("Use default initial adopters", value=True)

        board_options_display = [
            f"{data['name']} ({data['region']}, {data['language']})" 
            for node_id, data in G.nodes(data=True)
        ]

        if use_default:
            initial_adopters = select_initial_adopters(network, n_initial_adopters=n_initial_adopters, seed=random_seed)
        else:
            selected_boards = st.multiselect(
                "Select Initial Adopters",
                options=board_options_display,
                default=[]
            )
         
            display_to_id = {f"{data['name']} ({data['region']}, {data['language']})": node_id for node_id, data in G.nodes(data=True)}
            initial_adopters = [display_to_id[s] for s in selected_boards]

    initial_state = {node: ('I' if node in initial_adopters else 'S') for node in G.nodes()}
    model = DiffusionModel(G)
    
    def st_plot_counts(counts_history, title=None):
        """Plot time series of counts (list of dicts) in Streamlit."""
        df = model.history_to_dataframe(counts_history)
        fig, ax = plt.subplots(figsize=(7,4))

        # todo: specify colors for each state
        state_colors = {'S': 'lightgray', 'E': 'tab:orange', 'I': 'tab:blue', 'R': 'tab:green'}
        for state, color in state_colors.items():
            if state in df.columns:
                df[state].plot(ax=ax, color=color)
        # todo: specify the order of the legend
        handles, labels=ax.get_legend_handles_labels()
        order = ["S", "E", "I", "R"]
        label_dict={"S": "Susceptible","E": "Exposed", "I": "Informed", "R":"Recovered"}
        ordered_handles = [handles[labels.index(o)] for o in order if o in labels]
        ordered_labels = [label_dict[o] for o in order if o in labels]
        ax.legend(ordered_handles, ordered_labels, loc='upper right')
        
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        
        if title:
            ax.set_title(title)
        ax.grid(True)
        st.pyplot(fig)
   

    def st_plot_states_helper(state_dict, step, diffusion_plot, figsize=(14, 10)):
        """Plot the network with nodes colored by their state in Streamlit."""
        plt.figure(figsize=figsize)

        node_mapping = {node: G.nodes[node]['name'] for node in G.nodes()}

        G_relabel = nx.relabel_nodes(G, node_mapping)

        pos = {board: (data['location'][1], data['location'][0]) for board, data in G_relabel.nodes(data=True)}

        node_sizes = [data['size']*5 for _, data in G_relabel.nodes(data=True)]

        state_to_color = {'S': 'lightgray', 'E': 'tab:orange', 'I': 'tab:blue', 'R': 'tab:green'}
        colors = [state_to_color.get(state_dict.get(n,'S'),'lightgray') for n in G.nodes()]
        
        
        nx.draw(G_relabel, pos=pos, with_labels=True, node_size=node_sizes, node_color=colors, font_size=5, edge_color='gray', alpha=0.7)
        
        #legend
        legend_elements = [Patch(facecolor=color, edgecolor='k', label=label) 
                   for label, color in zip(["Susceptible", "Exposed", "Informed", "Recovered"], 
                                           ["lightgray", "tab:orange", "tab:blue", "tab:green"])]
        plt.legend(handles=legend_elements, loc='upper right')

        
        plt.title(f"Network State at step {step}")
        if diffusion_plot:
            diffusion_plot.pyplot(plt)
        else:
            diffusion_plot=st.pyplot(plt)
        
        plt.close()
        return diffusion_plot

    def st_plot_states_over_time(node_hist, step_interval=1, figsize=(14,10)):
        plt.figure(figsize=figsize)
        diffusion_plot=None
    
        for step,state in enumerate(node_hist):
            if step % step_interval==0:
                diffusion_plot=st_plot_states_helper(state,step,diffusion_plot=diffusion_plot)

    st.header("Simulation") 
    st.subheader("Initial Network State")
    st_plot_states_over_time([initial_state], step_interval=1)
    start_button=st.button("Start Simulation")
    
    if start_button:
        if option == "Simple":
            node_hist_si, counts_si = model.simple_diffusion(initial_adopters=initial_adopters, transmission_prob=0.1, steps=max_steps, seed=random_seed)
        
            st.subheader("Network state over time")
            st_plot_states_over_time(node_hist_si)
            
            st.subheader("Time series")
            st_plot_counts(counts_si, title="Simple Diffusion")
        
            final_state = node_hist_si[-1]  # last timestep
            df_unadopted = get_unadopted_boards_info(final_state, G)
            df_unadopted.index = df_unadopted.index + 1

            show_unadopted_boards(final_state, G, title="Unadopted School Boards At Final Step")
        
        elif option == "Threshold":
            node_hist_th, counts_th = model.threshold_diffusion(initial_adopters=initial_adopters, steps=max_steps, seed=random_seed)
        
            st.subheader("Network state over time")
            st_plot_states_over_time(node_hist_th)
            
            st.subheader("Time series")
            st_plot_counts(counts_th, title="Threshold Diffusion")

            final_state = node_hist_th[-1]  # last timestep
            df_unadopted = get_unadopted_boards_info(final_state, G)
            df_unadopted.index = df_unadopted.index + 1

            show_unadopted_boards(final_state, G, title="Unadopted School Boards At Final Step")
        
        
        
        elif option == "SIR":
            node_hist_sir, counts_sir = model.sir_diffusion(initial_adopters=initial_adopters, informed_prob=0.05, recovery_prob=0.02, steps=max_steps, seed=random_seed)
        
            st.subheader("Network state over time")
            st_plot_states_over_time(node_hist_sir)
            
            st.subheader("Time series")
            st_plot_counts(counts_sir, title="SIR Diffusion")
        
            final_state = node_hist_sir[-1]  # last timestep
            df_unadopted = get_unadopted_boards_info(final_state, G)
            df_unadopted.index = df_unadopted.index + 1

            show_unadopted_boards(final_state, G, title="Unadopted School Boards At Final Step")
        
        elif option == "SEIR":
            node_hist_seir, counts_seir = model.seir_diffusion(initial_adopters=initial_adopters, recovery_prob=0.02, steps=max_steps, seed=random_seed)
        
            st.subheader("Network state over time")
            st_plot_states_over_time(node_hist_seir)
            
            st.subheader("Time series")
            st_plot_counts(counts_seir, title="SEIR Diffusion")
        
            final_state = node_hist_seir[-1]  # last timestep
            df_unadopted = get_unadopted_boards_info(final_state, G)
            df_unadopted.index = df_unadopted.index + 1
        
            show_unadopted_boards(final_state, G, title="Unadopted School Boards At Final Step")
        
elif selected == "School Board Info":
    st.header("School Board Information")

    board_data = []
    for n, data in G.nodes(data=True):
        board_data.append({
            "Board ID": n,
            "Board Name": data.get("name"),
            "Region": data.get("region"),
            "Language": data.get("language"),
            "Funding": data.get("funding_orig"), 
            "Enrolment": data.get("enrolment_orig"),
            "Total Schools": data.get("node_size")
        })
    board_df = pd.DataFrame(board_data)

    regions = board_df["Region"].unique().tolist()
    languages = board_df["Language"].unique().tolist()

    selected_regions = st.multiselect("Select Region(s)", options=regions, default=regions)
    selected_languages = st.multiselect("Select Language(s)", options=languages, default=languages)

    filtered_df = board_df[
        (board_df["Region"].isin(selected_regions)) & 
        (board_df["Language"].isin(selected_languages))
    ]

    filtered_df = filtered_df.reset_index(drop=True)
    
    sort_option = st.selectbox(
        "Sort by",
        options=["Funding", "Enrolment", "Total Schools"]
    )
    sort_order = st.radio("Sort order", options=["Descending", "Ascending"])
    
    ascending = True if sort_order == "Ascending" else False
    filtered_df = filtered_df.sort_values(by=sort_option, ascending=ascending).reset_index(drop=True)


    
    filtered_df.index = filtered_df.index + 1        
    st.dataframe(filtered_df)

elif selected == "Feeling Confused?":
    st.header("Diffusion Models and Parameters Explained")
    
    st.subheader("Diffusion Models")
    st.markdown("""
    - **Simple Model (SI)**: Nodes are either Susceptible (S) or Informed (I). In each step, an informed node can transmit the information to connected susceptible nodes with a probability defined by *Transmission Probability*.
    - **Threshold Model**: A node adopts a practice if a certain proportion of its neighbors have adopted. The adoption threshold has been preset based on factors including funding, enrolment, and school board size. Nodes are Susceptible (S) or Informed (I).
    - **SIR Model**: Nodes can be Susceptible (S), Informed (I), or Recovered (R). Informed nodes may recover with a certain *Recovery Probability* and stop spreading the information.
    - **SEIR Model**: Nodes can be Susceptible (S), Exposed (E), Informed (I), or Recovered (R). Exposure introduces a delay before nodes become informed.
    """)

    st.subheader("Parameters")
    st.markdown("""
    - **Transmission Probability**: For Simple model, the chance that an informed node passes information to a susceptible neighbor in one timestep.
    - **Recovery Probability**: For SIR/SEIR models, the chance that an informed node stops spreading information in one timestep.
    - **Informed Probability**: For SIR model, the probability that a susceptible node becomes informed when interacting with an informed node.
    - **Number of Initial Adopters**: How many nodes start as informed at timestep 0.
    - **Max Steps**: The total number of simulation steps to run.
    - **Random Seed**: Ensures reproducibility of random processes.
    """)
