import streamlit as st
import data_operations as data_ops
from clustering import PositionalCluster
from sklearn.preprocessing import Normalizer

if 'player_data' not in st.session_state:
    st.session_state['player_data'] = data_ops.get_data()

subset = st.text_input("subset", "all")
n_clust = st.text_input("n_clust", 5)

if subset != "all":
    subset = [x.strip() for x in subset.split(",")]

try:
    if type(n_clust) == float:
        raise ValueError("Entered float for n_clust")
    n_clust = int(n_clust)
except ValueError:
    print("n_clust must be an integer, defaulting to 5")
    n_clust = 5

qb_clust = PositionalCluster(st.session_state['player_data'], 'QB', subset=subset, n_clust=n_clust)
norm = Normalizer('l1')
qb_clust.fit_clusters(normalizer=norm)
fig = qb_clust.plot_groups()
st.pyplot(fig)
