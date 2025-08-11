import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Add nodes
G.add_node("Data Ingestion")
G.add_node("Preprocessing")
G.add_node("Feature Extraction")
G.add_node("Top-level Binary Classifier (is_true)")
G.add_node("True Branch Subcategories")
G.add_node("False Branch Subcategories")
G.add_node("Misconception Classifier")
G.add_node("Probability Combination")
G.add_node("Top-3 Pairs Selection")
G.add_node("Submission Output")

# Add edges
G.add_edge("Data Ingestion", "Preprocessing")
G.add_edge("Preprocessing", "Feature Extraction")
G.add_edge("Feature Extraction", "Top-level Binary Classifier (is_true)")
G.add_edge("Feature Extraction", "Misconception Classifier")
G.add_edge("Top-level Binary Classifier (is_true)", "True Branch Subcategories")
G.add_edge("Top-level Binary Classifier (is_true)", "False Branch Subcategories")
G.add_edge("True Branch Subcategories", "Probability Combination")
G.add_edge("False Branch Subcategories", "Probability Combination")
G.add_edge("Misconception Classifier", "Probability Combination")
G.add_edge("Probability Combination", "Top-3 Pairs Selection")
G.add_edge("Top-3 Pairs Selection", "Submission Output")

# Positions for hierarchical layout
pos = {
    "Data Ingestion": (0, 10),
    "Preprocessing": (0, 9),
    "Feature Extraction": (0, 8),
    "Top-level Binary Classifier (is_true)": (-1, 7),
    "True Branch Subcategories": (-2, 6),
    "False Branch Subcategories": (0, 6),
    "Misconception Classifier": (2, 7),
    "Probability Combination": (0, 5),
    "Top-3 Pairs Selection": (0, 4),
    "Submission Output": (0, 3)
}

# Draw the graph
plt.figure(figsize=(14, 12))
nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True, arrowstyle="->", arrowsize=20)
plt.title("System Flowchart: Data Flow from Ingestion to Final Predictions with Hierarchical Branching")
plt.savefig("img/flowchart.png")
plt.show()