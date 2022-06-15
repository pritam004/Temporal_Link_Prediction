import networkx as nx
import dgl

G=nx.Graph()
G=nx.from_pandas_edgelist(train_edge[['cc_id','merchant_id']],'cc_id','merchant_id')
nx.write_gpickle(G,dir+'/train_graph.pkl')
G_dgl=dgl.from_networkx(G)

print(G_dgl.nodes())
print(G_dgl.edges())

