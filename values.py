device = "cpu"

source_emb_file=""
target_emb_file=""

compression=0

#----------------------emb-files-------------------------
# ---1.gcn---------------------------------------------------
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/gcn_emb.csv"
#---2.graph_sage--------------------------------------------
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/graph_sage_emb.csv"
#---3.egcn-o+f----------------------------------------------------
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/log/log_sbm50_link_pred_egcn_o_20220612015740_r0.log_train_nodeembs.csv.gz"
# compression=1
#--4.egcn-h---------------------------------------------------
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/log/log_sbm50_link_pred_egcn_h_20220611020756_r0.log_train_nodeembs.csv.gz"
# compression=1
#--5.graph_sage+F---------------------------'
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/graph_sage_emb_feat_new.csv"
#----6.GAT-------------------------------------------------------'
# emb_file="/content/drive/MyDrive/NXM/EvolveGCN/gat_emb.csv"

#--------------7.TGAT--------------------------------------
# emb_file="/content/drive/MyDrive/NXM/TGAT/tgat_emb.csv"

#-------------------------8.Graph sage 2 epochs--------------
emb_file="/content/drive/MyDrive/NXM/EvolveGCN/graph_sage_emb_feat_new_2.csv"

sep= 997
# compression=0

test_file='/content/drive/MyDrive/NXM/EvolveGCN/data/kag_trans_test.csv'
train_file='/content/drive/MyDrive/NXM/EvolveGCN/data/kag_trans_train.csv'

source_col="cc_id"
target_col="merchant_id"
time_col="discrete_time"

bipartite=0

test_ratio=0.3

##node id range of source and target
source_node_range=(0,996)
target_node_range=(997,1689)