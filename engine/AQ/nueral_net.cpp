#include "nueral_net.h"
#include <gomodel/model.h>

using namespace tensorflow;
using std::string;
using std::cerr;
using std::endl;

int cfg_sym_idx = 0;

#ifdef USE_52FEATURE
	constexpr int feature_cnt = 52;
#else
	constexpr int feature_cnt = 49;
#endif

using namespace lightmodel;

alphago::net_type policy_net;
alphago::vnet_type value_net;


void ReadPolicyProto(const std::string& sl_path) {
	if (!load_ago_policy(policy_net, sl_path))
		throw std::runtime_error(sl_path + " fail");
}

void ReadValueProto(const std::string& vl_path) {
	if (!load_ago_value(value_net, vl_path))
		throw std::runtime_error(vl_path + " fail");
}

/**
 *  ����m����\������l�b�g���[�N
 *  Calculate probability distribution with the Policy Network.
 */
void PolicyNet(Session* sess, std::vector<FeedTensor>& ft_list,
		std::vector<std::array<double,EBVCNT>>& prob_list,
		double temp, int sym_idx)
{

	prob_list.clear();
	int ft_cnt = (int)ft_list.size();
	resizable_tensor x;
	x.set_size(ft_cnt, feature_cnt, BSIZE, BSIZE);
	auto x_eigen = x.host_write_only();
	layer<0>(policy_net).layer_details().set_temprature(temp);

	std::vector<int> sym_idxs;
	if(sym_idx > 7){
		for(int i=0;i<ft_cnt;++i){
			sym_idxs.push_back(mt_int8(mt_32));
		}
	}

	for(int i=0;i<ft_cnt;++i){
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){

				if(sym_idx == 0) x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[j][k];
				else if(sym_idx > 7) x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idxs[i]][j][0]][k];
				else x_eigen[i*feature_cnt*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx][j][0]][k];

			}
		}
	}

	auto& outputs = policy_net.forward(x);

	auto policy = outputs.host();

	for(int i=0;i<ft_cnt;++i){
		std::array<double, EBVCNT> prob;
		prob.fill(0.0);

		for(int j=0;j<BVCNT;++j){
			int v = rtoe[j];

			if(sym_idx == 0) prob[v] = (double)policy[i*BVCNT+j];
			else if(sym_idx > 7) prob[v] = (double)policy[i*BVCNT+sv.rv[sym_idxs[i]][j][1]];
			else prob[v] = (double)policy[i*BVCNT+sv.rv[sym_idx][j][1]];

			// 3����蒆�����̃V�`���E�𓦂����̊m�������
			// Reduce probability of moves escaping from Ladder.
			if(ft_list[i].feature[j][LADDERESC] != 0 && DistEdge(v) > 2) prob[v] *= 0.001;
		}
		prob_list.push_back(prob);
	}

}


/**
 *  �ǖʕ]���l��\������l�b�g���[�N
 *  Calculate value of the board with the Value Network.
 */
void ValueNet(Session* sess, std::vector<FeedTensor>& ft_list,
		std::vector<float>& eval_list, int sym_idx)
{

	eval_list.clear();
	int ft_cnt = (int)ft_list.size();
	const int nk = feature_cnt+1;
	resizable_tensor x;
	x.set_size(ft_cnt, nk, BSIZE, BSIZE);
	auto x_eigen = x.host_write_only();
	
	int sym_idx_rand = mt_int8(mt_32);

	for(int i=0;i<ft_cnt;++i){
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){
				if(sym_idx == 0) x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[j][k];
				else if(sym_idx > 7) x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx_rand][j][0]][k];
				else x_eigen[i*nk*BVCNT + k*BVCNT + j] = ft_list[i].feature[sv.rv[sym_idx][j][0]][k];
			}
			x_eigen[i*nk*BVCNT + feature_cnt*BVCNT + j] = (float)ft_list[i].color;
		}
	}

	auto& outputs = value_net.forward(x);
	auto value = outputs.host();
	for(int i=0;i<ft_cnt;++i){
		eval_list.push_back((float)value[i]);
	}

}
