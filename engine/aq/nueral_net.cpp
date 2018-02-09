#include "nueral_net.h"
#include "../nn.h"

using std::string;
using std::cerr;
using std::endl;


int cfg_sym_idx = 0;

constexpr int feature_cnt = 18;

FeedTensor::FeedTensor(){
	Clear();
}

void FeedTensor::Clear(){

	for(auto& i:feature) {
		std::fill(i.begin(), i.end(), 1);
	}
	color = 0;
}


void FeedTensor::Set(Board& b, int nv) {

	Clear();		
	
	color = b.my;
	
	int to_move = b.my + 2;

	for (int h = 0; h < 8; h++) {

        // collect white, black occupation planes
        for (int y=0; y<BSIZE; y++) {
            for(int x = 0; x < BSIZE; x++) {

                int pos = xytor[x][y];
                auto color = b.board_history[h][xytoe[x+1][y+1]];
                
                if (color == 0) {
                    feature[h][pos] = 0; feature[8 + h][pos] = 0;
                } else if (color == to_move) {
                    feature[h][pos] = 1; feature[8 + h][pos] = 0;
                } else {
                    feature[h][pos] = 0; feature[8 + h][pos] = 1;
                }
            }
        }
    }

	if (to_move == 3)
    	std::fill(feature[16].begin(), feature[16].end(), 1);
	else
    	std::fill(feature[17].begin(), feature[17].end(), 1);
}


/**
 *  Calculate probability distribution with the Policy Network.
 */
void PolicyNet(
		tensorflow::Session* sess,
		std::vector<FeedTensor>& ft_list,
		std::vector<std::array<double,EBVCNT>>& prob_list,
		double temp,
		int sym_idx)
{
	prob_list.clear();
	int ft_cnt = (int)ft_list.size();

	//std::cout << "predict " << ft_cnt << std::endl;

	std::vector<FeedTensor::Feature> x_eigen;

	std::vector<int> sym_idxs;
	if(sym_idx > 7){
		for(int i=0;i<ft_cnt;++i){
			sym_idxs.push_back(mt_int8(mt_32));
		}
	}

	for(int i=0;i<ft_cnt;++i){
		FeedTensor::Feature feature;
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){

				if(sym_idx == 0) feature[k][j] = ft_list[i].feature[k][j];
				else if(sym_idx > 7) feature[k][j] = ft_list[i].feature[k][sv.rv[sym_idxs[i]][j][0]];
				else feature[k][j] = ft_list[i].feature[k][sv.rv[sym_idx][j][0]];

			}
		}
		x_eigen.push_back(feature);
	}

	auto outputs = zero_net->predict_policy(x_eigen, temp);

	for(int i=0;i<ft_cnt;++i){
		std::array<double, EBVCNT> prob;
		prob.fill(0.0);

		for(int j=0;j<BVCNT;++j) {
			int v = rtoe[j];

			if(sym_idx == 0) prob[v] = (double)outputs[i][j];
			else if(sym_idx > 7) prob[v] = (double)outputs[i][sv.rv[sym_idxs[i]][j][1]];
			else prob[v] = (double)outputs[i][sv.rv[sym_idx][j][1]];

			// 3線より中央側のシチョウを逃げる手の確率を下げる
			// Reduce probability of moves escaping from Ladder.
			// if(ft_list[i].feature[j][LADDERESC] != 0 && DistEdge(v) > 2) prob[v] *= 0.001;
		}
		//std::cout << max_index(outputs[i]) << std::endl;

		prob_list.push_back(prob);
	}
}


/**
 *  Calculate value of the board with the Value Network.
 */
void ValueNet(
		tensorflow::Session* sess,
		std::vector<FeedTensor>& ft_list,
		std::vector<float>& eval_list,
		int sym_idx)
{

	eval_list.clear();
	int ft_cnt = (int)ft_list.size();

	std::vector<FeedTensor::Feature> x_eigen;

	int sym_idx_rand = mt_int8(mt_32);

	for(int i=0;i<ft_cnt;++i){
		FeedTensor::Feature feature;
		for(int j=0;j<BVCNT;++j){
			for(int k=0;k<feature_cnt;++k){
				if(sym_idx == 0) feature[k][j] = ft_list[i].feature[k][j];
				else if(sym_idx > 7) feature[k][j] = ft_list[i].feature[k][sv.rv[sym_idx_rand][j][0]];
				else feature[k][j] = ft_list[i].feature[k][sv.rv[sym_idx][j][0]];
			}
		}
		x_eigen.push_back(feature);
	}


	auto outputs = zero_net->predict_value(x_eigen);

	for(int i=0;i<ft_cnt;++i){
		eval_list.push_back((float)outputs[i]);
	}
}
