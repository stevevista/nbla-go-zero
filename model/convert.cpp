#include "convert.hpp"
#include "sgf.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <cassert>
#include "utils.hpp"


using namespace napp;


static constexpr int64_t MAX_DATA_SIZE_KB = 12 * 1024 * 1024;
static constexpr float score_thres = 2.5;
static constexpr int move_count_thres = 30;


void proceed_moves(std::ostream& os, int result, std::vector<int> tree_moves,
    const int64_t total_games,
    const int64_t prcessed, 
    int64_t& total_moves,
    int64_t& total_size) {

    if (tree_moves.size() >=2 && 
        tree_moves[tree_moves.size()-2] == 361 && 
        tree_moves[tree_moves.size()-1] == 361)
        tree_moves.erase(tree_moves.end()-1);

    std::vector<short> seqs;
    if (!GoBoard::generate_move_seqs(tree_moves, seqs)) {
        //throw std::runtime_error("bad move sequence");
        return;
    }

    // format: magic(1) result(1) move_count(2) move_seqs
    
    if (tree_moves.size() < move_count_thres)
        return;

    int move_count = tree_moves.size();
    
    // magic
    os.write("g", 1);

    // result
    char cval = (char)result;
    os.write(&cval, 1);

    // count
    short sval = (short)move_count;
    os.write((char*)&sval, 2);

    // seqs
    os.write((char*)&seqs[0], seqs.size()*sizeof(short));

    total_moves += move_count;
    total_size += 4 + seqs.size()*sizeof(short);

    int percent = (prcessed*100)/total_games;
    float gb = float(total_size)/(1024*1024*1024);
    std::cout << percent << "%, " << "commiting " << move_count << ", total moves " << total_moves << ", size " << gb << "GB" << std::endl;
}

void parse_sgf_file(std::ostream& os, const std::string& sgf_name, 
    int64_t& total_games,
    int64_t& prcessed,
    int64_t& total_moves,
    int64_t& total_size,
    int random_drop) {

    auto games = SGFParser::chop_all(sgf_name);
    if (games.size() > 1)
        total_games += (games.size() - 1);

    for (const auto& s : games) {
            
        prcessed++;
            
        int boardsize;
        float komi;
        float result;
        std::vector<int> moves;

        std::istringstream pstream(s);
        SGFParser::parse(pstream, boardsize, komi, result, moves);

        if (boardsize != 19)
            continue;
            
        if (moves.size() < 0)
            continue;

        if (result == 0)
            continue;

        auto fixed_result = result - (7.5 - komi);
        if (std::abs(fixed_result) < score_thres)
            continue;

        int r = fixed_result > 0 ? 1 : -1;

        proceed_moves(os, r, moves,
            total_games,
            prcessed, 
            total_moves,
            total_size);

        if (total_size/1024 > MAX_DATA_SIZE_KB)
            break;
    }
}


void parse_index(const std::string& path, std::map<uint32_t, int>& scores) {
    
    std::ifstream ifs(path);
    
    
    auto read_nation = [&]() {
            std::string tmp;
            do {
                ifs >> tmp;
                if (ifs.eof()) break;
                if (tmp.size()==1 && tmp[0]>='0' && tmp[0]<='5')
                    break;
            } while(true);
    };
    
    while (!ifs.eof()) {
            
            uint32_t id;
            std::string date;
            std::string time;
            std::string white;
            std::string white_eng;
            std::string black;
            std::string black_eng;
            std::string result;
            int round;
            std::string byoyomi; 
            std::string minutes;
    
            ifs >> id;
            ifs >> date;
    
            //std::cout << date << std::endl;
    
            if (ifs.eof()) break;
    
            ifs >> time;
            ifs >> white;
            ifs >> white_eng;
            read_nation();
            ifs >> black;
            ifs >> black_eng;
            read_nation();
            ifs >> result;
            ifs >> round;
            ifs >> byoyomi;
            ifs >> minutes;
    
            int score = 0;
            if (result.find("B+") == 0)
                score = 1;
            else if (result.find("W+") == 0)
                score = -1;
            else 
                continue;
    
            auto reason = result.substr(2);
            
            auto v = std::atof(reason.c_str());
            if (v == 0) {
                if (reason != "Resign")
                    continue;
            } else {
                if (std::abs(v*score-1) < score_thres)
                    continue; // since default komi == 6.5 
            }
    
            scores.insert({id, score});
        }
        std::cout << "done" << std::endl;
}

        
int parse_kifu_moves(const std::string& path, const std::map<uint32_t, int>& scores, std::ostream& os, 
    int64_t total_games,
    int64_t& prcessed, 
    int64_t& total_moves, 
    int64_t& total_size, int random_drop) {

    std::ifstream ifs(path);
    std::cout << path << std::endl;

    int total = 0;

    while (!ifs.eof()) {
        uint32_t id;
        std::string seqs;

        ifs >> id;
        if (ifs.eof()) break;

        ifs >> seqs;

        auto it = scores.find(id);
        if (it == scores.end())
            continue;

        prcessed++;

        int result = it->second;

        auto tree_moves = seq_to_moves(seqs);

        proceed_moves(os, result, tree_moves,
                    total_games,
                    prcessed, 
                    total_moves,
                    total_size);

        if (total_size/1024 > MAX_DATA_SIZE_KB)
            break;
    }

    return total;
}


void GameArchive::generate(const std::string& path, const std::string& out_path, int random_drop) {

    std::vector<std::string> file_list;
    napp::enumerateDirectory(file_list, path, true, {});

    std::map<uint32_t, int> scores;
    int sgf_counts = 0;
    
    for (auto f : file_list) {
    
        auto ext = f.substr(f.rfind(".")+1);
        if (ext == "index") 
            parse_index(f, scores);
    
        if (ext == "sgf") 
            sgf_counts++;
    }

    std::ofstream ofs(out_path, std::ofstream::binary);
    ofs.write("G", 1);

    int64_t prcessed = 0;
    int64_t total_moves = 0;
    int64_t total_games = scores.size() + sgf_counts;
    int64_t total_size = 1;
    
    for (auto f : file_list) {
        
        auto ext = f.substr(f.rfind(".")+1);
        if (ext == "index")
            continue;
        
        if (ext == "txt" || ext == "sgf") {
            std::cout << f << std::endl;
            parse_sgf_file(ofs, f, total_games, prcessed, total_moves, total_size, random_drop); 
        } else {
            parse_kifu_moves(f, scores, ofs, total_games, prcessed, total_moves, total_size, random_drop);
        }
        
        if (total_size/1024 > MAX_DATA_SIZE_KB)
            break;
    }
}

GameArchive::GameArchive()
{
    data_index = 0;
}

int GameArchive::load(const std::string& path, bool append) {

    if (!append) {
        games_.clear();
        entries_.clear();
    }

    std::ifstream ifs(path, std::ifstream::binary);
    if (!ifs)
        return 0;

    auto read_char = [&]() {
        char v;
        if (ifs.read(&v, 1).gcount() != 1)
            throw std::runtime_error("uncomplate move data (read_char)");
        return v;
    };

    auto read_short = [&]() {
        short v;
        if (ifs.read((char*)&v, 2).gcount() != 2)
            throw std::runtime_error("uncomplate move data (read_short)");
        return v;
    };

    int read_moves = 0;

    auto type = read_char();
    if (type != 'G' && type != 'P') throw std::runtime_error("bad train data signature");

    follow_distribution_ = false;
    if (type == 'P')
        follow_distribution_ = true; // every move followed by 362 move probs
    
    while (true) {
        char c;
        if (ifs.read(&c, 1).gcount() == 0)
            break;
            
        if (c != 'g') throw std::runtime_error("bad move signature");

        int result = (int) read_char();
        auto steps = read_short();

        if (result !=0 && result !=1 && result != -1)
            throw std::runtime_error("bad game result");

        game_t game;
        game.result = result;
        int game_index = games_.size();

        for (auto step = 0; step < steps; step++) {
            auto move = read_short();
            game.seqs.emplace_back(move);
            entries_.push_back({game_index, step});

            if (move < 0) {
                auto count = read_short();
                game.seqs.emplace_back(count);
                for (int i=0; i<count; i++)
                    game.seqs.emplace_back(read_short());
            }  

            if (follow_distribution_) {
                std::array<float, 362> dist;
                if (ifs.read((char*)&dist[0], 362*sizeof(float)).gcount() != 362*sizeof(float))
                    throw std::runtime_error("uncomplate move data (read disttribution)");
                game.dists.emplace_back(dist);
            }
        }

        games_.push_back(std::move(game));
        read_moves += steps;
    }
    
    ifs.close();

    shuffle();
    return read_moves;
}


void GameArchive::shuffle() {
    std::random_shuffle(entries_.begin(), entries_.end());
}


vector<MoveData> GameArchive::next_batch(int count, bool& rewinded) {
    
    vector<MoveData> out;
    rewinded = false;

    if (total_moves() == 0)
        throw std::runtime_error("empty move archive");

    while (out.size() < count) {

        if (data_index >= total_moves()) {
            data_index = 0;
            rewinded = true;
        }
    
        MoveData data;
        extract_move(data_index++, data);
        out.emplace_back(data);
    }
    
    return std::move(out);
}

    
void GameArchive::extract_move(const int index, MoveData& out) {

    if (index < 0 || index >= entries_.size())
        throw std::runtime_error("extract_move index error");

    auto ind = entries_[index];

    const int steps = ind.second;
    const auto& game = games_[ind.first];
    const auto& game_seq = game.seqs;
    int result = game.result;

    const bool to_move_is_black = (steps%2 == 0);
    
    
    out.result = to_move_is_black ? result : -result;

    // replay board
    BoardPlane blacks;
    BoardPlane whites;

    out.input.resize(zero::input_channels);

    blacks = false;
    whites = false;
    for (auto& l : out.input)
        l = false;

    int i = 0;
    bool black_player = true;
    for (int step = 0; step < steps; step++) {
        
        auto sign_pos = game_seq[i++];
        auto pos = std::abs(sign_pos) - 1;

        if (pos > 361)
            throw std::runtime_error("bad move pos " + std::to_string(pos));

        auto& my_board = black_player ? blacks : whites;
        auto& op_board = black_player ? whites : blacks;

        if (pos == 361) {
            // pass
        } else {
            my_board[pos] = true;
            if (sign_pos < 0) {
                int rm = game_seq[i++];

                for (int n=0; n<rm; n++) {
                    auto rmpos = game_seq[i++];
                    
                    if (rmpos < 0 || rmpos >= 361)
                        throw std::runtime_error("bad rm move pos " + std::to_string(rmpos));
                    op_board[rmpos] = false;
                }
            }
        }

        // copy history
		int h = steps - step - 1;
		if (h < zero::input_history ) {
            // collect white, black occupation planes
            out.input[h] = to_move_is_black ? blacks : whites;
            out.input[zero::input_history + h] = to_move_is_black ? whites : blacks;
		}

        black_player = !black_player;
    }


    out.probs.resize(362, 0);

    if (follow_distribution_) {
        std::copy(game.dists[steps].begin(), game.dists[steps].end(), out.probs.begin());
    } else {
        const int cur_move = std::abs(game_seq[i]) - 1;
        out.probs[cur_move] = 1;
    }

    if (to_move_is_black)
        out.input[zero::input_channels - 2].set();
    else
        out.input[zero::input_channels - 1].set();
}
