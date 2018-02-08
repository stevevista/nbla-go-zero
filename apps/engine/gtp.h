#pragma once

#include "defs.h"

#include <functional>
#include <thread>
#include <queue>
#include <mutex>  
#include <condition_variable> 
#include <string>
#include <vector>

template<typename T>  
class safe_queue  
{  
  private:  
     mutable std::mutex mut;   
     std::queue<T> data_queue;  
     std::condition_variable data_cond;  
  public:  
    safe_queue(){}  
    safe_queue(safe_queue const& other)  
     {  
         std::lock_guard<std::mutex> lk(other.mut);  
         data_queue=other.data_queue;  
     }  
     void push(T new_value)//入队操作  
     {  
         std::lock_guard<std::mutex> lk(mut);  
         data_queue.push(new_value);  
         data_cond.notify_one();  
     }  
     void wait_and_pop(T& value) //直到有元素可以删除为止  
     {  
         std::unique_lock<std::mutex> lk(mut);  
         data_cond.wait(lk,[this]{return !data_queue.empty();});  
         value=data_queue.front();  
         data_queue.pop();  
     }  
     std::shared_ptr<T> wait_and_pop()  
     {  
         std::unique_lock<std::mutex> lk(mut);  
         data_cond.wait(lk,[this]{return !data_queue.empty();});  
         std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));  
         data_queue.pop();  
         return res;  
     }  
     bool try_pop(T& value) //不管有没有队首元素直接返回  
     {  
         std::lock_guard<std::mutex> lk(mut);  
         if(data_queue.empty())  
             return false;  
         value=data_queue.front();  
         data_queue.pop();  
         return true;  
     }  
     std::shared_ptr<T> try_pop()  
     {  
         std::lock_guard<std::mutex> lk(mut);  
         if(data_queue.empty())  
             return std::shared_ptr<T>();  
         std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));  
         data_queue.pop();  
         return res;  
     }  
     bool empty() const  
     {  
         std::lock_guard<std::mutex> lk(mut);  
         return data_queue.empty();  
     }  
}; 



class IGtpAgent {
public:
    virtual ~IGtpAgent() {}
    virtual std::string name() = 0;
    virtual void clear_board() = 0;
    virtual void komi(float) = 0;
    virtual void time_left(int player, double t) = 0;
    virtual void play(int player, int move) = 0;
    virtual void pass(int player) = 0;
    virtual void resign(int player) = 0;
    virtual int genmove(int player, bool commit) = 0;
    virtual int genmove() = 0;
    virtual void ponder_on_idle() = 0;
    virtual void ponder_enable() = 0;
    virtual void stop_ponder() = 0;
    virtual float final_score() = 0;
    virtual void game_over() = 0;
    virtual void quit() = 0;
    virtual void set_timecontrol(int maintime, int byotime, int byostones, int byoperiods) = 0;
    virtual void heatmap(int rotation) const = 0;
    virtual int get_color(int idx) const = 0;
    virtual bool dump_sgf(const std::string& path) const = 0;
};

void play_matchs(const std::string& sgffile, IGtpAgent* player1, IGtpAgent* player2, std::function<void(int, int[])> callback);

class Gtp {

    safe_queue<std::string> r_queue;
    safe_queue<std::pair<std::string, std::string>> s_queue;

    std::thread th_;

    int CallGTP();
    IGtpAgent* agent_;
    bool use_pondering_;

public:
    static constexpr int MAXPOS = 361;

    NENG_API Gtp(IGtpAgent*);

    void send_command(const std::string& cmd) {
        r_queue.push(cmd);
    }

    void unsolicite(std::string& cmd, std::string& rsp) {
        std::pair<std::string, std::string> out;
		s_queue.wait_and_pop(out);
        cmd = out.first;
        rsp = out.second;
    }

    NENG_API void enable_ponder(bool);
    NENG_API void run();
    NENG_API void stop_thinking();

    void join() {
        th_.join();
    }
};


NENG_API std::string xy2movetext(int x, int y);
NENG_API std::pair<int, int> movetext2xy(const std::string& text);
