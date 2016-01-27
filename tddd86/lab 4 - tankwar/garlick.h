/*
 * Example player class.
 */

#include "Board.h"
#include "Tank.h"
#include <map>
struct map_size{
    int rows=0;
    int columns=0;
};


class garlick: public Tank {

public:
    garlick();
    ~garlick() = default;

    /*
     * this simple player class calls the base class (tank) method to go to the opponents base.
     * It will move to the opponents base and then sit there, gaining a point per turn for
     * pillaging. It does not move thereafter, and becomes a sitting duck.
     */
    action doYourThing(const sensors&) override;
    string taunt(const string&) const override;
    action evasion(const sensors&);

private:
    // any data that I want to retain for my tank - probably want to store some tank state information
    void update_map_size(const sensors&);
    action away_from_enemy(const sensors&);
    bool powerupp_nearby(const sensors&);
    moves get_direction(int temp);
    action get_powerupps(const sensors &s);
    int location_of_powerupp(const sensors &s);
    action avoid_obstacle(const sensors &s,action temp_action);
    action go_to_my_base(const sensors &s);
    action go_to_mine_location(const sensors &s);
    action mineBase(const sensors &s);
    action get_to_enemy(const sensors &s);
    bool close_enough_to_enemy(const sensors &s);
    bool to_close_to_enemy(const sensors &s);
    action attack(const sensors &s);
    int ChangeInColumns(action temp_action);
    int ChangeInRows(action temp_action);
    int tempint=0;
    string mode="mine_base";
    map_size visable_size;
    int avoided_obstacle=0;
    void analyze(const sensors &s);
    int evasive_action=0;
    int case_mine=0;
    location mine_location;
    map<int,bool> rows;
    map<int,bool> columns;
    int been_here=0;
    bool obstacles_enabled=true;
    int iterations_of_obstacles_disabled=0;

};
