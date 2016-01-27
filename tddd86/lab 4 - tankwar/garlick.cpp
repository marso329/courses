/*
 * Implements the behaviour in garlick.h
 */

#include "garlick.h"
#include <iostream>
#include<fstream>

garlick::garlick() {
    name = "Ryan Garlick";
}

action garlick::doYourThing (const sensors &s) {
    analyze(s);
    update_map_size(s);
    if (powerupp_nearby(s)){
        return get_powerupps(s);
    }
    if (mode=="attack"){
        if (!(close_enough_to_enemy(s))){
            return avoid_obstacle(s, get_to_enemy(s));
        }
        if (to_close_to_enemy(s)){
            return avoid_obstacle(s,evasion(s));
        }
        else{
            return avoid_obstacle(s,attack(s));

        }
    }
    if (mode=="run_away"){
        evasive_action++;
        return avoid_obstacle(s,evasion(s));
    }
    if (mode=="mine_base"){
        return avoid_obstacle(s,mineBase(s));
    }
    mode="attack";
    return attack(s);

}
action garlick::mineBase(const sensors &s){
    if ((s.me!=s.myBase) && (case_mine==0)){
        return go_to_my_base(s);
    }
    if ((s.me==s.myBase) && (case_mine==0)){
        mine_location.c=s.myBase.c+1;
        mine_location.r=s.myBase.r;
        if (s.look[1]!=obs && s.look[1]!=edge){
        case_mine=1;
    }
    }
    switch(case_mine){
    case 1:
        if (s.me!=mine_location){
            return go_to_mine_location(s);
        }
        else{
            action move;
            move.theMove=mine;
            mine_location.r=mine_location.r+1;
            mine_location.c=mine_location.c-1;
            case_mine=2;
            return move;

        }
    case 2:
        if (s.me!=mine_location){
            return go_to_mine_location(s);
        }
        else{
            action move;
            move.theMove=mine;
            mine_location.r=mine_location.r-1;
            mine_location.c=mine_location.c-1;
            case_mine=3;
            return move;

        }
    case 3:
        if (s.me!=mine_location){
            return go_to_mine_location(s);
        }
        else{
            action move;
            move.theMove=mine;
            mine_location.r=mine_location.r-1;
            mine_location.c=mine_location.c+1;
            case_mine=4;
            return move;

        }
    case 4:
        if (s.me!=mine_location){
            return go_to_mine_location(s);
        }
        else{
            action move;
            move.theMove=mine;
            case_mine=0;
            mode="attack";
            return move;

        }
    }
mode="attack";
return attack(s);
}

void garlick:: analyze(const sensors &s){
    if (s.justHit){
        mode="run_away";
    }
    if ((rows[s.me.r])&&(columns[s.me.c])){
        been_here++;
        cout<<"been here"<< been_here<<"times" <<endl;
    }
    else{
        been_here=0;
        cout<<"not been here"<<endl;
    }
    rows[s.me.r]=true;
    columns[s.me.c]=true;
    if (evasive_action==5){
        mode="attack";
        evasive_action=0;
    }
    if (been_here>5){
        obstacles_enabled=false;
        been_here=0;
    }
}

void garlick::update_map_size(const sensors &s){
    int values[4];
    values[0]= s.myBase.c;
    values[1]=s.opp.c;
    values[2]=s.oppBase.c;
    values[3]=s.me.c;
    int max=0;
    for (int i =0;i<4;i++){
        if (values[i]>max){
            max=values[i];
        }}
    if (max>visable_size.columns){
        
        visable_size.columns=max;
    }
    values[0]= s.myBase.r;
    values[1]=s.opp.r;
    values[2]=s.oppBase.r;
    values[3]=s.me.r;
    max=0;
    for (int i =0;i<4;i++){
        if (values[i]>max){
            max=values[i];
        }}
    if (max>visable_size.rows){
        visable_size.rows=max;
    }
}
bool garlick::close_enough_to_enemy(const sensors &s){
    double temp=sqrt(pow((s.opp.c-s.me.c),2)+pow((s.opp.r-s.me.r),2));
    if (temp<15.0){
        return true;
    }
    else{
        return false;
    }
}
bool garlick::to_close_to_enemy(const sensors &s){
    double temp=sqrt(pow((s.opp.c-s.me.c),2)+pow((s.opp.r-s.me.r),2));
    if (temp<5.0){
        return true;
    }
    else{
        return false;
    }
}

string garlick::taunt(const string &otherguy) const{
    return "I am going to take your lunch money " + otherguy;
}
action garlick::evasion(const sensors &s) {
    return away_from_enemy(s);
}
action garlick::attack(const sensors &s){
    action move;
    move.theMove=fire;
    move.aim=s.opp;
    return move;
}

action garlick::go_to_my_base(const sensors &s) {
    action move;
    list<location> myPath;
    myPath = Board::getLine(s.me, s.myBase);
    list<location>::iterator i;

    if (s.me != s.myBase) {
        i = myPath.begin();
        if ((i->c == s.me.c-1) && (i->r == s.me.r-1)) move.theMove = moveNW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r-1)) move.theMove = moveNE;
        if ((i->c == s.me.c) && (i->r == s.me.r-1)) move.theMove = moveN;
        if ((i->c == s.me.c-1) && (i->r == s.me.r+1)) move.theMove = moveSW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r+1)) move.theMove = moveSE;
        if ((i->c == s.me.c) && (i->r == s.me.r+1)) move.theMove = moveS;
        if ((i->c == s.me.c-1) && (i->r == s.me.r)) move.theMove = moveW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r)) move.theMove = moveE;
    }
    else {
        move.theMove = sit;
    }
    return move;
}
action garlick::go_to_mine_location(const sensors &s) {
    action move;
    list<location> myPath;
    myPath = Board::getLine(s.me, mine_location);
    list<location>::iterator i;

    if (s.me != mine_location) {
        i = myPath.begin();
        if ((i->c == s.me.c-1) && (i->r == s.me.r-1)) move.theMove = moveNW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r-1)) move.theMove = moveNE;
        if ((i->c == s.me.c) && (i->r == s.me.r-1)) move.theMove = moveN;
        if ((i->c == s.me.c-1) && (i->r == s.me.r+1)) move.theMove = moveSW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r+1)) move.theMove = moveSE;
        if ((i->c == s.me.c) && (i->r == s.me.r+1)) move.theMove = moveS;
        if ((i->c == s.me.c-1) && (i->r == s.me.r)) move.theMove = moveW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r)) move.theMove = moveE;
    }
    else {
        move.theMove = sit;
    }
    return move;
}
action garlick::get_to_enemy(const sensors &s) {
    action move;
    list<location> myPath;
    myPath = Board::getLine(s.me, s.opp);
    list<location>::iterator i;

    if (s.me != s.opp) {
        i = myPath.begin();
        if ((i->c == s.me.c-1) && (i->r == s.me.r-1)) move.theMove = moveNW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r-1)) move.theMove = moveNE;
        if ((i->c == s.me.c) && (i->r == s.me.r-1)) move.theMove = moveN;
        if ((i->c == s.me.c-1) && (i->r == s.me.r+1)) move.theMove = moveSW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r+1)) move.theMove = moveSE;
        if ((i->c == s.me.c) && (i->r == s.me.r+1)) move.theMove = moveS;
        if ((i->c == s.me.c-1) && (i->r == s.me.r)) move.theMove = moveW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r)) move.theMove = moveE;
    }
    else {
        move.theMove = sit;
    }
    return move;
}

int garlick::ChangeInRows(action temp_action){
    int temp_int=-1;
    for (int i=0;i<9;i++){
        if ((temp_action.theMove==get_direction(i))){
            temp_int=i;
        }
    }
    if ((temp_int>-1)&&(temp_int<3)){
        return +1;
    }
    else if ((temp_int>5)&&(temp_int<9)){
        return -1;

    }
    else{
        return 0;
    }
}
int garlick::ChangeInColumns(action temp_action){
    int temp_int=-1;
    for (int i=0;i<9;i++){
        if ((temp_action.theMove==get_direction(i))){
            temp_int=i;
        }
    }
    if ((temp_int==2)||(temp_int==5)||(temp_int==8)){
        return +1;
    }
    else if ((temp_int==0)||(temp_int==3)||(temp_int==6)){
        return -1;

    }
    else{
        return 0;
    }
}
action garlick::avoid_obstacle(const sensors &s,action temp_action){
    if (iterations_of_obstacles_disabled>5){
        iterations_of_obstacles_disabled=0;
        obstacles_enabled=true;
    }
    if (!(obstacles_enabled)){
        iterations_of_obstacles_disabled++;
        return temp_action;
    }

    bool temp=true;
    int temp_int;
    for (int i=0;i<9;i++){
        if ((temp_action.theMove==get_direction(i))&&(s.look[i]==obs)){
            temp=false;
            temp_int=i;
        }
    }
    if (temp){
        avoided_obstacle=0;
        return temp_action;
    }
    if (avoided_obstacle>10){
        iterations_of_obstacles_disabled=0;
        obstacles_enabled=false;
        avoided_obstacle=0;
        return temp_action;

    }
    else{
        avoided_obstacle++;
        cout<<avoided_obstacle<<endl;
        if (temp_int==8){
            temp_action.theMove=get_direction(0);
            return avoid_obstacle(s,temp_action);
        }
        if (temp_int==3){
            temp_action.theMove=get_direction(5);
            return avoid_obstacle(s,temp_action);
        }
        else{
            temp_action.theMove=get_direction(temp_int+1);
            return avoid_obstacle(s,temp_action);
        }
    }
}

action garlick::get_powerupps(const sensors &s){
    action move;
    move.theMove=get_direction(location_of_powerupp(s));
    return move;
}
moves garlick::get_direction(int temp){
    moves direction[9];
    direction[0]=moveNW;
    direction[1]=moveN;
    direction[2]=moveNE;
    direction[3]=moveW;
    direction[4]=sit;
    direction[5]=moveE;
    direction[6]=moveSW;
    direction[7]=moveS;
    direction[8]=moveSE;
    return direction[temp];
}
int garlick::location_of_powerupp(const sensors &s){
    int temp=0;
    for(int i=0;i<9;i++){
        if((s.look[i]==pu_mines)||(s.look[i]==pu_ammo)||(s.look[i]==pu_points)){
            temp=i;
        }
    }
    return temp;
}

bool garlick::powerupp_nearby(const sensors &s){
    bool temp=false;
    for(int i=0;i<9;i++){
        if((s.look[i]==pu_mines)||(s.look[i]==pu_ammo)||(s.look[i]==pu_points)){
            temp=true;
        }
    }
    return temp;
}


action garlick::away_from_enemy(const sensors &s){
    location away;
    action move;
    away.r=visable_size.rows-s.opp.r;
    away.c=visable_size.columns- s.opp.c;
    cout<<away.r<<endl;
    cout<<away.c<<endl;
    cout<<"hej"<<endl;
    list<location> myPath;
    myPath = Board::getLine(s.me, away);
    list<location>::iterator i;

    if (s.me != away) {
        i = myPath.begin();
        if ((i->c == s.me.c-1) && (i->r == s.me.r-1)) move.theMove = moveNW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r-1)) move.theMove = moveNE;
        if ((i->c == s.me.c) && (i->r == s.me.r-1)) move.theMove = moveN;
        if ((i->c == s.me.c-1) && (i->r == s.me.r+1)) move.theMove = moveSW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r+1)) move.theMove = moveSE;
        if ((i->c == s.me.c) && (i->r == s.me.r+1)) move.theMove = moveS;
        if ((i->c == s.me.c-1) && (i->r == s.me.r)) move.theMove = moveW;
        if ((i->c == s.me.c+1) && (i->r == s.me.r)) move.theMove = moveE;
    }
    else {
        move.theMove = sit;
    }
    return move;
}
