
/*
 * TDDD86 Pattern Recognition
 * This program computes and plots all line segments involving 4 points
 * in a file using Qt.
 */

#include <QApplication>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include "Point.h"

// constants
static const int SCENE_WIDTH = 512;
static const int SCENE_HEIGHT = 512;

void render_points(QGraphicsScene* scene, const vector<Point>& points) {
    for(const auto& point : points) {
        point.draw(scene);
    }
}

void render_line(QGraphicsScene* scene, const Point& p1, const Point& p2) {
    p1.lineTo(scene, p2);
}

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    // open file
    string filename = "input100.txt";
    ifstream input;
    input.open(filename);

    // the vector of points
    vector<Point> points;

    // read points from file
    int N;
    int x;
    int y;

    input >> N;

    for (int i = 0; i < N; ++i) {
        input >> x >> y;
        points.push_back(Point(x, y));
    }
    input.close();

    // setup graphical window
    QGraphicsView *view = new QGraphicsView();
    QGraphicsScene *scene = new QGraphicsScene(0, 0, SCENE_WIDTH, SCENE_HEIGHT);
    view->setScene(scene);
    // draw points to screen all at once
    render_points(scene, points);
    view->scale(1, -1); //screen y-axis is inverted
    view->resize(view->sizeHint());
    view->setWindowTitle("Brute Force Pattern Recognition");
    view->show();

    // sort points by natural order
    // makes finding endpoints of line segments easy
    sort(points.begin(), points.end());
    auto begin = chrono::high_resolution_clock::now();

    //for each point int points
    for (unsigned int i=0;i<points.size();i++){
        //we create a map with the slope as key and a vector of points as value
        map<double,vector<Point>> point_map;

        //for each point that is "larger" than the point we takes its slopes to the current point i
        for (unsigned int j=i+1;j<points.size();++j){
            //and adds it to the maps in the vector with all other points with the same slope
            point_map[points.at(i).slopeTo(points.at(j))].push_back(points.at(j));
        }
        //after this we iterate through each key in the map
        for (map<double,vector<Point>>::iterator it=point_map.begin(); it!=point_map.end(); ++it){
            //and check if the vector has a size of of three or more
            if (it->second.size()>=3){
                //creates a line
                render_line(scene,points.at(i),*(it->second.end()-1));
                //process the event
                a.processEvents();
            }
        }


    }


    auto end = chrono::high_resolution_clock::now();
    cout << "Computing line segments took "
         << std::chrono::duration_cast<chrono::milliseconds>(end - begin).count()
         << " milliseconds." << endl;

    return a.exec(); // start Qt event loop
}
