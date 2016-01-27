/*An implementation of a tilelist which stores and manipulates tiles stored in a array by Marso320 and Joher316
 */

#ifndef TILELIST_H
#define TILELIST_H

#include <QGraphicsScene>
#include "Tile.h"

class TileList {
public:
    //constructor
    TileList();
    //destructor
    ~TileList();
    //adds a tile to the end of the TileList
    void addTile(Tile tile);
    //draws a the tiles in the TileList
    const void drawAll(QGraphicsScene* scene);
    //returns the 0-index of the top tile which is on the x,y coordinates
    const int indexOfTopTile(int x, int y);
    //lowers the top tile on the x,y coordinates the the beginning of the TileList
    void lower(int x, int y);
    //raises the top tile on the x,y coordinates to the end of the TileList
    void raise(int x, int y);
    //removes the the top tile of the x,y coordinates
    void remove(int x, int y);
    //removes all the tiles on the x,y coordinates
    void removeAll(int x, int y);

private:
    int m_size  = 0;                          // number of elements added
    int m_capacity = 10;                     // length of array
    Tile* m_elements = new Tile[m_capacity];   // array of elements
    void checkResize();
};

#endif // TILELIST_H
