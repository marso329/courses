/* An fun game for the whole family with alot of tiles and fun by marso329 and joher316
*/

#include "TileList.h"

TileList::TileList()
{
}

//deletes the array
TileList::~TileList()
{
    delete[] m_elements;
}

//checks if the array needs to be increased in size and then adds a tile to the end
void TileList::addTile(Tile tile)
{
    checkResize();
    m_elements[m_size] = tile;
    m_size++;
}
//loops through all tiles in the array and draws it to the scene element
const void TileList::drawAll(QGraphicsScene* scene)
{
    for (int i=0;i<m_size;++i){
        m_elements[i].draw(scene);
    }
}

//goes through the array and keeps tracks of the last element in the array that is on the x,y coordinates and then returns the o-index of that tile
const int TileList::indexOfTopTile(int x, int y)
{
    int temp=-1;
    for (int i=0;i<m_size;++i){
        if (m_elements[i].contains(x,y)){
            temp=i;
        }
    }
    return temp;
}

//gets the index of the top tile from indexOfTopTile and adds it to the end.
//Then moves all the other elements between the orginal position and the last postition down one step
void TileList::raise(int x, int y)
{
    int temp=indexOfTopTile(x,y);
    if (temp>-1){
        addTile(m_elements[temp]);
     for (int i=temp;i<m_size;++i){
            m_elements[i]=m_elements[i+1];
        }
    }

}
//takes out the element to lower and loops through the element from the original position and the 0-position and moves them up and step and then
//put the element in the beginning
void TileList::lower(int x, int y)
{
    int temp=indexOfTopTile(x,y);
    if (temp>-1){
        Tile temp_tile=m_elements[temp];
     for (int i=temp-1;i>=0;--i){
            m_elements[i+1]=m_elements[i];
        }
     m_elements[0]=temp_tile;
    }

}

//removes the top tile on the x,y coordinate and moves all elements above that element down and decreases the size int
void TileList::remove(int x, int y)
{
    int temp=indexOfTopTile(x,y);
    if (temp>-1){
     for (int i=temp;i<m_size;++i){
            m_elements[i]=m_elements[i+1];
        }
     m_size--;
    }
}

//removes all the tiles on the z,y coordinate by using remove and checking if the size of the array changes when it uses remove
void TileList::removeAll(int x, int y)
{
    int temp =0;
    while(temp!=m_size){
        temp=m_size;
        remove(x,y);
    }
}
//controls if the array needs to be resized and if that is the case it creates a new array with double the size and moves all the
//elements to the new array and deletes the old
void TileList::checkResize() {
    if (m_size == m_capacity) {
        // out of space; resize
        Tile* bigDaddy = new Tile[m_capacity * 2];
        for (int i = 0; i < m_size; i++) {
            bigDaddy[i] = m_elements[i];
        }
        delete[] m_elements;   // free old array's memory
        m_elements = bigDaddy;
        m_capacity *= 2;
    }
}
