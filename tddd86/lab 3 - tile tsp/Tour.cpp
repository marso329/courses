/*An implementation of a circular linked list that uses the Node class as linkes. By marso329 and joher316
 */

#include <iostream>
#include "Tour.h"
#include "Node.h"
#include "Point.h"

//constructor
Tour::Tour()
{
    Point p(-0.1, -0.1);
    startNode=new Node(p,startNode);
}

//destructor
Tour::~Tour()
{
    Node* temp;
    while (startNode->next!=startNode){
        temp=startNode->next;
        startNode->next=temp->next;
        delete[] temp;
    }
    delete[] startNode;
}

//Tours the circular linked list
const void Tour::show()
{
    if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
        Node* temp;
        temp=startNode->next;
        cout<<startNode->point;
        while (temp!=startNode){
            cout<<temp->point;
            temp=temp->next;
        }}
}

//Draws all the points and lines between the points in the circular linkes list
const void Tour::draw(QGraphicsScene *scene)
{
    if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
        Node* temp;
        temp=startNode->next;
        startNode->point.draw(scene);
        startNode->point.drawTo(temp->point,scene);
        while (temp!=startNode){
            temp->point.draw(scene);
            temp->point.drawTo(temp->next->point,scene);
            temp=temp->next;
        }}
}

//returns the size of the linked list
const int Tour::size()
{
    int tempInt=0;
    if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
        Node* temp;
        temp=startNode->next;
        tempInt++;
        while (temp!=startNode){
            tempInt++;
            temp=temp->next;
        }}
    return tempInt;
}

//returns the distance of the circular linked list
const double Tour::distance()
{
    {
        double tempDouble=0.0;
        if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
            Node* temp;
            temp=startNode->next;
            tempDouble=startNode->point.distanceTo(temp->point);
            while (temp!=startNode){
                tempDouble=tempDouble+temp->point.distanceTo(temp->next->point);
                temp=temp->next;
            }}
        return tempDouble;
    }
}

//Inserst a point after the point which is it closest to
void Tour::insertNearest(Point p)
{

    if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
        Node* temp=startNode->next;
        Node* nearest=startNode;
        double nearestDistance=startNode->point.distanceTo(p);
        double tempDistance=0.0;
        while (temp!=startNode){
            tempDistance=temp->point.distanceTo(p);
            if (tempDistance<nearestDistance){
                nearest=temp;
                nearestDistance=tempDistance;
            }
            temp=temp->next;
        }
        temp=new Node(p,nearest->next);
        nearest->next=temp;

    }
    else{
        delete[] startNode;
        startNode=new Node(p,startNode);
    }

}

//inserts the point where it resolves in the smallest amount of increase in total distance
void Tour::insertSmallest(Point p)
{
    if (startNode->point.x!=-0.1 && startNode->point.y!=-0.1){
        Node* temp=startNode->next;
        Node* nearest=startNode;
        double shorestIncrease=startNode->point.distanceTo(p)+p.distanceTo(startNode->next->point)-startNode->point.distanceTo(startNode->next->point);
        double tempDistance=0.0;
        while (temp!=startNode){
            tempDistance=temp->point.distanceTo(p)+p.distanceTo(temp->next->point)-temp->point.distanceTo(temp->next->point);
            if (tempDistance<shorestIncrease){
                nearest=temp;
                shorestIncrease=tempDistance;
            }
            temp=temp->next;
        }
        temp=new Node(p,nearest->next);
        nearest->next=temp;

    }
    else{
        delete[] startNode;
        startNode=new Node(p,startNode);
    }
}
